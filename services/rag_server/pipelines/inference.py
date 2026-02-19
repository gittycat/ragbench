"""
RAG Inference Pipeline

Complete flow for query processing and answer generation:
1. Initialize chat memory (session-based, PostgreSQL-backed)
2. Build hybrid retriever (pg_textsearch BM25 + ChromaDB with RRF fusion)
3. Create reranker postprocessor (optional)
4. Build chat engine (condense_plus_context mode)
5. Query processing (retrieval → reranking → LLM generation)
6. Source extraction and response formatting
"""

from typing import Dict, List, Optional, Generator
import json
import logging
import time
import uuid

from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

from infrastructure.config.models_config import get_models_config
from infrastructure.llm.prompts import get_system_prompt, get_context_prompt, get_condense_prompt

logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL STATE
# ============================================================================

# PostgreSQL-backed chat store (persists across container restarts)
_chat_store = None

# Cache of memory buffers per session
_memory_cache: Dict[str, ChatMemoryBuffer] = {}

# Temporary session cache (in-memory only, cleared on restart)
_temporary_sessions: Dict[str, ChatMemoryBuffer] = {}

# Token counting handler (global, reset before each query)
_token_counter: Optional[TokenCountingHandler] = None


def _get_token_counter() -> TokenCountingHandler:
    """Get or initialize the global token counter."""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCountingHandler(verbose=False)
        Settings.callback_manager = CallbackManager([_token_counter])
        logger.info("[TOKEN] Initialized TokenCountingHandler")
    return _token_counter


def reset_token_counter() -> None:
    """Reset token counts before a new query."""
    counter = _get_token_counter()
    counter.reset_counts()


def get_token_counts() -> Dict[str, int]:
    """Get current token counts from the handler."""
    counter = _get_token_counter()
    return {
        "prompt_tokens": counter.prompt_llm_token_count,
        "completion_tokens": counter.completion_llm_token_count,
        "total_tokens": counter.total_llm_token_count,
    }


# ============================================================================
# CONFIGURATION
# ============================================================================

def get_inference_config() -> Dict:
    """Get inference configuration from models config"""
    config = get_models_config()
    return {
        'reranker_enabled': config.reranker.enabled,
        'reranker_model': config.reranker.model,
        'reranker_top_n': config.reranker.top_n,
        'retrieval_top_k': config.retrieval.top_k,
        'hybrid_search_enabled': config.retrieval.enable_hybrid_search,
        'rrf_k': config.retrieval.rrf_k
    }


# ============================================================================
# STEP 1: CHAT MEMORY MANAGEMENT (PostgreSQL-backed)
# ============================================================================

def _get_chat_store():
    """Get or initialize PostgreSQL chat store (lazy initialization)"""
    global _chat_store
    if _chat_store is None:
        from infrastructure.database.sessions import PostgresChatStore
        _chat_store = PostgresChatStore()
        logger.info("[CHAT] Initialized PostgresChatStore (persistent)")
    return _chat_store


def _get_token_limit_for_chat_history() -> int:
    """
    Calculate token limit for chat history based on LLM context window.

    Strategy:
    - Reserve ~50% for chat history
    - Reserve ~40% for retrieved context
    - Reserve ~10% for response generation
    - Fallback to 3000 tokens if introspection unavailable
    """
    try:
        llm = Settings.llm

        # Try to get context window from LLM metadata
        if hasattr(llm, 'metadata') and hasattr(llm.metadata, 'context_window'):
            context_window = llm.metadata.context_window
            token_limit = int(context_window * 0.5)
            logger.info(f"[CHAT] Detected context window: {context_window} tokens, allocating {token_limit} for history (50%)")
            return token_limit

        # Try direct attribute
        if hasattr(llm, 'context_window'):
            context_window = llm.context_window
            token_limit = int(context_window * 0.5)
            logger.info(f"[CHAT] Detected context window: {context_window} tokens, allocating {token_limit} for history (50%)")
            return token_limit

    except Exception as e:
        logger.warning(f"[CHAT] Could not introspect model context window: {e}")

    # Fallback to safe default
    default_limit = 3000
    logger.info(f"[CHAT] Using default token limit: {default_limit}")
    return default_limit


def get_or_create_chat_memory(
    session_id: str,
    is_temporary: bool = False,
    ensure_metadata: bool = True,
) -> ChatMemoryBuffer:
    """
    Get or create chat memory for a session.

    If is_temporary=True:
    - Use in-memory dict (not PostgreSQL)
    - Lost on server restart
    - No metadata tracking

    Otherwise:
    - Uses PostgreSQL-backed storage (persistent)
    - Session metadata tracked
    """
    from services.session import get_session_metadata, create_session_metadata

    # Handle temporary sessions
    if is_temporary:
        if session_id in _temporary_sessions:
            logger.debug(f"[CHAT] Using temporary memory for session: {session_id}")
            return _temporary_sessions[session_id]

        # Create new temporary memory (no persistence)
        token_limit = _get_token_limit_for_chat_history()
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=token_limit,
            chat_store=None,  # No persistence
            chat_store_key=session_id
        )

        _temporary_sessions[session_id] = memory
        logger.info(f"[CHAT] Created temporary memory for session: {session_id} (in-memory only)")
        return memory

    # Check cache first
    if session_id in _memory_cache:
        logger.debug(f"[CHAT] Using cached memory for session: {session_id}")
        return _memory_cache[session_id]

    # Lazy-create metadata if missing (for existing sessions)
    if ensure_metadata:
        metadata = get_session_metadata(session_id)
        if not metadata:
            logger.info(f"[CHAT] Lazy-creating metadata for existing session: {session_id}")
            create_session_metadata(session_id)

    # Create new memory buffer with PostgreSQL chat store
    token_limit = _get_token_limit_for_chat_history()
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=token_limit,
        chat_store=_get_chat_store(),
        chat_store_key=session_id
    )

    # Cache it
    _memory_cache[session_id] = memory
    logger.info(f"[CHAT] Created new memory for session: {session_id} (token_limit={token_limit})")

    return memory


def clear_session_memory(session_id: str) -> None:
    """Clear chat history for a session from PostgreSQL."""
    # Remove from cache
    if session_id in _memory_cache:
        del _memory_cache[session_id]

    # Clear from PostgreSQL
    chat_store = _get_chat_store()
    messages = chat_store.get_messages(session_id)
    if messages:
        chat_store.delete_messages(session_id)
        logger.info(f"[CHAT] Cleared memory for session: {session_id}")


def get_chat_history(session_id: str) -> List:
    """Get full chat history for a session."""
    chat_store = _get_chat_store()
    messages = chat_store.get_messages(session_id)
    return messages if messages else []


# ============================================================================
# STEP 2: HYBRID RETRIEVAL (pg_textsearch BM25 + ChromaDB + RRF)
# ============================================================================

def create_hybrid_retriever(index: VectorStoreIndex, similarity_top_k: int = 10):
    """
    Create hybrid retriever combining pg_textsearch BM25 + ChromaDB with RRF fusion.

    Research (Pinecone benchmark):
    - 48% improvement in retrieval quality vs vector-only
    - BM25 excels at: exact keywords, IDs, names, abbreviations
    - Vector search excels at: semantic understanding, context
    - RRF (Reciprocal Rank Fusion): simple, robust fusion

    Returns None if hybrid search disabled (falls back to vector-only).
    """
    config = get_inference_config()

    if not config['hybrid_search_enabled']:
        logger.info("[HYBRID] Hybrid search disabled - using vector-only retrieval")
        return None

    logger.info(f"[HYBRID] Creating hybrid retriever (top_k={similarity_top_k}, rrf_k={config['rrf_k']})")

    from infrastructure.search.hybrid_retriever import create_hybrid_retriever as _create_hybrid

    retriever = _create_hybrid(
        vector_index=index,
        similarity_top_k=similarity_top_k,
        rrf_k=config['rrf_k'],
    )

    logger.info("[HYBRID] Hybrid retriever created (pg_textsearch BM25 + ChromaDB + RRF)")
    return retriever


# ============================================================================
# STEP 3: RERANKING
# ============================================================================

def create_reranker_postprocessor() -> Optional[List]:
    """
    Create reranker postprocessor for improving retrieval quality.

    Flow:
    1. Check if reranking enabled in config
    2. Create SentenceTransformerRerank with cross-encoder model
    3. Configure top_n (typically 5-7 nodes, or half of retrieval_top_k)

    Reranker uses cross-encoder model to score query-document pairs.
    This is more accurate than bi-encoder embeddings but slower.
    Model downloads on first use (~80MB, adds ~100-300ms latency).

    Returns None if reranking disabled.
    """
    config = get_inference_config()

    if not config['reranker_enabled']:
        logger.info("[RERANKER] Reranking disabled")
        return None

    logger.info(f"[RERANKER] Initializing reranker: {config['reranker_model']}")

    # Calculate top_n: return best reranked nodes (usually half of retrieved, min 5)
    top_n = max(5, config['retrieval_top_k'] // 2)
    logger.info(f"[RERANKER] Returning top {top_n} nodes after reranking")

    postprocessors = [
        SentenceTransformerRerank(
            model=config['reranker_model'],
            top_n=top_n
        )
    ]
    logger.info("[RERANKER] Postprocessor initialized")
    return postprocessors


def ensure_reranker_model_cached() -> None:
    """Fail fast if the reranker model is not present in the local HF cache."""
    config = get_inference_config()

    if not config['reranker_enabled']:
        return

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    model = config['reranker_model']
    try:
        snapshot_download(model, local_files_only=True)
    except LocalEntryNotFoundError as e:
        logger.error(
            "[STARTUP] Reranker model missing in local cache. "
            "Run `just init` to download '%s' into ./.hf_cache, then restart.",
            model,
        )
        raise RuntimeError(
            "[STARTUP] Reranker model not found in local cache. "
            f"Run `just init` to download '{model}' into ./.hf_cache, "
            "then restart the service."
        ) from e


# ============================================================================
# STEP 4: CHAT ENGINE CREATION
# ============================================================================

def create_chat_engine(
    index: VectorStoreIndex,
    session_id: str,
    retrieval_top_k: int = 10,
    is_temporary: bool = False,
    ensure_metadata: bool = True,
) -> CondensePlusContextChatEngine:
    """
    Create chat engine with hybrid search, reranking, and conversational memory.

    Flow:
    1. Load LlamaIndex native prompts (system, context, condense)
    2. Get or create chat memory for session
    3. Create hybrid retriever (or None for vector-only fallback)
    4. Create reranker postprocessor (if enabled)
    5. Build CondensePlusContextChatEngine with all components

    CondensePlusContextChatEngine mode:
    - Condenses chat history + new query into standalone question
    - Retrieves relevant context
    - Generates answer using context + chat history

    Returns configured chat engine ready for querying.
    """
    config = get_inference_config()

    logger.info(f"[CHAT_ENGINE] Creating chat engine for session: {session_id}")
    logger.info(f"[CHAT_ENGINE] Config: top_k={retrieval_top_k}, reranker={config['reranker_enabled']}, hybrid={config['hybrid_search_enabled']}")

    # Get prompts
    system_prompt = get_system_prompt()
    include_citations = False
    citation_format = "numeric"
    try:
        from infrastructure.config.models_config import get_models_config
        eval_config = get_models_config().eval
        include_citations = eval_config.citation_scope == "explicit"
        citation_format = eval_config.citation_format
    except Exception:
        include_citations = False
    context_prompt = get_context_prompt(
        include_citations=include_citations,
        citation_format=citation_format,
    )
    condense_prompt = get_condense_prompt()  # None = use LlamaIndex default

    # Get chat memory (with temporary support)
    memory = get_or_create_chat_memory(
        session_id,
        is_temporary=is_temporary,
        ensure_metadata=ensure_metadata,
    )

    # Create retriever (hybrid or vector-only)
    retriever = create_hybrid_retriever(index, similarity_top_k=retrieval_top_k)

    # Create chat engine
    if retriever is not None:
        logger.info("[CHAT_ENGINE] Using hybrid retriever (pg_textsearch BM25 + ChromaDB + RRF)")
        chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            memory=memory,
            node_postprocessors=create_reranker_postprocessor(),
            system_prompt=system_prompt,
            context_prompt=context_prompt,
            condense_prompt=condense_prompt,
            verbose=False
        )
    else:
        logger.info("[CHAT_ENGINE] Using vector retriever only")
        chat_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=memory,
            similarity_top_k=retrieval_top_k,
            node_postprocessors=create_reranker_postprocessor(),
            system_prompt=system_prompt,
            context_prompt=context_prompt,
            condense_prompt=condense_prompt,
            verbose=False
        )

    logger.info("[CHAT_ENGINE] Chat engine created successfully")
    return chat_engine


# ============================================================================
# STEP 5: QUERY PROCESSING
# ============================================================================

def extract_sources(
    source_nodes: List,
    include_chunks: bool = False,
    dedupe_by_document: bool = True,
) -> List[Dict]:
    """
    Extract source information from retrieved nodes.

    Flow:
    - Deduplicate by document_id (multiple chunks from same doc)
    - Extract metadata: document_id, file_name, path, score
    - Create excerpt (first 200 chars) and include full text
    - Return list of source dictionaries
    """
    sources = []
    seen_docs = set()

    for node in source_nodes:
        metadata = node.metadata
        doc_id = metadata.get('document_id')
        chunk_index = metadata.get('chunk_index')
        chunk_id = getattr(node, "id_", None)

        # Deduplicate by document_id
        if dedupe_by_document and doc_id:
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)

        full_text = node.get_content()
        source = {
            'document_id': doc_id,
            'document_name': metadata.get('file_name', 'Unknown'),
            'excerpt': full_text[:200] + '...' if len(full_text) > 200 else full_text,
            'full_text': full_text,
            'path': metadata.get('path', ''),
            'score': node.score if hasattr(node, 'score') and node.score else None
        }
        if include_chunks:
            source["chunk_id"] = chunk_id
            source["chunk_index"] = chunk_index
        sources.append(source)

    return sources


def extract_numeric_citations(answer: str, sources: List[Dict]) -> List[Dict]:
    """Extract numeric citations like [1], [1,2], [1-3] mapped to sources list."""
    import re

    if not answer or not sources:
        return []

    citation_indices: list[int] = []
    bracket_patterns = [
        r"\[(\d+(?:\s*-\s*\d+)?(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*)\]",
        r"\((\d+(?:\s*-\s*\d+)?(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*)\)",
    ]

    for pattern in bracket_patterns:
        for match in re.findall(pattern, answer):
            parts = [p.strip() for p in match.split(",")]
            for part in parts:
                if "-" in part:
                    bounds = [b.strip() for b in part.split("-", 1)]
                    if len(bounds) == 2 and bounds[0].isdigit() and bounds[1].isdigit():
                        start = int(bounds[0])
                        end = int(bounds[1])
                        if start <= end:
                            citation_indices.extend(range(start, end + 1))
                elif part.isdigit():
                    citation_indices.append(int(part))

    # De-duplicate while preserving order
    seen = set()
    unique_indices = []
    for idx in citation_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    citations = []
    for idx in unique_indices:
        source_idx = idx - 1
        if 0 <= source_idx < len(sources):
            source = sources[source_idx]
            citations.append(
                {
                    "source_index": idx,
                    "document_id": source.get("document_id"),
                    "chunk_id": source.get("chunk_id"),
                    "chunk_index": source.get("chunk_index"),
                }
            )

    return citations


def query_rag(
    query_text: str,
    session_id: str,
    is_temporary: bool = False,
    include_chunks: bool = False,
    ensure_metadata: bool = True,
    update_session_metadata: bool = True,
) -> Dict:
    """
    Execute RAG query pipeline (synchronous, non-streaming).

    Flow:
    1. Get VectorStoreIndex from ChromaDB
    2. Get inference configuration
    3. Create chat engine (with hybrid search, reranking, memory)
    4. Execute query (retrieval → reranking → LLM generation)
    5. Extract sources from retrieved nodes
    6. Update session metadata (touch timestamp, auto-generate title)
    7. Return response with answer, sources, session_id, metrics

    Returns:
        {
            'answer': str,
            'sources': List[Dict],
            'query': str,
            'session_id': str,
            'citations': List[Dict] | None,
            'metrics': {
                'latency_ms': float,
                'token_usage': {
                    'prompt_tokens': int,
                    'completion_tokens': int,
                    'total_tokens': int
                } | None
            }
        }
    """
    from infrastructure.search.vector_store import get_vector_index
    from services.session import touch_session, get_session_metadata, update_session_title
    from services.session_titles import generate_session_title

    logger.info(f"[QUERY] Processing query for session: {session_id} (temporary={is_temporary})")
    query_start = time.time()

    # Reset token counter before query
    reset_token_counter()

    # Get index and config
    index = get_vector_index()
    config = get_inference_config()

    logger.info(f"[QUERY] Config: top_k={config['retrieval_top_k']}, reranker={config['reranker_enabled']}, hybrid={config['hybrid_search_enabled']}")

    # Create chat engine
    chat_engine = create_chat_engine(
        index,
        session_id,
        retrieval_top_k=config['retrieval_top_k'],
        is_temporary=is_temporary,
        ensure_metadata=ensure_metadata,
    )

    # Execute query
    logger.info(f"[QUERY] Executing RAG query...")
    response = chat_engine.chat(query_text)

    # Capture token usage immediately after query
    token_counts = get_token_counts()

    # Log retrieved nodes
    logger.info(f"[QUERY] Retrieved {len(response.source_nodes)} nodes for context")

    if response.source_nodes:
        total_context_length = 0
        for i, node in enumerate(response.source_nodes):
            node_text = node.get_content()
            total_context_length += len(node_text)
            score_info = f" (score: {node.score:.4f})" if hasattr(node, 'score') and node.score else ""
            logger.debug(f"[QUERY] Node {i+1}{score_info}: {node_text[:150]}...")

        logger.info(f"[QUERY] Total context length: {total_context_length} chars ({len(response.source_nodes)} nodes)")
    else:
        logger.warning("[QUERY] No context nodes retrieved - LLM will respond without context")

    # Extract sources
    sources = extract_sources(
        response.source_nodes,
        include_chunks=include_chunks,
        dedupe_by_document=not include_chunks,
    )
    citations = None
    if include_chunks:
        try:
            from infrastructure.config.models_config import get_models_config

            models_config = get_models_config()
            if models_config.eval.citation_format == "numeric":
                citations = extract_numeric_citations(str(response), sources)
        except Exception:
            citations = None

    # Update session metadata (non-temporary sessions only)
    if update_session_metadata and not is_temporary:
        touch_session(session_id)

        # Auto-generate title from first user message if needed
        metadata = get_session_metadata(session_id)
        if metadata and metadata.title == "New Chat":
            title = generate_session_title(query_text)
            update_session_title(session_id, title)

    query_duration = time.time() - query_start
    latency_ms = query_duration * 1000
    logger.info(f"[QUERY] Query complete ({query_duration:.2f}s) - {len(sources)} sources returned")
    logger.info(f"[QUERY] Token usage: {token_counts}")

    return {
        'answer': str(response),
        'sources': sources,
        'query': query_text,
        'session_id': session_id,
        'citations': citations,
        'metrics': {
            'latency_ms': latency_ms,
            'token_usage': token_counts if token_counts['total_tokens'] > 0 else None,
        },
    }


def query_rag_with_context(
    query_text: str,
    context_passages: List[Dict],
    session_id: Optional[str] = None,
) -> Dict:
    """
    Execute RAG generation with pre-injected context (Tier 1 eval).

    Bypasses hybrid retrieval entirely. Uses the provided passages as context
    and calls the LLM directly via Settings.llm.chat().

    Args:
        query_text: The user's question
        context_passages: List of {"text": str, "doc_id": str} dicts
        session_id: Optional session ID for response tracking

    Returns:
        Same shape as query_rag(): answer, sources, session_id, citations, metrics
    """
    from llama_index.core.llms import ChatMessage

    reset_token_counter()
    query_start = time.time()

    # Format passages as numbered context block
    context_lines = []
    for i, p in enumerate(context_passages, 1):
        context_lines.append(f"[{i}] (source: {p['doc_id']})\n{p['text']}")
    context_str = "\n\n".join(context_lines)

    system_prompt = get_system_prompt()
    user_content = f"Context:\n{context_str}\n\nQuestion: {query_text}"

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_content),
    ]

    logger.info(f"[QUERY_WITH_CONTEXT] Calling LLM with {len(context_passages)} passages")
    response = Settings.llm.chat(messages)
    answer = str(response)

    token_counts = get_token_counts()

    # Build sources from provided passages (no retrieval)
    sources = [
        {
            "document_id": p["doc_id"],
            "document_name": p["doc_id"],
            "excerpt": p["text"][:200] + "..." if len(p["text"]) > 200 else p["text"],
            "full_text": p["text"],
            "path": "",
            "score": None,
        }
        for p in context_passages
    ]

    latency_ms = (time.time() - query_start) * 1000
    logger.info(f"[QUERY_WITH_CONTEXT] Complete ({latency_ms:.0f}ms)")

    return {
        "answer": answer,
        "sources": sources,
        "query": query_text,
        "session_id": session_id or str(uuid.uuid4()),
        "citations": None,
        "metrics": {
            "latency_ms": latency_ms,
            "token_usage": token_counts if token_counts["total_tokens"] > 0 else None,
        },
    }


def query_rag_stream(
    query_text: str,
    session_id: str,
    is_temporary: bool = False,
    include_chunks: bool = False,
    ensure_metadata: bool = True,
    update_session_metadata: bool = True,
) -> Generator[str, None, None]:
    """
    Execute RAG query pipeline with streaming response (Server-Sent Events).

    Flow:
    1. Create chat engine (same as non-streaming)
    2. Use stream_chat for token-by-token generation
    3. Yield SSE events:
       - event: token, data: {"token": "..."}  (for each token)
       - event: sources, data: {"sources": [...], "session_id": "..."}  (after streaming completes)
       - event: done, data: {}  (completion signal)
       - event: error, data: {"error": "..."}  (on error)
    4. Update session metadata after streaming completes

    Yields SSE-formatted strings for client consumption.
    """
    from infrastructure.search.vector_store import get_vector_index
    from services.session import touch_session, get_session_metadata, update_session_title
    from services.session_titles import generate_session_title

    try:
        logger.info(f"[QUERY_STREAM] Starting streaming query for session: {session_id} (temporary={is_temporary})")

        # Get index and create chat engine
        index = get_vector_index()
        config = get_inference_config()
        chat_engine = create_chat_engine(
            index,
            session_id,
            retrieval_top_k=config['retrieval_top_k'],
            is_temporary=is_temporary,
            ensure_metadata=ensure_metadata,
        )

        # Stream response tokens
        logger.info(f"[QUERY_STREAM] Executing streaming RAG query...")
        streaming_response = chat_engine.stream_chat(query_text)

        for token in streaming_response.response_gen:
            yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

        # After streaming completes, send sources
        sources = extract_sources(
            streaming_response.source_nodes,
            include_chunks=include_chunks,
            dedupe_by_document=not include_chunks,
        )
        citations = None
        if include_chunks:
            try:
                from infrastructure.config.models_config import get_models_config

                models_config = get_models_config()
                if models_config.eval.citation_format == "numeric":
                    citations = extract_numeric_citations(str(streaming_response), sources)
            except Exception:
                citations = None
        logger.info(f"[QUERY_STREAM] Streaming complete - {len(sources)} sources")

        # Update session metadata (non-temporary sessions only)
        if update_session_metadata and not is_temporary:
            touch_session(session_id)

            # Auto-generate title from first user message if needed
            metadata = get_session_metadata(session_id)
            if metadata and metadata.title == "New Chat":
                title = generate_session_title(query_text)
                update_session_title(session_id, title)

        yield f"event: sources\ndata: {json.dumps({'sources': sources, 'citations': citations, 'session_id': session_id})}\n\n"

        # Send done event
        yield f"event: done\ndata: {{}}\n\n"

    except Exception as e:
        logger.error(f"[QUERY_STREAM] Error during streaming: {str(e)}")
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

def refresh_bm25_retriever(index: VectorStoreIndex = None) -> None:
    """No-op for backward compatibility. pg_textsearch BM25 index refreshes automatically."""
    logger.debug("[HYBRID] refresh_bm25_retriever called - pg_textsearch handles this automatically")
    pass


def initialize_bm25_retriever(index: VectorStoreIndex = None, similarity_top_k: int = 10):
    """No-op for backward compatibility. pg_textsearch BM25 is used instead."""
    logger.debug("[HYBRID] initialize_bm25_retriever called - pg_textsearch handles this automatically")
    return None


def get_bm25_retriever():
    """No-op for backward compatibility."""
    return None
