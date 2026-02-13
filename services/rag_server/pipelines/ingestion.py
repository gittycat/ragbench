"""
Document Ingestion Pipeline

Complete flow for processing documents from upload to indexing:
1. Validate file format and extract metadata
2. Chunk document using Docling (complex) or SentenceSplitter (text)
3. Optionally add contextual prefixes via LLM (Anthropic method)
4. Generate embeddings and index in ChromaDB
5. BM25 index is automatic via pg_textsearch
"""

from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime, timezone
import time
import hashlib
import logging

from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex

from infrastructure.config.models_config import get_models_config
from infrastructure.llm.factory import get_llm_client

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

SUPPORTED_EXTENSIONS = {
    '.txt', '.md', '.pdf', '.docx', '.pptx', '.xlsx',
    '.html', '.htm', '.asciidoc', '.adoc'
}

SIMPLE_TEXT_EXTENSIONS = {'.txt', '.md'}


def get_ingestion_config() -> Dict[str, bool]:
    """Get ingestion configuration from models config"""
    config = get_models_config()
    return {
        'contextual_retrieval_enabled': config.retrieval.enable_contextual_retrieval,
    }


# ============================================================================
# STEP 1: METADATA EXTRACTION
# ============================================================================

def compute_file_hash(file_path: str) -> str:
    """
    Compute SHA256 hash of file content for duplicate detection.
    Matches LlamaIndex's document hashing approach.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def extract_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract basic metadata from file for storage.
    PostgreSQL JSONB supports nested structures, no flattening needed.
    """
    file_path_obj = Path(file_path)
    file_size = file_path_obj.stat().st_size
    file_hash = compute_file_hash(file_path)

    return {
        "file_name": file_path_obj.name,
        "file_type": file_path_obj.suffix,
        "path": str(file_path_obj.parent),
        "file_size_bytes": file_size,
        "file_hash": file_hash
    }


def clean_metadata_for_storage(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean metadata for storage. PostgreSQL JSONB handles most types,
    but we still need to handle non-JSON-serializable types.
    """
    cleaned = {}
    for key, value in metadata.items():
        if value is None or isinstance(value, (str, int, float, bool, dict, list)):
            cleaned[key] = value
        else:
            # Convert other types to string
            logger.debug(f"[METADATA] Converting {key} ({type(value).__name__}) to string")
            cleaned[key] = str(value)

    return cleaned


# Backward compatibility alias
clean_metadata_for_chroma = clean_metadata_for_storage


# ============================================================================
# STEP 2: DOCUMENT CHUNKING
# ============================================================================

def chunk_document_with_docling(file_path: str) -> List[TextNode]:
    """
    Process complex documents (PDF, DOCX, etc.) using Docling.

    Flow:
    - DoclingReader extracts structured content (must use JSON export)
    - DoclingNodeParser creates chunks preserving document structure
    - Metadata is cleaned for storage

    Returns list of TextNode objects ready for embedding.
    """
    logger.info(f"[CHUNKING] Using DoclingReader for complex document: {file_path}")

    # CRITICAL: Must use JSON export for DoclingNodeParser compatibility
    reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)

    # Phase 1: Read document structure
    logger.info(f"[CHUNKING] Phase 1: Reading document with Docling...")
    read_start = time.time()
    try:
        documents = reader.load_data(file_path=str(file_path))
        read_duration = time.time() - read_start
        logger.info(f"[CHUNKING] Phase 1 complete ({read_duration:.2f}s) - {len(documents)} documents extracted")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found during processing: {file_path}") from e
    except Exception as e:
        read_duration = time.time() - read_start
        logger.error(f"[CHUNKING] DoclingReader failed after {read_duration:.2f}s: {str(e)}")
        raise ValueError(f"Failed to process document {file_path}: {str(e)}") from e

    if not documents:
        raise ValueError(f"Could not load document: {file_path}")

    # Phase 2: Parse into chunks
    logger.info(f"[CHUNKING] Phase 2: Parsing into chunks...")
    parse_start = time.time()
    try:
        node_parser = DoclingNodeParser()
        nodes = node_parser.get_nodes_from_documents(documents)
        parse_duration = time.time() - parse_start
        logger.info(f"[CHUNKING] Phase 2 complete ({parse_duration:.2f}s) - {len(nodes)} chunks created")
    except Exception as e:
        parse_duration = time.time() - parse_start
        logger.error(f"[CHUNKING] DoclingNodeParser failed after {parse_duration:.2f}s: {str(e)}")
        raise ValueError(f"Failed to parse document into chunks: {str(e)}") from e

    # Clean metadata for storage
    logger.info(f"[CHUNKING] Cleaning metadata for storage")
    for node in nodes:
        node.metadata = clean_metadata_for_storage(node.metadata)

    return nodes


def chunk_document_with_text_splitter(file_path: str, chunk_size: int = 500) -> List[TextNode]:
    """
    Process simple text documents using SentenceSplitter.

    Flow:
    - SimpleDirectoryReader loads text file
    - SentenceSplitter creates semantic chunks with overlap

    Returns list of TextNode objects ready for embedding.
    """
    logger.info(f"[CHUNKING] Using SimpleDirectoryReader for text file: {file_path}")

    # Phase 1: Load text file
    logger.info(f"[CHUNKING] Phase 1: Loading text file...")
    try:
        reader = SimpleDirectoryReader(input_files=[str(file_path)])
        documents = reader.load_data()
        logger.info(f"[CHUNKING] Phase 1 complete - {len(documents)} documents loaded")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found during processing: {file_path}") from e
    except Exception as e:
        raise ValueError(f"Failed to read file {file_path}: {str(e)}") from e

    if not documents:
        raise ValueError(f"Could not load document: {file_path}")

    # Phase 2: Split into chunks
    logger.info(f"[CHUNKING] Phase 2: Splitting into chunks (chunk_size={chunk_size})...")
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info(f"[CHUNKING] Phase 2 complete - {len(nodes)} chunks created")

    return nodes


def chunk_document(file_path: str, chunk_size: int = 500) -> List[TextNode]:
    """
    Main chunking dispatcher - routes to appropriate chunking method based on file type.

    Returns list of TextNode objects ready for contextual enrichment and embedding.
    """
    file_path_obj = Path(file_path)
    extension = file_path_obj.suffix.lower()

    # Validate file exists
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    # Validate file type
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    logger.info(f"[CHUNKING] Starting chunking for {file_path_obj.name} (type: {extension})")

    # Route to appropriate chunker
    if extension in SIMPLE_TEXT_EXTENSIONS:
        nodes = chunk_document_with_text_splitter(file_path, chunk_size)
    else:
        nodes = chunk_document_with_docling(file_path)

    # Log preview of first chunk
    if nodes:
        first_text = nodes[0].get_content()
        preview = first_text[:80] + "..." if len(first_text) > 80 else first_text
        logger.info(f"[CHUNKING] First chunk preview: {preview}")

    logger.info(f"[CHUNKING] Chunking complete - {len(nodes)} chunks created from {file_path_obj.name}")
    return nodes


# ============================================================================
# STEP 3: CONTEXTUAL RETRIEVAL (OPTIONAL)
# ============================================================================

def add_contextual_prefix_to_chunk(node: TextNode, document_name: str, document_type: str) -> TextNode:
    """
    Add LLM-generated contextual prefix to chunk (Anthropic method).

    Research (Anthropic 2024): 49% reduction in retrieval failures
    Combined with hybrid search + reranking: 67% reduction

    Flow:
    - Extract chunk preview (first 400 chars)
    - Send to LLM with prompt for 1-2 sentence context
    - Prepend context to original chunk text
    - Return enhanced node (or original if LLM fails)
    """
    from infrastructure.llm import get_contextual_prefix_prompt

    logger.info(f"[CONTEXTUAL] Generating contextual prefix for chunk via LLM...")
    start_time = time.time()

    chunk_preview = node.get_content()[:400]
    prompt = get_contextual_prefix_prompt(document_name, document_type, chunk_preview)

    try:
        llm = get_llm_client()
        llm_start = time.time()
        response = llm.complete(prompt)
        llm_duration = time.time() - llm_start

        context = response.text.strip()

        # Prepend context to original text
        enhanced_text = f"{context}\n\n{node.text}"
        node.text = enhanced_text

        total_duration = time.time() - start_time
        logger.info(f"[CONTEXTUAL] LLM call completed in {llm_duration:.2f}s (total: {total_duration:.2f}s)")
        logger.debug(f"[CONTEXTUAL] Added prefix: {context[:80]}...")
        return node

    except Exception as e:
        duration = time.time() - start_time
        logger.warning(f"[CONTEXTUAL] Failed to generate context after {duration:.2f}s: {e}")
        # Return original node if context generation fails
        return node


def add_contextual_retrieval(nodes: List[TextNode], file_path: str) -> List[TextNode]:
    """
    Add contextual prefixes to all chunks using LLM (if enabled).

    This is the most time-consuming step (~85% of processing time).
    Each chunk requires a separate LLM call.

    Returns enhanced nodes with contextual prefixes prepended.
    """
    config = get_ingestion_config()

    if not config['contextual_retrieval_enabled']:
        logger.info("[CONTEXTUAL] Contextual retrieval disabled - skipping")
        return nodes

    if not nodes:
        return nodes

    file_path_obj = Path(file_path)
    extension = file_path_obj.suffix.lower()

    logger.info(f"[CONTEXTUAL] Starting contextual prefix generation for {len(nodes)} chunks")
    logger.info(f"[CONTEXTUAL] This will make {len(nodes)} LLM calls - estimated time: ~{len(nodes) * 2}s")

    contextual_start = time.time()
    enhanced_nodes = []

    for i, node in enumerate(nodes):
        # Log progress every 10 chunks
        if i % 10 == 0 and i > 0:
            elapsed = time.time() - contextual_start
            avg_per_node = elapsed / i
            est_remaining = avg_per_node * (len(nodes) - i)
            logger.info(f"[CONTEXTUAL] Progress: {i}/{len(nodes)} - Elapsed: {elapsed:.1f}s, Est. remaining: {est_remaining:.1f}s")

        enhanced_node = add_contextual_prefix_to_chunk(node, file_path_obj.name, extension)
        enhanced_nodes.append(enhanced_node)

    contextual_duration = time.time() - contextual_start
    avg_per_node = contextual_duration / len(nodes)
    logger.info(f"[CONTEXTUAL] Contextual prefixes complete ({contextual_duration:.2f}s, avg: {avg_per_node:.2f}s per chunk)")

    return enhanced_nodes


# ============================================================================
# STEP 4: EMBEDDING & INDEXING
# ============================================================================

def add_document_metadata_to_chunks(
    nodes: List[TextNode],
    document_id: str,
    file_metadata: Dict[str, Any],
    uploaded_at: Optional[str] = None
) -> List[TextNode]:
    """
    Add document-level metadata and IDs to all chunks.

    Each chunk gets:
    - All file metadata (name, type, size, hash, path)
    - document_id for tracking and deletion
    - chunk_index for ordering
    - uploaded_at timestamp (ISO 8601 format)
    - Unique node ID: {document_id}-chunk-{index}

    Args:
        uploaded_at: ISO 8601 timestamp representing when document processing
                     completed and the document was ingested into the vector db.
                     If not provided, uses current UTC time.
    """
    if uploaded_at is None:
        uploaded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for i, node in enumerate(nodes):
        node.metadata.update(file_metadata)
        node.metadata["chunk_index"] = i
        node.metadata["document_id"] = document_id
        node.metadata["uploaded_at"] = uploaded_at
        node.id_ = f"{document_id}-chunk-{i}"

    logger.info(f"[METADATA] Added metadata to {len(nodes)} chunks (document_id={document_id}, uploaded_at={uploaded_at})")
    return nodes


def embed_and_index_chunks(
    index: VectorStoreIndex,
    nodes: List[TextNode],
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> None:
    """
    Generate embeddings and index chunks in ChromaDB.

    Flow:
    - For each chunk:
      - Generate embedding via Ollama (or configured provider)
      - Insert into ChromaDB vector store
      - Call progress callback for tracking
    - Includes retry logic for Ollama connection errors

    This is the second most time-consuming step (~15% of processing time).
    """
    logger.info(f"[EMBEDDING] Starting embedding generation for {len(nodes)} chunks")

    if nodes:
        first_text = nodes[0].get_content()
        preview = first_text[:100] + "..." if len(first_text) > 100 else first_text
        logger.info(f"[EMBEDDING] First chunk preview: {preview}")

    embedding_start = time.time()
    total_nodes = len(nodes)

    for i, node in enumerate(nodes, 1):
        node_start = time.time()
        logger.info(f"[EMBEDDING] Embedding chunk {i}/{total_nodes}...")

        try:
            # Retry logic for Ollama connection errors
            _insert_node_with_retry(index, node, max_retries=3, base_delay=2.0)
        except Exception as e:
            raise Exception(f"Failed to embed chunk {i}/{total_nodes}: {str(e)}") from e

        node_duration = time.time() - node_start
        elapsed = time.time() - embedding_start
        avg_per_node = elapsed / i
        est_remaining = avg_per_node * (total_nodes - i)

        logger.info(f"[EMBEDDING] Chunk {i}/{total_nodes} embedded ({node_duration:.2f}s) - Elapsed: {elapsed:.1f}s, Est. remaining: {est_remaining:.1f}s")

        if progress_callback:
            progress_callback(i, total_nodes)

    total_duration = time.time() - embedding_start
    avg_per_node = total_duration / len(nodes)
    logger.info(f"[EMBEDDING] Embedding complete ({total_duration:.2f}s, avg: {avg_per_node:.2f}s per chunk)")


def _insert_node_with_retry(index: VectorStoreIndex, node: TextNode, max_retries: int = 3, base_delay: float = 2.0):
    """
    Insert a node with exponential backoff retry for Ollama connection errors.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            index.insert_nodes([node])
            return  # Success
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()

            # Check if it's a connection error
            is_connection_error = any(term in error_msg for term in [
                'eof', 'connection', 'timeout', 'refused', 'unavailable'
            ])

            if is_connection_error and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"[EMBEDDING] Connection error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                logger.info(f"[EMBEDDING] Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                # Not a connection error or last attempt - raise immediately
                raise

    # All retries failed
    raise Exception(f"Failed to embed node after {max_retries} attempts. Last error: {str(last_error)}") from last_error


# ============================================================================
# STEP 5: HYBRID SEARCH INDEX REFRESH (NO-OP for pg_textsearch)
# ============================================================================

def refresh_hybrid_search_index(index: VectorStoreIndex) -> None:
    """
    No-op for pg_textsearch. BM25 index refreshes automatically with inserts.

    pg_textsearch maintains the BM25 index automatically when rows are
    inserted/updated/deleted, so no manual refresh is needed.
    """
    logger.debug("[HYBRID] pg_textsearch BM25 index refreshes automatically - no manual refresh needed")


# ============================================================================
# MAIN INGESTION PIPELINE
# ============================================================================

def ingest_document(
    file_path: str,
    index: VectorStoreIndex,
    document_id: str,
    filename: str,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """
    Complete document ingestion pipeline.

    Flow:
    1. Extract file metadata (hash, size, type)
    2. Chunk document (Docling or SentenceSplitter)
    3. Add contextual prefixes (optional, LLM-based)
    4. Add document metadata to chunks
    5. Generate embeddings and index in ChromaDB
    6. BM25 index auto-refreshes via pg_textsearch

    Args:
        file_path: Path to document file
        index: VectorStoreIndex for ChromaDB
        document_id: Unique document identifier
        filename: Display name for document
        progress_callback: Optional callback for progress tracking (current, total)

    Returns:
        Dictionary with ingestion results:
        {
            'document_id': str,
            'filename': str,
            'chunks': int,
            'status': 'success'
        }
    """
    pipeline_start = time.time()
    logger.info(f"[INGESTION] ========== Starting ingestion pipeline for {filename} ==========")

    # STEP 1: Extract metadata
    logger.info(f"[INGESTION] Step 1: Extracting file metadata...")
    metadata = extract_file_metadata(file_path)
    metadata["file_name"] = filename  # Use provided filename instead of temp file name
    logger.info(f"[INGESTION] Metadata extracted: {metadata}")

    # STEP 2: Chunk document
    logger.info(f"[INGESTION] Step 2: Chunking document...")
    chunk_start = time.time()
    nodes = chunk_document(file_path)
    chunk_duration = time.time() - chunk_start
    logger.info(f"[INGESTION] Step 2 complete ({chunk_duration:.2f}s) - {len(nodes)} chunks created")

    # STEP 3: Add contextual prefixes (optional)
    logger.info(f"[INGESTION] Step 3: Adding contextual retrieval prefixes...")
    contextual_start = time.time()
    nodes = add_contextual_retrieval(nodes, file_path)
    contextual_duration = time.time() - contextual_start
    logger.info(f"[INGESTION] Step 3 complete ({contextual_duration:.2f}s)")

    # STEP 4: Add document metadata
    logger.info(f"[INGESTION] Step 4: Adding document metadata to chunks...")
    nodes = add_document_metadata_to_chunks(nodes, document_id, metadata)
    logger.info(f"[INGESTION] Step 4 complete")

    # Sanitize metadata for ChromaDB (only scalar values allowed)
    for node in nodes:
        node.metadata = {
            k: v for k, v in node.metadata.items()
            if isinstance(v, (str, int, float, bool)) or v is None
        }

    # STEP 5: Embed and index
    logger.info(f"[INGESTION] Step 5: Generating embeddings and indexing in ChromaDB...")
    embed_start = time.time()
    embed_and_index_chunks(index, nodes, progress_callback)
    embed_duration = time.time() - embed_start
    logger.info(f"[INGESTION] Step 5 complete ({embed_duration:.2f}s)")

    # STEP 6: BM25 index auto-refreshes (no-op)
    logger.info(f"[INGESTION] Step 6: BM25 index auto-refreshes via pg_textsearch")
    refresh_hybrid_search_index(index)
    logger.info(f"[INGESTION] Step 6 complete")

    # Summary
    pipeline_duration = time.time() - pipeline_start
    logger.info(f"[INGESTION] ========== Ingestion pipeline complete ({pipeline_duration:.2f}s) ==========")
    logger.info(f"[INGESTION] Performance breakdown:")
    logger.info(f"[INGESTION]   - Chunking: {chunk_duration:.2f}s ({chunk_duration/pipeline_duration*100:.1f}%)")
    logger.info(f"[INGESTION]   - Contextual: {contextual_duration:.2f}s ({contextual_duration/pipeline_duration*100:.1f}%)")
    logger.info(f"[INGESTION]   - Embedding: {embed_duration:.2f}s ({embed_duration/pipeline_duration*100:.1f}%)")
    logger.info(f"[INGESTION]   - Total: {pipeline_duration:.2f}s")

    # Build chunk data for PostgreSQL storage
    chunks_data = []
    for i, node in enumerate(nodes):
        chunks_data.append({
            "chunk_index": i,
            "content": node.get_content(),
            "content_with_context": node.metadata.get("contextual_prefix", ""),
            "metadata": {k: v for k, v in node.metadata.items() if k != "contextual_prefix"},
        })

    return {
        'document_id': document_id,
        'filename': filename,
        'chunks': len(nodes),
        'chunks_data': chunks_data,
        'status': 'success'
    }


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES (for tests)
# ============================================================================

# Alias old function names to new ones for backward compatibility
chunk_document_from_file = chunk_document
extract_metadata = extract_file_metadata
get_contextual_retrieval_config = get_ingestion_config
