import asyncio
import uuid
import logging
from functools import partial
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from schemas.query import QueryRequest, QueryResponse, QueryMetrics, TokenUsage, QueryWithContextRequest
from pipelines.inference import query_rag, query_rag_stream, query_rag_with_context

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        # Generate session_id if not provided
        session_id = request.session_id or str(uuid.uuid4())
        logger.info(f"[QUERY] Processing query with session_id: {session_id} (temporary={request.is_temporary})")

        # Create session metadata if needed (non-temporary sessions only)
        if not request.is_temporary:
            from services.session import get_session_metadata_async, create_session_metadata_async
            metadata = await get_session_metadata_async(session_id)
            if not metadata:
                await create_session_metadata_async(session_id, is_temporary=False)

        # Run in executor to keep the main event loop free for async DB operations
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                query_rag,
                request.query,
                session_id=session_id,
                is_temporary=request.is_temporary,
                include_chunks=request.include_chunks,
                ensure_metadata=False,
                update_session_metadata=False,
            ),
        )

        # Update session metadata after query (non-temporary sessions only)
        if not request.is_temporary:
            from services.session import touch_session_async, update_session_title_async, get_session_metadata_async
            from services.session_titles import generate_session_title

            await touch_session_async(session_id)
            metadata = await get_session_metadata_async(session_id)
            if metadata and metadata.title == "New Chat":
                title = generate_session_title(request.query)
                await update_session_title_async(session_id, title)

        # Build metrics if available
        metrics = None
        if result.get('metrics'):
            raw_metrics = result['metrics']
            token_usage = None
            if raw_metrics.get('token_usage'):
                token_usage = TokenUsage(
                    prompt_tokens=raw_metrics['token_usage']['prompt_tokens'],
                    completion_tokens=raw_metrics['token_usage']['completion_tokens'],
                    total_tokens=raw_metrics['token_usage']['total_tokens'],
                )
            metrics = QueryMetrics(
                latency_ms=raw_metrics['latency_ms'],
                token_usage=token_usage,
            )

        return QueryResponse(
            answer=result['answer'],
            sources=result['sources'],
            session_id=result['session_id'],
            citations=result.get("citations"),
            metrics=metrics,
        )
    except Exception as e:
        import traceback
        logger.error(f"[QUERY] Error processing query: {str(e)}")
        logger.error(f"[QUERY] Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/with-context", response_model=QueryResponse)
async def query_with_context(request: QueryWithContextRequest):
    """Execute RAG generation with pre-injected context (bypasses retrieval)."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        passages = [{"text": p.text, "doc_id": p.doc_id} for p in request.context_passages]

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            partial(query_rag_with_context, request.query, passages, session_id),
        )

        metrics = None
        if result.get("metrics"):
            raw_metrics = result["metrics"]
            token_usage = None
            if raw_metrics.get("token_usage"):
                token_usage = TokenUsage(
                    prompt_tokens=raw_metrics["token_usage"]["prompt_tokens"],
                    completion_tokens=raw_metrics["token_usage"]["completion_tokens"],
                    total_tokens=raw_metrics["token_usage"]["total_tokens"],
                )
            metrics = QueryMetrics(
                latency_ms=raw_metrics["latency_ms"],
                token_usage=token_usage,
            )

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=result["session_id"],
            citations=None,
            metrics=metrics,
        )
    except Exception as e:
        import traceback
        logger.error(f"[QUERY_WITH_CONTEXT] Error: {str(e)}")
        logger.error(f"[QUERY_WITH_CONTEXT] Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Stream RAG query response using Server-Sent Events.

    Returns a stream of SSE events:
    - event: token, data: {"token": "..."}  (streamed response tokens)
    - event: sources, data: {"sources": [...], "session_id": "..."}  (source documents)
    - event: done, data: {}  (completion signal)
    - event: error, data: {"error": "..."}  (on error)
    """
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"[QUERY_STREAM] Starting streaming query with session_id: {session_id} (temporary={request.is_temporary})")

    # Create session metadata if needed (non-temporary sessions only)
    if not request.is_temporary:
        from services.session import get_session_metadata_async, create_session_metadata_async
        metadata = await get_session_metadata_async(session_id)
        if not metadata:
            await create_session_metadata_async(session_id, is_temporary=False)

    async def stream_wrapper():
        """Run sync generator in a thread, feed chunks via async queue."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def sync_producer():
            try:
                for chunk in query_rag_stream(
                    request.query,
                    session_id,
                    is_temporary=request.is_temporary,
                    include_chunks=request.include_chunks,
                    ensure_metadata=False,
                ):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop).result()

        loop.run_in_executor(None, sync_producer)
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield item

    return StreamingResponse(
        stream_wrapper(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
