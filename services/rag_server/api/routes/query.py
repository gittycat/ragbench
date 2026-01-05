import uuid
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from schemas.query import QueryRequest, QueryResponse, QueryMetrics, TokenUsage
from pipelines.inference import query_rag, query_rag_stream

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
            from services.session import get_session_metadata, create_session_metadata
            metadata = get_session_metadata(session_id)
            if not metadata:
                create_session_metadata(session_id, is_temporary=False)

        result = query_rag(
            request.query,
            session_id=session_id,
            is_temporary=request.is_temporary,
            include_chunks=request.include_chunks,
        )

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
        from services.session import get_session_metadata, create_session_metadata
        metadata = get_session_metadata(session_id)
        if not metadata:
            create_session_metadata(session_id, is_temporary=False)

    return StreamingResponse(
        query_rag_stream(
            request.query,
            session_id,
            is_temporary=request.is_temporary,
            include_chunks=request.include_chunks,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
