import asyncio
import logging
from functools import partial
from fastapi import APIRouter, HTTPException

from schemas.chat import ChatHistoryResponse, ClearSessionRequest, ClearSessionResponse, SessionMetadataResponse
from pipelines.inference import get_chat_history, clear_session_memory
from services.session import get_session_metadata_async

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/chat/history/{session_id}", response_model=ChatHistoryResponse)
async def get_session_history(session_id: str):
    """Get the full chat history for a session, including metadata"""
    try:
        loop = asyncio.get_running_loop()
        messages = await loop.run_in_executor(None, get_chat_history, session_id)

        # Convert ChatMessage objects to dicts
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                "content": msg.content
            })

        # Get session metadata
        session_meta = await get_session_metadata_async(session_id)
        metadata = None
        if session_meta:
            metadata = SessionMetadataResponse(
                session_id=session_meta.session_id,
                title=session_meta.title,
                created_at=session_meta.created_at,
                updated_at=session_meta.updated_at,
                is_archived=session_meta.is_archived,
                is_temporary=session_meta.is_temporary,
                llm_model=session_meta.llm_model,
                search_type=session_meta.search_type
            )

        return ChatHistoryResponse(
            session_id=session_id,
            messages=formatted_messages,
            metadata=metadata
        )
    except Exception as e:
        logger.error(f"[CHAT_HISTORY] Error retrieving history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/clear", response_model=ClearSessionResponse)
async def clear_chat_session(request: ClearSessionRequest):
    """Clear the chat history for a session"""
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, clear_session_memory, request.session_id)
        return ClearSessionResponse(
            status="success",
            message=f"Chat history cleared for session {request.session_id}"
        )
    except Exception as e:
        logger.error(f"[CHAT_CLEAR] Error clearing session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
