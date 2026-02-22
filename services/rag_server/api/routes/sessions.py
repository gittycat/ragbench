"""
Session Management API Routes

Endpoints:
- GET /chat/sessions - List sessions (with pagination, filters)
- GET /chat/sessions/{session_id} - Get session metadata
- POST /chat/sessions/new - Create new session
- DELETE /chat/sessions/{session_id} - Delete session
- POST /chat/sessions/{session_id}/archive - Archive session
- POST /chat/sessions/{session_id}/unarchive - Unarchive session
"""

import asyncio
import uuid
import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from schemas.chat import (
    SessionMetadataResponse,
    SessionListResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    DeleteSessionResponse,
    ArchiveSessionResponse
)
from services.session import (
    create_session_metadata,
    create_session_async,
    get_session_metadata,
    list_sessions_async,
    get_session_metadata_async,
    delete_session,
    archive_session_async,
    unarchive_session_async,
)
from services.session_titles import generate_ai_title

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/chat/sessions", response_model=SessionListResponse)
async def get_sessions(
    include_archived: bool = Query(default=False, description="Include archived sessions"),
    limit: int = Query(default=100, ge=1, le=500, description="Max sessions to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset")
):
    """
    List all chat sessions (excluding temporary).

    Returns sessions sorted by updated_at (newest first).
    """
    try:
        sessions = await list_sessions_async(
            include_archived=include_archived,
            limit=limit,
            offset=offset
        )

        return SessionListResponse(
            sessions=[
                SessionMetadataResponse(
                    session_id=s.session_id,
                    title=s.title,
                    created_at=s.created_at,
                    updated_at=s.updated_at,
                    is_archived=s.is_archived,
                    is_temporary=s.is_temporary,
                    llm_model=s.llm_model,
                    search_type=s.search_type
                )
                for s in sessions
            ],
            total=len(sessions)
        )
    except Exception as e:
        logger.error(f"[SESSIONS_API] Error listing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/sessions/{session_id}", response_model=SessionMetadataResponse)
async def get_session(session_id: str):
    """Get metadata for a specific session"""
    try:
        metadata = await get_session_metadata_async(session_id)

        if not metadata:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        return SessionMetadataResponse(
            session_id=metadata.session_id,
            title=metadata.title,
            created_at=metadata.created_at,
            updated_at=metadata.updated_at,
            is_archived=metadata.is_archived,
            is_temporary=metadata.is_temporary,
            llm_model=metadata.llm_model,
            search_type=metadata.search_type
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SESSIONS_API] Error getting session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/sessions/new", response_model=CreateSessionResponse)
async def create_new_session(request: CreateSessionRequest):
    """
    Create a new chat session.

    If is_temporary=True, session is not persisted to PostgreSQL.
    If first_message is provided, generates an AI title from the message.
    """
    try:
        from datetime import datetime, timezone
        from uuid import UUID

        session_id = str(uuid.uuid4())

        # Determine title: explicit > AI-generated from first_message > default
        if request.title:
            title = request.title
        elif request.first_message:
            loop = asyncio.get_running_loop()
            title = await loop.run_in_executor(None, generate_ai_title, request.first_message)
        else:
            title = "New Chat"

        # Get inference settings
        try:
            from infrastructure.config.models_config import get_models_config
            config = get_models_config()
            llm_model = config.llm.model
            search_type = "hybrid" if config.retrieval.enable_hybrid_search else "vector"
        except Exception as e:
            logger.warning(f"[SESSION] Could not get inference settings: {e}")
            llm_model = None
            search_type = None

        now = datetime.now(timezone.utc).isoformat()

        # Persist to database if not temporary
        if not request.is_temporary:
            await create_session_async(session_id, title, llm_model, search_type)
            logger.info(f"[SESSION] Created session: {session_id}")
        else:
            logger.info(f"[SESSION] Created temporary session: {session_id} (not persisted)")

        return CreateSessionResponse(
            session_id=session_id,
            title=title,
            created_at=now,
            is_temporary=request.is_temporary,
            llm_model=llm_model,
            search_type=search_type
        )
    except Exception as e:
        logger.error(f"[SESSIONS_API] Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/sessions/{session_id}", response_model=DeleteSessionResponse)
async def delete_chat_session(session_id: str):
    """
    Delete a chat session completely (metadata + messages).

    This cannot be undone.
    """
    try:
        # Check if session exists
        metadata = await get_session_metadata_async(session_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, delete_session, session_id)

        return DeleteSessionResponse(
            status="success",
            message=f"Session {session_id} deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SESSIONS_API] Error deleting session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/sessions/{session_id}/archive", response_model=ArchiveSessionResponse)
async def archive_chat_session(session_id: str):
    """Mark a session as archived"""
    try:
        # Check if session exists
        metadata = await get_session_metadata_async(session_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        await archive_session_async(session_id)

        return ArchiveSessionResponse(
            status="success",
            message=f"Session {session_id} archived"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SESSIONS_API] Error archiving session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/sessions/{session_id}/unarchive", response_model=ArchiveSessionResponse)
async def unarchive_chat_session(session_id: str):
    """Restore a session from archive"""
    try:
        # Check if session exists
        metadata = await get_session_metadata_async(session_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        await unarchive_session_async(session_id)

        return ArchiveSessionResponse(
            status="success",
            message=f"Session {session_id} restored from archive"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SESSIONS_API] Error unarchiving session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
