"""
Session Metadata Management Service

Manages chat session metadata using PostgreSQL:
- Session metadata: title, timestamps, archive status
- Stored in chat_sessions table
- No TTL (persist indefinitely until deleted)
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional
from dataclasses import dataclass
from uuid import UUID

from infrastructure.database.postgres import get_session
from infrastructure.database.repositories.sessions import SessionRepository

logger = logging.getLogger(__name__)


@dataclass
class SessionMetadata:
    """Chat session metadata"""
    session_id: str
    title: str
    created_at: str  # ISO 8601 timestamp
    updated_at: str  # ISO 8601 timestamp
    is_archived: bool = False
    is_temporary: bool = False
    llm_model: str | None = None      # e.g., "gemma3:4b"
    search_type: str | None = None    # "vector" | "hybrid"


def _run_async(coro):
    """Run async function from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in async context - use nest_asyncio approach
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


def _get_inference_settings() -> tuple[str | None, str | None]:
    try:
        from infrastructure.config.models_config import get_models_config
        config = get_models_config()
        llm_model = config.llm.model
        search_type = "hybrid" if config.retrieval.enable_hybrid_search else "vector"
        return llm_model, search_type
    except Exception as e:
        logger.warning(f"[SESSION] Could not get inference settings: {e}")
        return None, None


def create_session_metadata(
    session_id: str,
    is_temporary: bool = False,
    title: str = "New Chat"
) -> SessionMetadata:
    """
    Create new session metadata.

    If is_temporary=True, metadata is not persisted to PostgreSQL.
    Captures current inference settings (LLM model, search type) at creation time.
    """
    now = datetime.now(timezone.utc).isoformat()
    llm_model, search_type = _get_inference_settings()

    metadata = SessionMetadata(
        session_id=session_id,
        title=title,
        created_at=now,
        updated_at=now,
        is_archived=False,
        is_temporary=is_temporary,
        llm_model=llm_model,
        search_type=search_type
    )

    if not is_temporary:
        _run_async(_create_session_async(session_id, title, llm_model, search_type))
        logger.info(f"[SESSION] Created metadata for session: {session_id}")
    else:
        logger.info(f"[SESSION] Created temporary session: {session_id} (not persisted)")

    return metadata


async def _create_session_async(
    session_id: str,
    title: str,
    llm_model: str | None,
    search_type: str | None,
) -> None:
    """Create session in PostgreSQL."""
    async with get_session() as session:
        repo = SessionRepository(session)
        try:
            uuid_id = UUID(session_id)
        except ValueError:
            logger.error(f"[SESSION] Invalid session_id format: {session_id}")
            return
        await repo.create_session(
            session_id=uuid_id,
            title=title,
            llm_model=llm_model,
            search_type=search_type,
            is_temporary=False,
        )


def get_session_metadata(session_id: str) -> Optional[SessionMetadata]:
    """Get session metadata from PostgreSQL"""
    return _run_async(_get_session_metadata_async(session_id))


async def _get_session_metadata_async(session_id: str) -> Optional[SessionMetadata]:
    """Get session metadata from PostgreSQL."""
    try:
        uuid_id = UUID(session_id)
    except ValueError:
        return None

    async with get_session() as session:
        repo = SessionRepository(session)
        data = await repo.get_session_metadata(uuid_id)

        if not data:
            logger.debug(f"[SESSION] Metadata not found: {session_id}")
            return None

        return SessionMetadata(
            session_id=data["session_id"],
            title=data["title"],
            created_at=data["created_at"] or "",
            updated_at=data["updated_at"] or "",
            is_archived=data["is_archived"],
            is_temporary=data["is_temporary"],
            llm_model=data["llm_model"],
            search_type=data["search_type"],
        )


def update_session_title(session_id: str, title: str) -> None:
    """Update session title (e.g., from first user message)"""
    _run_async(_update_session_title_async(session_id, title))
    logger.info(f"[SESSION] Updated title for {session_id}: {title}")


async def _update_session_title_async(session_id: str, title: str) -> None:
    """Update session title in PostgreSQL."""
    try:
        uuid_id = UUID(session_id)
    except ValueError:
        logger.warning(f"[SESSION] Invalid session_id format: {session_id}")
        return

    async with get_session() as session:
        repo = SessionRepository(session)
        await repo.update_title(uuid_id, title)


def touch_session(session_id: str) -> None:
    """Update session's updated_at timestamp (on each message)"""
    metadata = get_session_metadata(session_id)
    if not metadata:
        # Lazy initialization for existing sessions without metadata
        logger.info(f"[SESSION] Lazy-creating metadata for existing session: {session_id}")
        create_session_metadata(session_id)
        return

    _run_async(_touch_session_async(session_id))


async def _touch_session_async(session_id: str) -> None:
    """Touch session in PostgreSQL."""
    try:
        uuid_id = UUID(session_id)
    except ValueError:
        return

    async with get_session() as session:
        repo = SessionRepository(session)
        await repo.touch(uuid_id)


def list_sessions(
    include_archived: bool = False,
    limit: int = 100,
    offset: int = 0
) -> List[SessionMetadata]:
    """
    List all sessions (excluding temporary).

    Returns sessions sorted by updated_at (newest first).
    """
    return _run_async(_list_sessions_async(include_archived, limit, offset))


async def _list_sessions_async(
    include_archived: bool,
    limit: int,
    offset: int,
) -> List[SessionMetadata]:
    """List sessions from PostgreSQL."""
    async with get_session() as session:
        repo = SessionRepository(session)
        sessions_data = await repo.list_sessions(include_archived, limit, offset)

        return [
            SessionMetadata(
                session_id=s["session_id"],
                title=s["title"],
                created_at=s["created_at"] or "",
                updated_at=s["updated_at"] or "",
                is_archived=s["is_archived"],
                is_temporary=s["is_temporary"],
                llm_model=s["llm_model"],
                search_type=s["search_type"],
            )
            for s in sessions_data
        ]


def archive_session(session_id: str) -> None:
    """Mark session as archived"""
    _run_async(_archive_session_async(session_id))
    logger.info(f"[SESSION] Archived session: {session_id}")


async def _archive_session_async(session_id: str) -> None:
    """Archive session in PostgreSQL."""
    try:
        uuid_id = UUID(session_id)
    except ValueError:
        logger.warning(f"[SESSION] Invalid session_id format: {session_id}")
        return

    async with get_session() as session:
        repo = SessionRepository(session)
        await repo.archive(uuid_id)


def unarchive_session(session_id: str) -> None:
    """Restore session from archive"""
    _run_async(_unarchive_session_async(session_id))
    logger.info(f"[SESSION] Unarchived session: {session_id}")


async def _unarchive_session_async(session_id: str) -> None:
    """Unarchive session in PostgreSQL."""
    try:
        uuid_id = UUID(session_id)
    except ValueError:
        logger.warning(f"[SESSION] Invalid session_id format: {session_id}")
        return

    async with get_session() as session:
        repo = SessionRepository(session)
        await repo.unarchive(uuid_id)


def delete_session(session_id: str) -> None:
    """
    Delete session completely (metadata + messages).

    This deletes:
    1. Chat messages via PostgresChatStore
    2. Session metadata (CASCADE deletes messages)
    """
    from pipelines.inference import clear_session_memory

    # Delete messages via LlamaIndex
    clear_session_memory(session_id)

    # Delete session (CASCADE deletes messages anyway)
    _run_async(_delete_session_async(session_id))
    logger.info(f"[SESSION] Deleted session: {session_id}")


async def _delete_session_async(session_id: str) -> None:
    """Delete session from PostgreSQL."""
    try:
        uuid_id = UUID(session_id)
    except ValueError:
        logger.warning(f"[SESSION] Invalid session_id format: {session_id}")
        return

    async with get_session() as session:
        repo = SessionRepository(session)
        await repo.delete_by_id(uuid_id)
