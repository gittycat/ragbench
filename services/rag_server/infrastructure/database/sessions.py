"""Session database operations and PostgreSQL-backed chat store."""

import logging
from typing import Any, List, Optional
from uuid import UUID

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.storage.chat_store import BaseChatStore
from sqlalchemy import select, delete, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ChatSession, ChatMessage as ChatMessageModel
from infrastructure.database.postgres import get_session, run_async_safely

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

async def create_session(
    session: AsyncSession,
    session_id: UUID,
    title: str = "New Chat",
    llm_model: str | None = None,
    search_type: str | None = None,
    is_temporary: bool = False,
) -> ChatSession:
    chat_session = ChatSession(
        id=session_id,
        title=title,
        llm_model=llm_model,
        search_type=search_type,
        is_temporary=is_temporary,
    )
    session.add(chat_session)
    await session.flush()
    return chat_session


async def get_session_by_id(session: AsyncSession, session_id: UUID) -> ChatSession | None:
    return await session.get(ChatSession, session_id)


async def get_session_metadata(session: AsyncSession, session_id: UUID) -> dict[str, Any] | None:
    chat_session = await session.get(ChatSession, session_id)
    if not chat_session:
        return None
    return {
        "session_id": str(chat_session.id),
        "title": chat_session.title,
        "llm_model": chat_session.llm_model,
        "search_type": chat_session.search_type,
        "is_archived": chat_session.is_archived,
        "is_temporary": chat_session.is_temporary,
        "created_at": chat_session.created_at.isoformat() if chat_session.created_at else None,
        "updated_at": chat_session.updated_at.isoformat() if chat_session.updated_at else None,
    }


async def update_title(session: AsyncSession, session_id: UUID, title: str) -> None:
    await session.execute(
        update(ChatSession)
        .where(ChatSession.id == session_id)
        .values(title=title, updated_at=func.now())
    )
    await session.flush()


async def touch(session: AsyncSession, session_id: UUID) -> None:
    await session.execute(
        update(ChatSession)
        .where(ChatSession.id == session_id)
        .values(updated_at=func.now())
    )
    await session.flush()


async def archive(session: AsyncSession, session_id: UUID) -> None:
    await session.execute(
        update(ChatSession)
        .where(ChatSession.id == session_id)
        .values(is_archived=True, updated_at=func.now())
    )
    await session.flush()


async def unarchive(session: AsyncSession, session_id: UUID) -> None:
    await session.execute(
        update(ChatSession)
        .where(ChatSession.id == session_id)
        .values(is_archived=False, updated_at=func.now())
    )
    await session.flush()


async def list_sessions(
    session: AsyncSession,
    include_archived: bool = False,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    query = select(ChatSession).where(ChatSession.is_temporary == False)
    if not include_archived:
        query = query.where(ChatSession.is_archived == False)
    query = query.order_by(ChatSession.updated_at.desc()).limit(limit).offset(offset)

    result = await session.execute(query)
    sessions = []
    for row in result.scalars():
        sessions.append({
            "session_id": str(row.id),
            "title": row.title,
            "llm_model": row.llm_model,
            "search_type": row.search_type,
            "is_archived": row.is_archived,
            "is_temporary": row.is_temporary,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        })
    return sessions


async def delete_session(session: AsyncSession, session_id: UUID) -> bool:
    result = await session.execute(
        delete(ChatSession).where(ChatSession.id == session_id)
    )
    await session.flush()
    return result.rowcount > 0


# ---------------------------------------------------------------------------
# Message ops
# ---------------------------------------------------------------------------

async def add_message(
    session: AsyncSession,
    session_id: UUID,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> ChatMessageModel:
    msg = ChatMessageModel(
        session_id=session_id,
        role=role,
        content=content,
        metadata_=metadata or {},
    )
    session.add(msg)
    await session.flush()
    return msg


async def get_messages(session: AsyncSession, session_id: UUID) -> list[ChatMessageModel]:
    result = await session.execute(
        select(ChatMessageModel)
        .where(ChatMessageModel.session_id == session_id)
        .order_by(ChatMessageModel.created_at)
    )
    return list(result.scalars().all())


async def delete_messages(session: AsyncSession, session_id: UUID) -> None:
    await session.execute(
        delete(ChatMessageModel).where(ChatMessageModel.session_id == session_id)
    )
    await session.flush()


async def get_all_session_ids(session: AsyncSession) -> list[str]:
    result = await session.execute(select(ChatSession.id))
    return [str(row[0]) for row in result.all()]


# ---------------------------------------------------------------------------
# LlamaIndex integration (class required by BaseChatStore)
# ---------------------------------------------------------------------------

class PostgresChatStore(BaseChatStore):
    """PostgreSQL-backed chat store implementing LlamaIndex's BaseChatStore.

    Sync methods bridge to async via run_async_safely().
    The _async_* methods are available for direct use from async contexts.
    """

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        run_async_safely(self._async_set_messages(key, messages))

    def get_messages(self, key: str) -> List[ChatMessage]:
        return run_async_safely(self._async_get_messages(key))

    def add_message(self, key: str, message: ChatMessage) -> None:
        run_async_safely(self._async_add_message(key, message))

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        return run_async_safely(self._async_delete_messages(key))

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        return run_async_safely(self._async_delete_message(key, idx))

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        return run_async_safely(self._async_delete_last_message(key))

    def get_keys(self) -> List[str]:
        return run_async_safely(self._async_get_keys())

    # Async implementations

    async def _async_set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        session_id = UUID(key)
        async with get_session() as db:
            existing = await get_session_by_id(db, session_id)
            if not existing:
                await create_session(db, session_id)

            await delete_messages(db, session_id)

            for msg in messages:
                await add_message(
                    db,
                    session_id,
                    role=msg.role.value,
                    content=msg.content,
                    metadata=msg.additional_kwargs,
                )

    async def _async_get_messages(self, key: str) -> List[ChatMessage]:
        try:
            session_id = UUID(key)
        except ValueError:
            return []

        async with get_session() as db:
            db_messages = await get_messages(db, session_id)

            return [
                ChatMessage(
                    role=MessageRole(msg.role),
                    content=msg.content,
                    additional_kwargs=msg.metadata_ or {},
                )
                for msg in db_messages
            ]

    async def _async_add_message(self, key: str, message: ChatMessage) -> None:
        session_id = UUID(key)
        async with get_session() as db:
            existing = await get_session_by_id(db, session_id)
            if not existing:
                await create_session(db, session_id)

            await add_message(
                db,
                session_id,
                role=message.role.value,
                content=message.content,
                metadata=message.additional_kwargs,
            )

    async def _async_delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        try:
            session_id = UUID(key)
        except ValueError:
            return None

        async with get_session() as db:
            # Read and delete in same transaction
            db_messages = await get_messages(db, session_id)
            await delete_messages(db, session_id)

            if not db_messages:
                return None

            return [
                ChatMessage(
                    role=MessageRole(msg.role),
                    content=msg.content,
                    additional_kwargs=msg.metadata_ or {},
                )
                for msg in db_messages
            ]

    async def _async_delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        messages = await self._async_get_messages(key)
        if not messages or idx >= len(messages):
            return None

        deleted = messages.pop(idx)
        await self._async_set_messages(key, messages)
        return deleted

    async def _async_delete_last_message(self, key: str) -> Optional[ChatMessage]:
        messages = await self._async_get_messages(key)
        if not messages:
            return None

        deleted = messages.pop()
        await self._async_set_messages(key, messages)
        return deleted

    async def _async_get_keys(self) -> List[str]:
        async with get_session() as db:
            return await get_all_session_ids(db)
