"""Session repository and PostgreSQL-backed chat store."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence
from uuid import UUID

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.storage.chat_store import BaseChatStore
from sqlalchemy import select, delete, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ChatSession, ChatMessage as ChatMessageModel
from infrastructure.database.postgres import get_session
from infrastructure.database.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class SessionRepository(BaseRepository[ChatSession]):
    """Repository for chat session operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, ChatSession)

    async def create_session(
        self,
        session_id: UUID,
        title: str = "New Chat",
        llm_model: str | None = None,
        search_type: str | None = None,
        is_temporary: bool = False,
    ) -> ChatSession:
        """Create a new chat session."""
        session = ChatSession(
            id=session_id,
            title=title,
            llm_model=llm_model,
            search_type=search_type,
            is_temporary=is_temporary,
        )
        return await self.add(session)

    async def get_session_metadata(self, session_id: UUID) -> dict[str, Any] | None:
        """Get session metadata as a dict."""
        session = await self.get_by_id(session_id)
        if not session:
            return None
        return {
            "session_id": str(session.id),
            "title": session.title,
            "llm_model": session.llm_model,
            "search_type": session.search_type,
            "is_archived": session.is_archived,
            "is_temporary": session.is_temporary,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "updated_at": session.updated_at.isoformat() if session.updated_at else None,
        }

    async def update_title(self, session_id: UUID, title: str) -> None:
        """Update session title."""
        await self.session.execute(
            update(ChatSession)
            .where(ChatSession.id == session_id)
            .values(title=title, updated_at=func.now())
        )
        await self.session.flush()

    async def touch(self, session_id: UUID) -> None:
        """Update session's updated_at timestamp."""
        await self.session.execute(
            update(ChatSession)
            .where(ChatSession.id == session_id)
            .values(updated_at=func.now())
        )
        await self.session.flush()

    async def archive(self, session_id: UUID) -> None:
        """Mark session as archived."""
        await self.session.execute(
            update(ChatSession)
            .where(ChatSession.id == session_id)
            .values(is_archived=True, updated_at=func.now())
        )
        await self.session.flush()

    async def unarchive(self, session_id: UUID) -> None:
        """Restore session from archive."""
        await self.session.execute(
            update(ChatSession)
            .where(ChatSession.id == session_id)
            .values(is_archived=False, updated_at=func.now())
        )
        await self.session.flush()

    async def list_sessions(
        self,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List sessions sorted by updated_at (newest first)."""
        query = select(ChatSession).where(ChatSession.is_temporary == False)
        if not include_archived:
            query = query.where(ChatSession.is_archived == False)
        query = query.order_by(ChatSession.updated_at.desc()).limit(limit).offset(offset)

        result = await self.session.execute(query)
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

    async def add_message(
        self,
        session_id: UUID,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessageModel:
        """Add a message to a session."""
        msg = ChatMessageModel(
            session_id=session_id,
            role=role,
            content=content,
            metadata_=metadata or {},
        )
        self.session.add(msg)
        await self.session.flush()
        return msg

    async def get_messages(self, session_id: UUID) -> list[ChatMessageModel]:
        """Get all messages for a session, ordered by creation time."""
        result = await self.session.execute(
            select(ChatMessageModel)
            .where(ChatMessageModel.session_id == session_id)
            .order_by(ChatMessageModel.created_at)
        )
        return list(result.scalars().all())

    async def delete_messages(self, session_id: UUID) -> None:
        """Delete all messages for a session."""
        await self.session.execute(
            delete(ChatMessageModel).where(ChatMessageModel.session_id == session_id)
        )
        await self.session.flush()


class PostgresChatStore(BaseChatStore):
    """
    PostgreSQL-backed chat store implementing LlamaIndex's BaseChatStore interface.

    This replaces RedisChatStore for persistent chat history storage.
    """

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Set messages for a key (overwrites existing)."""
        asyncio.run(self._async_set_messages(key, messages))

    def get_messages(self, key: str) -> List[ChatMessage]:
        """Get messages for a key."""
        return asyncio.run(self._async_get_messages(key))

    def add_message(self, key: str, message: ChatMessage) -> None:
        """Add a message for a key."""
        asyncio.run(self._async_add_message(key, message))

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Delete messages for a key. Returns deleted messages."""
        return asyncio.run(self._async_delete_messages(key))

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete a specific message by index."""
        return asyncio.run(self._async_delete_message(key, idx))

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Delete the last message for a key."""
        return asyncio.run(self._async_delete_last_message(key))

    def get_keys(self) -> List[str]:
        """Get all keys (session IDs)."""
        return asyncio.run(self._async_get_keys())

    # Async implementations

    async def _async_set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Set messages for a session (overwrites existing)."""
        session_id = UUID(key)
        async with get_session() as session:
            repo = SessionRepository(session)

            # Ensure session exists
            existing = await repo.get_by_id(session_id)
            if not existing:
                await repo.create_session(session_id)

            # Delete existing messages
            await repo.delete_messages(session_id)

            # Add new messages
            for msg in messages:
                await repo.add_message(
                    session_id,
                    role=msg.role.value,
                    content=msg.content,
                    metadata=msg.additional_kwargs,
                )

    async def _async_get_messages(self, key: str) -> List[ChatMessage]:
        """Get messages for a session."""
        try:
            session_id = UUID(key)
        except ValueError:
            return []

        async with get_session() as session:
            repo = SessionRepository(session)
            db_messages = await repo.get_messages(session_id)

            return [
                ChatMessage(
                    role=MessageRole(msg.role),
                    content=msg.content,
                    additional_kwargs=msg.metadata_ or {},
                )
                for msg in db_messages
            ]

    async def _async_add_message(self, key: str, message: ChatMessage) -> None:
        """Add a message to a session."""
        session_id = UUID(key)
        async with get_session() as session:
            repo = SessionRepository(session)

            # Ensure session exists
            existing = await repo.get_by_id(session_id)
            if not existing:
                await repo.create_session(session_id)

            await repo.add_message(
                session_id,
                role=message.role.value,
                content=message.content,
                metadata=message.additional_kwargs,
            )

    async def _async_delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Delete all messages for a session."""
        try:
            session_id = UUID(key)
        except ValueError:
            return None

        async with get_session() as session:
            repo = SessionRepository(session)

            # Get messages before deleting
            messages = await self._async_get_messages(key)

            # Delete
            await repo.delete_messages(session_id)

            return messages if messages else None

    async def _async_delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete a specific message by index."""
        messages = await self._async_get_messages(key)
        if not messages or idx >= len(messages):
            return None

        deleted = messages.pop(idx)
        await self._async_set_messages(key, messages)
        return deleted

    async def _async_delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Delete the last message for a session."""
        messages = await self._async_get_messages(key)
        if not messages:
            return None

        deleted = messages.pop()
        await self._async_set_messages(key, messages)
        return deleted

    async def _async_get_keys(self) -> List[str]:
        """Get all session IDs."""
        async with get_session() as session:
            result = await session.execute(select(ChatSession.id))
            return [str(row[0]) for row in result.all()]
