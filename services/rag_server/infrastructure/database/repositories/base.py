"""Base repository with common database operations."""

from typing import Generic, TypeVar, Type
from uuid import UUID

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload  # noqa: F401 - Available for subclasses

from infrastructure.database.models import Base

T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T]):
    """
    Base repository providing common CRUD operations.

    Relationship Loading
    --------------------
    To avoid N+1 queries when accessing relationships, use explicit loading:

    Example with selectinload (separate query):
        result = await session.execute(
            select(Document).options(selectinload(Document.chunks))
        )

    Example with joinedload (single query with JOIN):
        result = await session.execute(
            select(Document).options(joinedload(Document.chunks))
        )

    See: https://docs.sqlalchemy.org/en/20/orm/queryguide/relationships.html
    """

    def __init__(self, session: AsyncSession, model: Type[T]):
        self.session = session
        self.model = model

    async def get_by_id(self, id: UUID) -> T | None:
        """Get entity by ID."""
        return await self.session.get(self.model, id)

    async def get_all(self) -> list[T]:
        """Get all entities."""
        result = await self.session.execute(select(self.model))
        return list(result.scalars().all())

    async def add(self, entity: T) -> T:
        """Add new entity."""
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def add_all(self, entities: list[T]) -> list[T]:
        """Add multiple entities."""
        self.session.add_all(entities)
        await self.session.flush()
        return entities

    async def delete(self, entity: T) -> None:
        """Delete entity."""
        await self.session.delete(entity)
        await self.session.flush()

    async def delete_by_id(self, id: UUID) -> bool:
        """Delete entity by ID. Returns True if deleted."""
        result = await self.session.execute(
            delete(self.model).where(self.model.id == id)
        )
        await self.session.flush()
        return result.rowcount > 0
