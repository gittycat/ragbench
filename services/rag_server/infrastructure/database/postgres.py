"""PostgreSQL async connection pool using SQLAlchemy 2.0 + asyncpg."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import text, pool
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.settings import get_database_url
from infrastructure.config.models_config import get_database_config

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None

# Main event loop reference for cross-thread async scheduling
_main_loop: asyncio.AbstractEventLoop | None = None


def set_main_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Store the main event loop for cross-thread async scheduling."""
    global _main_loop
    _main_loop = loop
    logger.info("Main event loop stored for async scheduling")


def run_async_safely(coro):
    """
    Run an async coroutine from sync context on the main event loop.

    Uses run_coroutine_threadsafe to keep all asyncpg connections on the
    same event loop. Creating new event loops in threads contaminates the
    connection pool with connections bound to different loops, causing
    'Future attached to a different loop' errors.

    IMPORTANT: Must be called from a thread OTHER than the main loop's
    thread. Async route handlers should use run_in_executor for blocking
    sync calls that go through this bridge.
    """
    if _main_loop is not None and _main_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, _main_loop)
        return future.result(timeout=60)
    # Fallback for startup/testing when main loop isn't available
    return asyncio.run(coro)


def get_engine() -> AsyncEngine:
    """Get or create the async engine singleton."""
    global _engine
    if _engine is None:
        database_url = get_database_url()
        db_config = get_database_config()
        _engine = create_async_engine(
            database_url,
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow,
            pool_pre_ping=db_config.pool_pre_ping,
            pool_recycle=db_config.pool_recycle,
            echo=os.environ.get("LOG_LEVEL", "").upper() == "DEBUG",
        )
        logger.info(
            f"Created async PostgreSQL engine "
            f"(pool_size={db_config.pool_size}, max_overflow={db_config.max_overflow}, "
            f"pre_ping={db_config.pool_pre_ping}, recycle={db_config.pool_recycle}s)"
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the session factory singleton."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info("Created async session factory")
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async session for database operations.

    Uses explicit transaction control via session.begin() as recommended
    in SQLAlchemy 2.0. Commits on success, rolls back on exception.
    """
    factory = get_session_factory()
    async with factory() as session:
        async with session.begin():
            yield session
            # Commit happens automatically at context exit
            # Rollback happens automatically on exception


async def init_db() -> None:
    """Initialize database connection pool. Call on startup."""
    engine = get_engine()
    # Test connection
    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))
    logger.info("Database connection pool initialized")


async def close_db() -> None:
    """Close database connection pool. Call on shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connection pool closed")


def close_all_connections() -> None:
    """
    Close all database connections synchronously.

    Used by pgmq-worker to cleanup connections before creating a new event loop.
    This is necessary because the worker runs in a separate process and needs to
    dispose of any inherited connections from the parent process.

    Note: Creates a temporary event loop for async cleanup since this must be
    called from sync context during worker initialization.
    """
    global _engine, _session_factory
    if _engine is not None:
        try:
            import asyncio

            # Try to use existing event loop if available
            try:
                loop = asyncio.get_running_loop()
                # If we're in a running loop, we can't call run_until_complete
                # Just log and reset the globals
                logger.warning(
                    "Cannot dispose engine from running event loop. "
                    "Resetting connection globals."
                )
                _engine = None
                _session_factory = None
                return
            except RuntimeError:
                # No running loop - create temporary one for cleanup
                pass

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_engine.dispose())
                logger.info("Database connections closed successfully")
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except Exception as e:
            logger.warning(f"Error closing database connections: {e}")
        finally:
            _engine = None
            _session_factory = None
