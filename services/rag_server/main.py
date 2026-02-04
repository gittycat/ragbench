import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*validate_default.*")

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from core.logging import configure_logging
from core.config import initialize_settings

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    # Startup
    await startup()
    yield
    # Shutdown
    await shutdown()


async def startup():
    """Initialize services and pre-load models on startup."""
    initialize_settings()

    # Initialize PostgreSQL connection pool
    logger.info("[STARTUP] Initializing PostgreSQL connection pool...")
    try:
        from infrastructure.database.postgres import init_db
        await init_db()
        logger.info("[STARTUP] PostgreSQL connection pool ready")
    except Exception as e:
        logger.error(f"[STARTUP] PostgreSQL initialization failed: {e}")
        raise

    # Pre-initialize reranker to download model during startup (if enabled)
    from pipelines.inference import create_reranker_postprocessor
    from infrastructure.config.models_config import get_models_config
    config = get_models_config()
    if config.reranker.enabled:
        logger.info("[STARTUP] Pre-initializing reranker model...")
        create_reranker_postprocessor()
        logger.info("[STARTUP] Reranker model ready")
    else:
        logger.info("[STARTUP] Reranker disabled, skipping initialization")

    # Log hybrid search status
    if config.retrieval.enable_hybrid_search:
        logger.info("[STARTUP] Hybrid search enabled (pg_search BM25 + pgvector)")
    else:
        logger.info("[STARTUP] Hybrid search disabled, using vector-only retrieval")


async def shutdown():
    """Clean up resources on shutdown."""
    logger.info("[SHUTDOWN] Closing PostgreSQL connection pool...")
    try:
        from infrastructure.database.postgres import close_db
        await close_db()
        logger.info("[SHUTDOWN] PostgreSQL connection pool closed")
    except Exception as e:
        logger.error(f"[SHUTDOWN] Error closing PostgreSQL connection: {e}")


# Initialize FastAPI app with lifespan
app = FastAPI(title="RAG Bench", lifespan=lifespan)

# Include routers
from api.routes import health, query, documents, chat, metrics, sessions

app.include_router(health.router, tags=["health"])
app.include_router(query.router, tags=["query"])
app.include_router(documents.router, tags=["documents"])
app.include_router(chat.router, tags=["chat"])
app.include_router(sessions.router, tags=["sessions"])
app.include_router(metrics.router, tags=["metrics"])
