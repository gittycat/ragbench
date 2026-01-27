import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*validate_default.*")

import logging
from fastapi import FastAPI

from core.logging import configure_logging
from core.config import initialize_settings

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="RAG Bench")


@app.on_event("startup")
async def startup_event():
    """Initialize services and pre-load models on startup"""
    initialize_settings()

    # Pre-initialize reranker to download model during startup (if enabled)
    # This prevents timeout on first query when model downloads from HuggingFace
    from pipelines.inference import create_reranker_postprocessor
    from infrastructure.config.models_config import get_models_config
    config = get_models_config()
    if config.reranker.enabled:
        logger.info("[STARTUP] Pre-initializing reranker model...")
        create_reranker_postprocessor()
        logger.info("[STARTUP] Reranker model ready")
    else:
        logger.info("[STARTUP] Reranker disabled, skipping initialization")

    # Verify ChromaDB persistence (defensive measure against reported 2025 reliability issues)
    try:
        from infrastructure.database.chroma import get_or_create_collection
        index = get_or_create_collection()
        collection = index._vector_store._collection
        count = collection.count()
        logger.info(f"[STARTUP] ChromaDB persistence check: {count} documents in collection")
        if count == 0:
            logger.warning("[STARTUP] ChromaDB collection is empty - may need document upload or restore from backup")

        # Pre-initialize BM25 retriever for hybrid search (if enabled)
        from pipelines.inference import initialize_bm25_retriever
        if config.retrieval.enable_hybrid_search and count > 0:
            logger.info("[STARTUP] Pre-initializing BM25 retriever for hybrid search...")
            initialize_bm25_retriever(index, config.retrieval.top_k)
            logger.info("[STARTUP] BM25 retriever ready")
        elif config.retrieval.enable_hybrid_search:
            logger.warning("[STARTUP] Hybrid search enabled but no documents in ChromaDB - BM25 will initialize after first upload")
        else:
            logger.info("[STARTUP] Hybrid search disabled")
    except Exception as e:
        logger.error(f"[STARTUP] ChromaDB persistence check failed: {str(e)}")
        # Don't fail startup, but log the error prominently
        logger.error("[STARTUP] ChromaDB may not be accessible - check service connectivity")


# Include routers
from api.routes import health, query, documents, chat, metrics, sessions

app.include_router(health.router, tags=["health"])
app.include_router(query.router, tags=["query"])
app.include_router(documents.router, tags=["documents"])
app.include_router(chat.router, tags=["chat"])
app.include_router(sessions.router, tags=["sessions"])
app.include_router(metrics.router, tags=["metrics"])
