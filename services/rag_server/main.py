import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*validate_default.*")

import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI

from core.logging import configure_logging
from core.config import initialize_settings
from app.settings import init_settings, get_api_key_for_provider
from infrastructure.llm.validation import validate_api_key

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


async def _validate_file_loaded_api_keys():
    """Validate all file-loaded API keys at startup."""
    from pathlib import Path
    import yaml

    # Load config to get providers that require API keys
    config_paths = [
        Path("/app/config.yml"),
        Path(__file__).parent.parent / "config.yml",
    ]

    config = None
    for config_path in config_paths:
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            break

    if not config:
        logger.warning("[STARTUP] Could not load config.yml, skipping API key validation")
        return

    # Get all providers that require API keys
    providers_to_check = set()
    for model_type in ["inference", "embedding", "eval"]:
        if model_type in config.get("models", {}):
            for model_def in config["models"][model_type].values():
                if model_def.get("requires_api_key", False):
                    provider = model_def.get("provider")
                    if provider:
                        providers_to_check.add(provider.lower())

    if not providers_to_check:
        logger.info("[STARTUP] No providers require API key validation")
        return

    logger.info(f"[STARTUP] Validating API keys for providers: {', '.join(sorted(providers_to_check))}")

    # Collect providers that have keys
    import asyncio
    providers_with_keys = []
    for provider in sorted(providers_to_check):
        api_key = get_api_key_for_provider(provider)
        if not api_key or api_key.strip() == "":
            logger.warning(f"[STARTUP] No API key found for provider: {provider}")
            continue
        providers_with_keys.append((provider, api_key))

    # Validate all keys concurrently
    if providers_with_keys:
        results = await asyncio.gather(
            *(validate_api_key(p, k) for p, k in providers_with_keys)
        )
        validation_errors = []
        for (provider, _), (valid, error_message) in zip(providers_with_keys, results):
            if valid:
                logger.info(f"[STARTUP] API key valid for {provider}")
            else:
                error_msg = f"API key validation failed for {provider}: {error_message}"
                logger.error(f"[STARTUP] {error_msg}")
                validation_errors.append(error_msg)

        if validation_errors:
            logger.error("[STARTUP] API key validation failed. Shutting down.")
            for error in validation_errors:
                logger.error(f"  - {error}")
            os._exit(1)


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
    import asyncio
    from infrastructure.database.postgres import set_main_event_loop
    set_main_event_loop(asyncio.get_running_loop())

    # Skip HuggingFace network calls when reranker model is pre-cached (via `just init`)
    if os.environ.get("USE_CACHED_RERANKER", "").lower() in ("1", "true", "yes"):
        os.environ["HF_HUB_OFFLINE"] = "1"

    init_settings()
    initialize_settings()

    # Validate file-loaded API keys
    await _validate_file_loaded_api_keys()

    # Initialize PostgreSQL connection pool
    logger.info("[STARTUP] Initializing PostgreSQL connection pool...")
    try:
        from infrastructure.database.postgres import init_db
        await init_db()
        logger.info("[STARTUP] PostgreSQL connection pool ready")
    except Exception as e:
        logger.error(f"[STARTUP] PostgreSQL initialization failed: {e}")
        raise

    # Pre-initialize reranker (if enabled) and fail fast if model is missing
    from pipelines.inference import create_reranker_postprocessor, ensure_reranker_model_cached
    from infrastructure.config.models_config import get_models_config
    config = get_models_config()
    if config.reranker.enabled:
        ensure_reranker_model_cached()
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
from api.routes import health, query, documents, chat, metrics, sessions, api_keys

app.include_router(health.router, tags=["health"])
app.include_router(query.router, tags=["query"])
app.include_router(documents.router, tags=["documents"])
app.include_router(chat.router, tags=["chat"])
app.include_router(sessions.router, tags=["sessions"])
app.include_router(metrics.router, tags=["metrics"])
app.include_router(api_keys.router, tags=["api-keys"])
