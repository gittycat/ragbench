import os
import sys
import logging
from llama_index.core import Settings

logger = logging.getLogger(__name__)


def get_required_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        print(f"ERROR: Required environment variable '{var_name}' is not set.", file=sys.stderr)
        print(f"Please define {var_name} in docker-compose.yml", file=sys.stderr)
        sys.exit(1)
    return value


def get_optional_env(var_name: str, default: str = "") -> str:
    return os.getenv(var_name, default)


def check_embedding_dimension_match():
    """Guard against silent retrieval corruption from switching embedding models.

    If the ChromaDB collection already holds vectors, their dimension must match
    the active embedding model's output dimension.
    """
    from infrastructure.search.vector_store import get_chroma_client
    from infrastructure.config.models_config import get_models_config

    config = get_models_config()

    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(config.chromadb.collection)
        count = collection.count()
    except Exception as e:
        logger.warning(f"[SETTINGS] Could not reach ChromaDB to verify embedding dimension, skipping check: {e}")
        return

    if count == 0:
        return

    existing = collection.peek(limit=1)
    embeddings = existing.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return

    stored_dim = len(embeddings[0])
    active_dim = len(Settings.embed_model.get_text_embedding("dim-probe"))

    if stored_dim != active_dim:
        raise ValueError(
            f"Embedding dimension mismatch: ChromaDB collection '{config.chromadb.collection}' "
            f"stores {stored_dim}-dimensional vectors, but the active embedding model "
            f"'{config.embedding.model}' produces {active_dim}-dimensional vectors. "
            f"Delete and re-index the collection, or switch the embedding model back."
        )


def initialize_settings():
    """Initialize global LlamaIndex Settings"""
    from infrastructure.llm.embeddings import get_embedding_function
    from infrastructure.llm.factory import get_llm_client

    logger.info("[SETTINGS] Initializing global LlamaIndex Settings")

    Settings.embed_model = get_embedding_function()
    logger.info("[SETTINGS] Embedding model configured")

    check_embedding_dimension_match()
    logger.info("[SETTINGS] Embedding dimension check passed")

    Settings.llm = get_llm_client()
    logger.info("[SETTINGS] LLM configured")

    Settings.chunk_size = 500
    Settings.chunk_overlap = 50
    logger.info(f"[SETTINGS] Chunk settings: size={Settings.chunk_size}, overlap={Settings.chunk_overlap}")

    logger.info("[SETTINGS] Global Settings initialization complete")
