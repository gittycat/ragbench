"""PGVectorStore wrapper for LlamaIndex integration."""

import logging

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore

from app.settings import get_database_params

logger = logging.getLogger(__name__)

_vector_store: PGVectorStore | None = None
_vector_index: VectorStoreIndex | None = None


def get_vector_store() -> PGVectorStore:
    """Get or create the PGVectorStore singleton."""
    global _vector_store
    if _vector_store is None:
        params = get_database_params()
        _vector_store = PGVectorStore.from_params(
            database=params["database"],
            host=params["host"],
            port=params["port"],
            user=params["user"],
            password=params["password"],
            table_name="document_chunks",
            embed_dim=768,  # nomic-embed-text dimension
            hybrid_search=False,  # We use pg_search for BM25 separately
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 100,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )
        logger.info("Created PGVectorStore connection")
    return _vector_store


def get_vector_index() -> VectorStoreIndex:
    """Get or create the VectorStoreIndex singleton."""
    global _vector_index
    if _vector_index is None:
        vector_store = get_vector_store()
        _vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        logger.info("Created VectorStoreIndex from PGVectorStore")
    return _vector_index


def reset_vector_store() -> None:
    """Reset the vector store singletons (for testing)."""
    global _vector_store, _vector_index
    _vector_store = None
    _vector_index = None
