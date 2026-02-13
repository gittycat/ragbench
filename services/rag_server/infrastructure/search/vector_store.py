"""ChromaDB vector store wrapper for LlamaIndex integration."""

import logging
import os
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from infrastructure.config.models_config import get_models_config

logger = logging.getLogger(__name__)

_chroma_client: Optional[chromadb.HttpClient] = None
_vector_store: Optional[ChromaVectorStore] = None
_vector_index: Optional[VectorStoreIndex] = None


def get_chroma_client() -> chromadb.HttpClient:
    """Get or create the ChromaDB HTTP client singleton."""
    global _chroma_client
    if _chroma_client is None:
        host = os.getenv("CHROMADB_HOST", "localhost")
        port = int(os.getenv("CHROMADB_PORT", "8000"))

        _chroma_client = chromadb.HttpClient(host=host, port=port)
        logger.info(f"Created ChromaDB client connection (host={host}, port={port})")
    return _chroma_client


def get_vector_store() -> ChromaVectorStore:
    """Get or create the ChromaVectorStore singleton."""
    global _vector_store
    if _vector_store is None:
        config = get_models_config()
        collection_name = config.chromadb.collection

        chroma_client = get_chroma_client()
        chroma_collection = chroma_client.get_or_create_collection(collection_name)

        _vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        logger.info(f"Created ChromaVectorStore (collection={collection_name})")
    return _vector_store


def get_vector_index() -> VectorStoreIndex:
    """Get or create the VectorStoreIndex singleton."""
    global _vector_index
    if _vector_index is None:
        vector_store = get_vector_store()
        _vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        logger.info("Created VectorStoreIndex from ChromaVectorStore")
    return _vector_index


def reset_vector_store() -> None:
    """Reset the vector store singletons (for testing)."""
    global _vector_store, _vector_index
    _vector_store = None
    _vector_index = None
