import pytest
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.config.models_config import (
    ModelsConfig,
    LLMConfig,
    EmbeddingConfig,
    EvalConfig,
    RerankerConfig,
    RetrievalConfig,
    ChromaDBConfig,
)


def create_mock_models_config():
    return ModelsConfig(
        llm=LLMConfig(provider="ollama", model="gemma3:4b", base_url="http://localhost:11434"),
        embedding=EmbeddingConfig(provider="ollama", model="nomic-embed-text:latest", base_url="http://localhost:11434"),
        eval=EvalConfig(provider="anthropic", model="claude-sonnet-4-20250514", api_key="test-key"),
        reranker=RerankerConfig(enabled=True),
        retrieval=RetrievalConfig(),
        chromadb=ChromaDBConfig(collection="document_chunks"),
    )


def _mock_client(count, embeddings):
    collection = MagicMock()
    collection.count.return_value = count
    collection.peek.return_value = {"embeddings": embeddings}
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client


def test_dimension_match_skips_empty_collection():
    from core.config import check_embedding_dimension_match

    config = create_mock_models_config()
    client = _mock_client(count=0, embeddings=[])

    with patch("infrastructure.search.vector_store.get_chroma_client", return_value=client), \
         patch("infrastructure.config.models_config.get_models_config", return_value=config):
        check_embedding_dimension_match()  # no error


def test_dimension_match_skips_when_chromadb_unreachable():
    """Startup must not hard-fail if ChromaDB isn't reachable yet (e.g. during tests or startup ordering)."""
    from core.config import check_embedding_dimension_match

    config = create_mock_models_config()

    with patch("infrastructure.search.vector_store.get_chroma_client", side_effect=ValueError("Could not connect to a Chroma server")), \
         patch("infrastructure.config.models_config.get_models_config", return_value=config):
        check_embedding_dimension_match()  # no error, logs a warning instead


def test_dimension_match_passes_when_equal():
    from core.config import check_embedding_dimension_match
    from llama_index.core import Settings

    config = create_mock_models_config()
    client = _mock_client(count=5, embeddings=[[0.1] * 768])

    from llama_index.core.embeddings import BaseEmbedding

    mock_embed_model = MagicMock(spec=BaseEmbedding)
    mock_embed_model.get_text_embedding.return_value = [0.1] * 768

    Settings.embed_model = mock_embed_model
    try:
        with patch("infrastructure.search.vector_store.get_chroma_client", return_value=client), \
             patch("infrastructure.config.models_config.get_models_config", return_value=config):
            check_embedding_dimension_match()  # no error
    finally:
        Settings._embed_model = None


def test_dimension_match_raises_on_mismatch():
    from core.config import check_embedding_dimension_match
    from llama_index.core import Settings

    config = create_mock_models_config()
    client = _mock_client(count=5, embeddings=[[0.1] * 768])

    from llama_index.core.embeddings import BaseEmbedding

    mock_embed_model = MagicMock(spec=BaseEmbedding)
    mock_embed_model.get_text_embedding.return_value = [0.1] * 1536

    Settings.embed_model = mock_embed_model
    try:
        with patch("infrastructure.search.vector_store.get_chroma_client", return_value=client), \
             patch("infrastructure.config.models_config.get_models_config", return_value=config):
            with pytest.raises(ValueError, match="Embedding dimension mismatch"):
                check_embedding_dimension_match()
    finally:
        Settings._embed_model = None
