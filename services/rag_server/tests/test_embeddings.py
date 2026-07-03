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
)


def create_mock_models_config():
    """Create a mock ModelsConfig for tests (uses ollama, no API key required)."""
    return ModelsConfig(
        llm=LLMConfig(
            provider="ollama",
            model="gemma3:4b",
            base_url="http://localhost:11434",
            timeout=120,
            keep_alive="10m",
        ),
        embedding=EmbeddingConfig(
            provider="ollama",
            model="nomic-embed-text:latest",
            base_url="http://localhost:11434",
        ),
        eval=EvalConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="test-key",
        ),
        reranker=RerankerConfig(enabled=True),
        retrieval=RetrievalConfig(),
    )


@pytest.fixture
def mock_config():
    """Provide mock models config for embedding tests."""
    config = create_mock_models_config()
    with patch("infrastructure.llm.embeddings.get_models_config", return_value=config):
        yield config


def test_embedding_function_initializes(mock_config):
    """Ollama provider should dispatch to OllamaEmbedding with configured model"""
    from infrastructure.llm.embeddings import get_embedding_function

    with patch("llama_index.embeddings.ollama.OllamaEmbedding") as mock_embeddings:
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance

        embedding_fn = get_embedding_function()
        assert embedding_fn is not None

        mock_embeddings.assert_called_once()
        call_kwargs = mock_embeddings.call_args.kwargs
        assert call_kwargs["model_name"] == "nomic-embed-text:latest"
        assert call_kwargs["embed_batch_size"] == 64


def test_embedding_function_has_correct_endpoint(mock_config):
    """Embedding function should use correct Ollama endpoint"""
    from infrastructure.llm.embeddings import get_embedding_function

    with patch("llama_index.embeddings.ollama.OllamaEmbedding") as mock_embeddings:
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance

        embedding_fn = get_embedding_function()

        call_kwargs = mock_embeddings.call_args.kwargs
        assert "base_url" in call_kwargs
        assert "11434" in call_kwargs["base_url"]


def test_embedding_function_with_custom_config():
    """Test embedding function respects config values"""
    from infrastructure.llm.embeddings import get_embedding_function

    config = create_mock_models_config()
    config.embedding.base_url = "http://custom-ollama:12345"

    with patch("infrastructure.llm.embeddings.get_models_config", return_value=config):
        with patch("llama_index.embeddings.ollama.OllamaEmbedding") as mock_embeddings:
            mock_instance = MagicMock()
            mock_embeddings.return_value = mock_instance

            embedding_fn = get_embedding_function()

            call_kwargs = mock_embeddings.call_args.kwargs
            assert call_kwargs["base_url"] == "http://custom-ollama:12345"


def test_embedding_function_generates_embeddings(mock_config):
    """Embedding function should generate 768-dimensional embeddings"""
    from infrastructure.llm.embeddings import get_embedding_function

    with patch("llama_index.embeddings.ollama.OllamaEmbedding") as mock_embeddings_class:
        mock_instance = MagicMock()
        mock_instance.get_text_embedding.return_value = [0.1] * 768
        mock_embeddings_class.return_value = mock_instance

        embedding_fn = get_embedding_function()
        embedding = embedding_fn.get_text_embedding("test document")

        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)
        mock_instance.get_text_embedding.assert_called_once()


def test_embedding_function_handles_multiple_texts(mock_config):
    """Embedding function should handle batch processing"""
    from infrastructure.llm.embeddings import get_embedding_function

    with patch("llama_index.embeddings.ollama.OllamaEmbedding") as mock_embeddings_class:
        mock_instance = MagicMock()
        mock_instance.get_text_embedding_batch.return_value = [
            [0.1] * 768,
            [0.2] * 768,
            [0.3] * 768,
        ]
        mock_embeddings_class.return_value = mock_instance

        embedding_fn = get_embedding_function()
        texts = ["text1", "text2", "text3"]
        embeddings = embedding_fn.get_text_embedding_batch(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 768 for emb in embeddings)
        mock_instance.get_text_embedding_batch.assert_called_once_with(texts)


def test_embedding_function_dispatches_openai():
    """OpenAI provider should dispatch to OpenAIEmbedding with mapped params"""
    from infrastructure.llm.embeddings import get_embedding_function

    config = create_mock_models_config()
    config.embedding = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        requires_api_key=True,
    )

    with patch("infrastructure.llm.embeddings.get_models_config", return_value=config):
        with patch("llama_index.embeddings.openai.OpenAIEmbedding") as mock_embeddings_class:
            mock_instance = MagicMock()
            mock_embeddings_class.return_value = mock_instance

            embedding_fn = get_embedding_function()
            assert embedding_fn is not None

            call_kwargs = mock_embeddings_class.call_args.kwargs
            assert call_kwargs["model"] == "text-embedding-3-small"
            assert call_kwargs["api_key"] == "sk-test"
            assert call_kwargs["api_base"] == "https://api.openai.com/v1"
            assert call_kwargs["embed_batch_size"] == 100


def test_embedding_function_unsupported_provider():
    """Unknown provider should raise a clear error"""
    from infrastructure.llm.embeddings import get_embedding_function

    config = create_mock_models_config()
    config.embedding = EmbeddingConfig(provider="bogus", model="whatever")

    with patch("infrastructure.llm.embeddings.get_models_config", return_value=config):
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            get_embedding_function()
