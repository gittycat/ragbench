import logging
from typing import Any

from llama_index.core.embeddings import BaseEmbedding

from infrastructure.config.models_config import EmbeddingConfig, get_models_config

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = {"ollama": 64, "openai": 100}

# Provider configuration: maps provider to (module_path, class_name, param_mapping)
_PROVIDER_CONFIG: dict[str, tuple[str, str, dict[str, str | None]]] = {
    "ollama": (
        "llama_index.embeddings.ollama",
        "OllamaEmbedding",
        {"model": "model_name", "base_url": None},
    ),
    "openai": (
        "llama_index.embeddings.openai",
        "OpenAIEmbedding",
        {"model": None, "api_key": None, "base_url": "api_base"},
    ),
}


def create_embedding_function(config: EmbeddingConfig) -> BaseEmbedding:
    """Create embedding client based on provider configuration."""
    provider_config = _PROVIDER_CONFIG.get(config.provider)
    if not provider_config:
        raise ValueError(f"Unsupported embedding provider: {config.provider}")

    module_path, class_name, param_mapping = provider_config

    import importlib

    module = importlib.import_module(module_path)
    embedding_class = getattr(module, class_name)

    kwargs: dict[str, Any] = {}
    for config_field, param_name in param_mapping.items():
        value = getattr(config, config_field, None)
        if value is not None:
            key = param_name if param_name else config_field
            kwargs[key] = value

    batch_size = config.embed_batch_size or _DEFAULT_BATCH_SIZE.get(config.provider, 64)
    kwargs["embed_batch_size"] = batch_size

    return embedding_class(**kwargs)


def get_embedding_function() -> BaseEmbedding:
    config = get_models_config()
    embedding_config = config.embedding

    logger.info(f"[EMBEDDINGS] Initializing {embedding_config.provider} embedding")
    logger.info(f"[EMBEDDINGS] Model: {embedding_config.model}")

    embedding_function = create_embedding_function(embedding_config)

    logger.info("[EMBEDDINGS] Embedding function initialized successfully")
    return embedding_function
