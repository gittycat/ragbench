import pytest

from infrastructure.config.models_config import (
    ModelsConfig,
    LLMConfig,
    EmbeddingConfig,
    EvalConfig,
    RerankerConfig,
    RetrievalConfig,
    PiiConfig,
)


def build_config(embedding_provider: str, pii_enabled: bool) -> ModelsConfig:
    return ModelsConfig(
        llm=LLMConfig(provider="ollama", model="gemma3:4b"),
        embedding=EmbeddingConfig(provider=embedding_provider, model="some-embed-model"),
        eval=EvalConfig(provider="anthropic", model="claude-sonnet-4-20250514", api_key="test-key"),
        reranker=RerankerConfig(),
        retrieval=RetrievalConfig(),
        pii=PiiConfig(enabled=pii_enabled),
    )


def test_privacy_posture_allows_local_embedding_with_pii_enabled():
    config = build_config("ollama", pii_enabled=True)
    config.validate_privacy_posture()  # should not raise


def test_privacy_posture_allows_cloud_embedding_when_pii_disabled():
    config = build_config("openai", pii_enabled=False)
    config.validate_privacy_posture()  # should not raise, user's explicit choice


def test_privacy_posture_rejects_cloud_embedding_with_pii_enabled():
    config = build_config("openai", pii_enabled=True)
    with pytest.raises(ValueError, match="not local"):
        config.validate_privacy_posture()


def test_pii_disabled_by_default():
    assert PiiConfig().enabled is False
