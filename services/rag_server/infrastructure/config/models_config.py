"""Model configuration management using Pydantic for type safety and validation."""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class LLMConfig(BaseModel):
    """Configuration for the main LLM."""

    provider: Literal["ollama", "openai", "anthropic", "google", "deepseek", "moonshot"]
    model: str
    base_url: str | None = None
    timeout: int = 120
    keep_alive: str | None = None
    api_key: str | None = None

    @field_validator("model")
    @classmethod
    def model_must_exist(cls, v: str) -> str:
        """Validate that model name is not empty."""
        if not v or not v.strip():
            raise ValueError("LLM model name is required and cannot be empty")
        return v

    def validate_provider_requirements(self) -> None:
        """Validate that required fields are present for the selected provider."""
        if self.provider != "ollama" and not self.api_key:
            raise ValueError(
                f"API key is required for provider '{self.provider}'. "
                f"Set LLM_API_KEY environment variable."
            )


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding model."""

    provider: str
    model: str
    base_url: str | None = None

    @field_validator("model")
    @classmethod
    def model_must_exist(cls, v: str) -> str:
        """Validate that model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Embedding model name is required and cannot be empty")
        return v


class EvalModelConfig(BaseModel):
    """Configuration for an evaluation model (without settings)."""

    provider: str
    model: str
    api_key: str | None = None

    @field_validator("model")
    @classmethod
    def model_must_exist(cls, v: str) -> str:
        """Validate that model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Eval model name is required and cannot be empty")
        return v

    def validate_provider_requirements(self) -> None:
        """Validate that required fields are present for the selected provider."""
        if self.provider != "ollama" and not self.api_key:
            raise ValueError(
                f"API key is required for eval provider '{self.provider}'. "
                f"Set ANTHROPIC_API_KEY environment variable."
            )


class EvalSettings(BaseModel):
    """Evaluation settings (non-model-specific)."""

    citation_scope: Literal["retrieved", "explicit"] = "retrieved"
    citation_format: Literal["numeric"] = "numeric"
    abstention_phrases: list[str] = Field(
        default_factory=lambda: [
            "I don't have enough information to answer this question.",
            "I do not have enough information to answer this question.",
            "I don't have enough information to answer the question.",
            "I do not have enough information to answer the question.",
            "Not enough information to answer.",
            "Insufficient information to answer.",
        ]
    )


class EvalConfig(BaseModel):
    """Combined evaluation configuration (model + settings)."""

    provider: str
    model: str
    api_key: str | None = None
    citation_scope: Literal["retrieved", "explicit"] = "retrieved"
    citation_format: Literal["numeric"] = "numeric"
    abstention_phrases: list[str] = Field(
        default_factory=lambda: [
            "I don't have enough information to answer this question.",
            "I do not have enough information to answer this question.",
            "I don't have enough information to answer the question.",
            "I do not have enough information to answer the question.",
            "Not enough information to answer.",
            "Insufficient information to answer.",
        ]
    )

    @field_validator("model")
    @classmethod
    def model_must_exist(cls, v: str) -> str:
        """Validate that model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Eval model name is required and cannot be empty")
        return v

    def validate_provider_requirements(self) -> None:
        """Validate that required fields are present for the selected provider."""
        if self.provider != "ollama" and not self.api_key:
            raise ValueError(
                f"API key is required for eval provider '{self.provider}'. "
                f"Set ANTHROPIC_API_KEY environment variable."
            )


class RerankerModelConfig(BaseModel):
    """Configuration for a reranker model (without enabled flag)."""

    model: str
    top_n: int = 5


class RerankerSettings(BaseModel):
    """Reranker settings (non-model-specific)."""

    enabled: bool = True


class RerankerConfig(BaseModel):
    """Combined reranker configuration (model + settings)."""

    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5


class RetrievalConfig(BaseModel):
    """Configuration for retrieval settings."""

    top_k: int = 10
    enable_hybrid_search: bool = True
    rrf_k: int = 60
    enable_contextual_retrieval: bool = False


class ActiveModels(BaseModel):
    """Active model selection."""

    inference: str
    embedding: str
    eval: str
    reranker: str


class ModelDefinitions(BaseModel):
    """All available model definitions."""

    inference: dict[str, dict[str, Any]]
    embedding: dict[str, dict[str, Any]]
    eval: dict[str, dict[str, Any]]
    reranker: dict[str, dict[str, Any]]


class ModelsConfig(BaseModel):
    """Root configuration for all models and retrieval settings."""

    llm: LLMConfig
    embedding: EmbeddingConfig
    eval: EvalConfig
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "ModelsConfig":
        """Load configuration from YAML file and inject secrets from environment.

        Args:
            config_path: Path to the models.yml file. If None, searches in standard locations.

        Returns:
            ModelsConfig instance with secrets injected.

        Raises:
            FileNotFoundError: If config file is not found.
            ValueError: If required secrets are missing or invalid.
        """
        # Determine config file path
        if config_path is None:
            # Try multiple standard locations
            possible_paths = [
                Path("/app/config.yml"),  # Docker path
                Path(__file__).parent.parent.parent.parent.parent
                / "config.yml",  # Development path
            ]
            config_path = next((p for p in possible_paths if p.exists()), None)
            if config_path is None:
                raise FileNotFoundError(
                    f"config.yml not found in standard locations: {possible_paths}"
                )
        else:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load YAML config
        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Check if using new format (with 'models' and 'active' sections)
        if "models" in data and "active" in data:
            # New format - resolve model references
            resolved_data = cls._resolve_model_references(data)
        else:
            # Legacy format - use as-is
            resolved_data = data

        # Inject secrets from environment variables
        llm_api_key = os.getenv("LLM_API_KEY")
        if "llm" in resolved_data and llm_api_key:
            resolved_data["llm"]["api_key"] = llm_api_key

        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if "eval" in resolved_data and anthropic_api_key:
            resolved_data["eval"]["api_key"] = anthropic_api_key

        # Create and validate config
        config = cls(**resolved_data)

        # Run provider-specific validations
        config.llm.validate_provider_requirements()

        return config

    @staticmethod
    def _resolve_model_references(data: dict[str, Any]) -> dict[str, Any]:
        """Resolve model references from new config format.

        Args:
            data: Raw YAML data with 'models' and 'active' sections

        Returns:
            Resolved configuration data in legacy format

        Raises:
            ValueError: If referenced model is not defined
        """
        models = data.get("models", {})
        active = data.get("active", {})

        resolved = {}

        # Resolve Inference
        inference_key = active.get("inference")
        if inference_key and inference_key in models.get("inference", {}):
            resolved["llm"] = models["inference"][inference_key].copy()
        else:
            raise ValueError(
                f"Active inference model '{inference_key}' not found in models.inference definitions"
            )

        # Resolve Embedding
        embedding_key = active.get("embedding")
        if embedding_key and embedding_key in models.get("embedding", {}):
            resolved["embedding"] = models["embedding"][embedding_key].copy()
        else:
            raise ValueError(
                f"Active embedding model '{embedding_key}' not found in models.embedding definitions"
            )

        # Resolve Eval (merge model config with eval settings)
        eval_key = active.get("eval")
        if eval_key and eval_key in models.get("eval", {}):
            resolved["eval"] = models["eval"][eval_key].copy()
            # Merge eval settings if present
            if "eval" in data:
                eval_settings = data["eval"]
                if "citation_scope" in eval_settings:
                    resolved["eval"]["citation_scope"] = eval_settings["citation_scope"]
                if "citation_format" in eval_settings:
                    resolved["eval"]["citation_format"] = eval_settings["citation_format"]
                if "abstention_phrases" in eval_settings:
                    resolved["eval"]["abstention_phrases"] = eval_settings[
                        "abstention_phrases"
                    ]
        else:
            raise ValueError(
                f"Active eval model '{eval_key}' not found in models.eval definitions"
            )

        # Resolve Reranker (merge model config with reranker settings)
        reranker_key = active.get("reranker")
        if reranker_key and reranker_key in models.get("reranker", {}):
            resolved["reranker"] = models["reranker"][reranker_key].copy()
            # Merge reranker settings if present
            if "reranker" in data:
                reranker_settings = data["reranker"]
                if "enabled" in reranker_settings:
                    resolved["reranker"]["enabled"] = reranker_settings["enabled"]
        else:
            raise ValueError(
                f"Active reranker model '{reranker_key}' not found in models.reranker definitions"
            )

        # Copy retrieval settings unchanged
        if "retrieval" in data:
            resolved["retrieval"] = data["retrieval"]

        return resolved


class ModelsConfigManager:
    """
    Manages ModelsConfig lifecycle with lazy initialization.

    Supports dependency injection for testing and reconfiguration.
    """

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize models config manager.

        Args:
            config_path: Optional path to config file. If None, searches standard locations.
        """
        self._config_path = config_path
        self._config: ModelsConfig | None = None

    def get_config(self) -> ModelsConfig:
        """
        Get or load ModelsConfig.

        Lazy initialization - config is loaded on first access.

        Returns:
            ModelsConfig instance
        """
        if self._config is None:
            self._config = ModelsConfig.load(self._config_path)
        return self._config

    def reset(self) -> None:
        """Reset the config instance. Useful for testing."""
        self._config = None


# Global instance for backward compatibility
_default_manager = ModelsConfigManager()


def get_models_config(config_path: str | Path | None = None) -> ModelsConfig:
    """
    Get or load ModelsConfig using default manager.

    Backward-compatible convenience function.
    For dependency injection, use ModelsConfigManager directly.

    Args:
        config_path: Optional path to config file. Only used on first call.

    Returns:
        ModelsConfig instance
    """
    # Note: config_path only affects first call (lazy initialization)
    if _default_manager._config is None and config_path is not None:
        _default_manager._config_path = config_path
    return _default_manager.get_config()


def reset_models_config() -> None:
    """Reset the default models config. Useful for testing."""
    _default_manager.reset()
