"""Model configuration management using Pydantic for type safety and validation."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from app.settings import get_api_key_for_provider
class LLMConfig(BaseModel):
    """Configuration for the main LLM."""

    provider: str
    model: str
    base_url: str | None = None
    timeout: int = 120
    keep_alive: str | None = None
    api_key: str | None = None
    requires_api_key: bool = False

    @field_validator("model")
    @classmethod
    def model_must_exist(cls, v: str) -> str:
        """Validate that model name is not empty."""
        if not v or not v.strip():
            raise ValueError("LLM model name is required and cannot be empty")
        return v

    def validate_provider_requirements(self) -> None:
        """Validate that required fields are present for the selected provider."""
        if self.requires_api_key and not self.api_key:
            raise ValueError(
                f"API key is required for provider '{self.provider}'. "
                f"Mount /run/secrets/{self.provider.upper()}_API_KEY."
            )


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding model."""

    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    requires_api_key: bool = False

    @field_validator("model")
    @classmethod
    def model_must_exist(cls, v: str) -> str:
        """Validate that model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Embedding model name is required and cannot be empty")
        return v

    def validate_provider_requirements(self) -> None:
        """Validate that required fields are present for the selected provider."""
        if self.requires_api_key and not self.api_key:
            raise ValueError(
                f"API key is required for embedding provider '{self.provider}'. "
                f"Mount /run/secrets/{self.provider.upper()}_API_KEY."
            )


class EvalModelConfig(BaseModel):
    """Configuration for an evaluation model (without settings)."""

    provider: str
    model: str
    api_key: str | None = None
    requires_api_key: bool = False

    @field_validator("model")
    @classmethod
    def model_must_exist(cls, v: str) -> str:
        """Validate that model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Eval model name is required and cannot be empty")
        return v

    def validate_provider_requirements(self) -> None:
        """Validate that required fields are present for the selected provider."""
        if self.requires_api_key and not self.api_key:
            raise ValueError(
                f"API key is required for eval provider '{self.provider}'. "
                f"Mount /run/secrets/{self.provider.upper()}_API_KEY."
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
    requires_api_key: bool = False
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
        if self.requires_api_key and not self.api_key:
            raise ValueError(
                f"API key is required for eval provider '{self.provider}'. "
                f"Mount /run/secrets/{self.provider.upper()}_API_KEY."
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


class CitationInstructions(BaseModel):
    """Citation instruction templates by format."""

    numeric: str = (
        "\n- Add numeric citations in square brackets like [1], [2] that map to the "
        "order of context chunks provided above."
    )


class PromptConfig(BaseModel):
    """Configuration for RAG pipeline prompts."""

    system: str = (
        "You are a professional assistant providing accurate answers based on document context. "
        "Be direct and concise. Avoid conversational fillers like 'Let me explain', 'Okay', 'Well', or 'Sure'. "
        "Start responses immediately with the answer. "
        "Use bullet points for lists when appropriate."
    )
    context: str = (
        "Context from retrieved documents:\n"
        "{context_str}\n\n"
        "Instructions:\n"
        "- Answer using ONLY the context provided above\n"
        "- If the context does not contain sufficient information, respond: \"I don't have enough information to answer this question.\"\n"
        "- Never use prior knowledge or make assumptions beyond what is explicitly stated\n"
        "- Be specific and cite details from the context when relevant\n"
        "- Use citations consistently when referencing facts{citation_instructions}\n"
        "- Previous conversation context is available for reference\n\n"
        "Provide a direct, accurate answer based on the context:"
    )
    citation_instructions: CitationInstructions = Field(default_factory=CitationInstructions)
    condense: str | None = None  # None = use LlamaIndex default
    contextual_prefix: str = (
        "Document: {document_name} ({document_type})\n\n"
        "Chunk content:\n"
        "{chunk_preview}\n\n"
        "Provide a concise 1-2 sentence context for this chunk, explaining what document it's from and what topic it discusses.\n"
        'Format: "This section from [document/topic] discusses [specific topic/concept]."\n\n'
        "Context (1-2 sentences only):"
    )


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
    prompts: PromptConfig = Field(default_factory=PromptConfig)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "ModelsConfig":
        """Load configuration from YAML file and inject secrets from /run/secrets.

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

        # Inject API keys from secrets based on requires_api_key flag
        for key in ["llm", "embedding", "eval"]:
            if key in resolved_data and resolved_data[key].get("requires_api_key"):
                provider = resolved_data[key].get("provider")
                if provider:
                    api_key = get_api_key_for_provider(provider)
                    if api_key:
                        resolved_data[key]["api_key"] = api_key

        # Create and validate config
        config = cls(**resolved_data)

        # Run provider-specific validations
        config.llm.validate_provider_requirements()
        config.embedding.validate_provider_requirements()
        config.eval.validate_provider_requirements()

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

        # Copy prompts unchanged
        if "prompts" in data:
            resolved["prompts"] = data["prompts"]

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
