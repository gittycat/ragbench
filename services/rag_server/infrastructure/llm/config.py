"""
LLM provider types and internal configuration container.

Provides:
- LLMProvider enum: Canonical list of supported providers
- LLMConfig dataclass: Internal config container for provider functions

Configuration is loaded from config/models.yml via infrastructure.config.models_config.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    MOONSHOT = "moonshot"


@dataclass
class LLMConfig:
    """Internal configuration container for LLM provider functions."""

    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 120.0
    keep_alive: Optional[str] = None  # Ollama-only

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load LLM config from config/models.yml."""
        from infrastructure.config.models_config import get_models_config

        try:
            models_config = get_models_config()
            llm_config = models_config.llm

            try:
                provider = LLMProvider(llm_config.provider)
            except ValueError:
                valid_providers = ", ".join(p.value for p in LLMProvider)
                raise ValueError(
                    f"Invalid LLM provider in config: '{llm_config.provider}'. "
                    f"Valid options: {valid_providers}"
                )

            return cls(
                provider=provider,
                model=llm_config.model,
                api_key=llm_config.api_key,
                base_url=llm_config.base_url,
                timeout=llm_config.timeout,
                keep_alive=llm_config.keep_alive,
            )
        except Exception as e:
            logger.error(f"Failed to load LLM config from file: {e}")
            raise

    def __repr__(self) -> str:
        """Safe repr that doesn't expose API key."""
        return (
            f"LLMConfig(provider={self.provider.value}, model={self.model}, "
            f"base_url={self.base_url}, timeout={self.timeout})"
        )
