"""
LLM Client Factory - single entry point for LLM instantiation.

Usage:
    from infrastructure.llm.factory import get_llm_client
    llm = get_llm_client()  # Returns configured LLM based on env vars

    # Or use dependency injection directly:
    from infrastructure.llm.factory import LLMClientManager
    manager = LLMClientManager()
    llm = manager.get_client()
"""
import logging

from llama_index.core.llms import LLM
from .config import LLMConfig, LLMProvider
from .providers import (
    create_ollama_client,
    create_openai_client,
    create_anthropic_client,
    create_google_client,
    create_deepseek_client,
)

logger = logging.getLogger(__name__)


class LLMClientManager:
    """
    Manages LLM client lifecycle with lazy initialization.

    Supports dependency injection for testing and reconfiguration.
    """

    def __init__(self, config: LLMConfig | None = None):
        """
        Initialize LLM client manager.

        Args:
            config: Optional LLMConfig. If None, loads from environment.
        """
        self._config = config
        self._client: LLM | None = None

    def get_client(self) -> LLM:
        """
        Get or create LLM client.

        Lazy initialization - client is created on first access.

        Returns:
            Configured LLM instance
        """
        if self._client is not None:
            return self._client

        config = self._config or LLMConfig.from_env()
        logger.info(f"[LLM] Initializing {config.provider.value} provider: {config.model}")

        # Provider-to-creator mapping
        creators = {
            LLMProvider.OLLAMA: create_ollama_client,
            LLMProvider.OPENAI: create_openai_client,
            LLMProvider.ANTHROPIC: create_anthropic_client,
            LLMProvider.GOOGLE: create_google_client,
            LLMProvider.DEEPSEEK: create_deepseek_client,
            LLMProvider.MOONSHOT: create_openai_client,  # Uses OpenAI-compatible API
        }

        creator = creators.get(config.provider)
        if not creator:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

        self._client = creator(config)

        if config.provider == LLMProvider.OLLAMA:
            logger.info(f"[LLM] Ollama client initialized: keep_alive={config.keep_alive}")
        elif config.provider == LLMProvider.MOONSHOT:
            logger.info(f"[LLM] Moonshot (OpenAI-compatible) client initialized: base_url={config.base_url}")
        else:
            logger.info(f"[LLM] {config.provider.value.capitalize()} client initialized successfully")

        return self._client

    def reset(self) -> None:
        """Reset the client instance. Useful for testing or reconfiguration."""
        self._client = None
        logger.info("[LLM] Client reset")


# Global instance for backward compatibility
_default_manager = LLMClientManager()


def get_llm_client() -> LLM:
    """
    Get or create LLM client using default manager.

    Backward-compatible convenience function.
    For dependency injection, use LLMClientManager directly.

    Returns:
        Configured LLM instance
    """
    return _default_manager.get_client()


def get_llm_config() -> LLMConfig:
    """Get current LLM configuration without creating client."""
    return LLMConfig.from_env()


def reset_llm_client() -> None:
    """Reset the default LLM client. Useful for testing."""
    _default_manager.reset()
