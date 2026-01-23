"""
LLM Infrastructure - Multi-provider LLM support.

Supported providers:
    - Ollama (local inference)
    - OpenAI
    - Anthropic
    - Google Gemini
    - DeepSeek
    - Moonshot (Kimi K2)

Configure via environment variables:
    LLM_PROVIDER: Provider name (default: ollama)
    LLM_MODEL: Model name (required)
    LLM_API_KEY: API key (required for cloud providers)
    LLM_BASE_URL: Custom endpoint (optional)
    LLM_TIMEOUT: Request timeout in seconds (default: 120)
"""
from .factory import get_llm_client, get_llm_config, reset_llm_client, LLMClientManager
from .prompts import get_system_prompt, get_context_prompt, get_condense_prompt
from .config import LLMConfig, LLMProvider
from .embeddings import get_embedding_function

__all__ = [
    # Factory
    "get_llm_client",
    "get_llm_config",
    "reset_llm_client",
    "LLMClientManager",  # For dependency injection
    # Prompts
    "get_system_prompt",
    "get_context_prompt",
    "get_condense_prompt",
    # Config
    "LLMConfig",
    "LLMProvider",
    # Embeddings
    "get_embedding_function",
]
