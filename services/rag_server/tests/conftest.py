"""
Shared pytest fixtures and configuration for RAG server tests.
"""
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set minimal env vars required for module imports (unit tests mock everything)
# These are only used if not already set (e.g., in Docker or integration tests)
_DEFAULT_ENV = {
    "DATABASE_URL": "postgresql+asyncpg://raguser:ragpass@localhost:5432/ragbench",
    "OLLAMA_URL": "http://localhost:11434",
    "EMBEDDING_MODEL": "nomic-embed-text:latest",
    "LLM_MODEL": "gemma3:4b",
}

for key, value in _DEFAULT_ENV.items():
    if key not in os.environ:
        os.environ[key] = value


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires docker services)",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )
    parser.addoption(
        "--run-eval",
        action="store_true",
        default=False,
        help="Run evaluation tests (requires ANTHROPIC_API_KEY)",
    )
    parser.addoption(
        "--eval-samples",
        action="store",
        default=None,
        type=int,
        help="Number of samples to evaluate (default: all)",
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires docker services)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (> 30s)"
    )
    config.addinivalue_line(
        "markers", "eval: mark test as evaluation test (requires ANTHROPIC_API_KEY)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and command line options."""

    # Skip integration tests unless --run-integration is provided
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    # Skip slow tests unless --run-slow is provided
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Skip eval tests unless --run-eval is provided
    if not config.getoption("--run-eval"):
        skip_eval = pytest.mark.skip(reason="need --run-eval option to run")
        for item in items:
            if "eval" in item.keywords:
                item.add_marker(skip_eval)


@pytest.fixture(scope="session")
def integration_env():
    """
    Set up environment variables for integration tests.
    These should match docker-compose.yml settings.
    """
    env_vars = {
        "DATABASE_URL": os.getenv("DATABASE_URL", "postgresql+asyncpg://raguser:ragpass@localhost:5432/ragbench"),
        "OLLAMA_URL": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest"),
        "LLM_MODEL": os.getenv("LLM_MODEL", "gemma3:4b"),
        "ENABLE_HYBRID_SEARCH": "true",
        "ENABLE_RERANKER": "true",
        "ENABLE_CONTEXTUAL_RETRIEVAL": "false",  # Disable for faster tests
        "RETRIEVAL_TOP_K": "10",
        "LOG_LEVEL": "DEBUG",
    }

    # Set env vars
    original_env = {}
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    yield env_vars

    # Restore original env vars
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def create_mock_models_config():
    """Create a mock ModelsConfig for unit tests (uses ollama, no API key required)."""
    from infrastructure.config.models_config import (
        ModelsConfig,
        LLMConfig,
        EmbeddingConfig,
        EvalConfig,
        RerankerConfig,
        RetrievalConfig,
    )

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
            api_key="test-key",  # Mock API key for tests
        ),
        reranker=RerankerConfig(
            enabled=True,
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=5,
        ),
        retrieval=RetrievalConfig(
            top_k=10,
            enable_hybrid_search=True,
            rrf_k=60,
            enable_contextual_retrieval=False,
        ),
    )


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset the models config singleton before each test."""
    from infrastructure.config.models_config import reset_models_config

    reset_models_config()
    yield
    reset_models_config()


@pytest.fixture
def mock_models_config():
    """Provide a mock ModelsConfig and patch get_models_config to return it."""
    mock_config = create_mock_models_config()

    with patch(
        "infrastructure.config.models_config.get_models_config",
        return_value=mock_config,
    ):
        # Also patch the singleton directly
        with patch(
            "infrastructure.config.models_config._models_config",
            mock_config,
        ):
            yield mock_config
