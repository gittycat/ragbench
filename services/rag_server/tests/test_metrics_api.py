"""Tests for the metrics and configuration API endpoints.

Tests the following endpoints:
- GET /metrics/system
- GET /metrics/models
- GET /metrics/retrieval
- GET /metrics/eval/definitions
- GET /metrics/eval/runs
- GET /metrics/eval/summary
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
import json

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


from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_models_config_fixture():
    """Auto-use fixture to mock models config for all tests in this file."""
    mock_config = create_mock_models_config()
    with patch(
        "infrastructure.config.models_config.get_models_config",
        return_value=mock_config,
    ):
        with patch(
            "infrastructure.config.models_config._default_manager._config",
            mock_config,
        ):
            yield mock_config


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_ollama():
    """Mock Ollama API responses."""
    with patch('services.metrics.httpx.AsyncClient') as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "size": 4000000000,  # 4GB
            "parameters": "4B",
        }

        # Create async context manager mock
        async_client_instance = AsyncMock()
        async_client_instance.post.return_value = mock_response
        async_client_instance.get.return_value = mock_response
        async_client_instance.__aenter__.return_value = async_client_instance
        async_client_instance.__aexit__.return_value = None

        mock_client.return_value = async_client_instance

        yield mock_client


@pytest.fixture
def mock_system_metrics():
    """Mock the get_system_metrics function to return test data."""
    from schemas.metrics import (
        SystemMetrics,
        ModelsConfig,
        ModelInfo,
        ModelSize,
        RetrievalConfig,
        HybridSearchConfig,
        BM25Config,
        VectorSearchConfig,
        ContextualRetrievalConfig,
        RerankerConfig,
    )

    mock_metrics = SystemMetrics(
        system_name="ragbench",
        version="1.0.0",
        models=ModelsConfig(
            llm=ModelInfo(
                name="gemma3:4b",
                provider="Ollama",
                model_type="llm",
                is_local=True,
                size=ModelSize(parameters="4B"),
                reference_url="https://ollama.com/library/gemma3",
                description="Test LLM",
                status="available",
            ),
            embedding=ModelInfo(
                name="nomic-embed-text:latest",
                provider="Ollama",
                model_type="embedding",
                is_local=True,
                size=ModelSize(parameters="137M"),
                reference_url="https://ollama.com/library/nomic-embed-text",
                description="Test embeddings",
                status="available",
            ),
            reranker=ModelInfo(
                name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                provider="HuggingFace",
                model_type="reranker",
                is_local=True,
                size=ModelSize(parameters="22M", disk_size_mb=80),
                reference_url="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2",
                description="Test reranker",
                status="available",
            ),
            eval=ModelInfo(
                name="claude-sonnet-4-20250514",
                provider="Anthropic",
                model_type="eval",
                is_local=False,
                reference_url="https://docs.anthropic.com",
                description="Test eval",
                status="available",
            ),
        ),
        retrieval=RetrievalConfig(
            retrieval_top_k=10,
            final_top_n=5,
            hybrid_search=HybridSearchConfig(
                enabled=True,
                bm25=BM25Config(enabled=True),
                vector=VectorSearchConfig(
                    enabled=True,
                    chunk_size=500,
                    chunk_overlap=50,
                    vector_store="ChromaDB",
                    collection_name="documents",
                ),
                fusion_method="reciprocal_rank_fusion",
                rrf_k=60,
            ),
            contextual_retrieval=ContextualRetrievalConfig(enabled=False),
            reranker=RerankerConfig(
                enabled=True,
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_n=5,
            ),
        ),
        document_count=2,
        chunk_count=15,
        health_status="healthy",
        component_status={"chromadb": "healthy", "redis": "healthy", "ollama": "healthy"},
    )

    async def mock_get_system_metrics():
        return mock_metrics

    with patch('services.metrics.get_system_metrics', mock_get_system_metrics):
        yield mock_metrics


@pytest.fixture
def mock_env_vars():
    """Mock environment variables."""
    env_vars = {
        'LLM_MODEL': 'gemma3:4b',
        'EMBEDDING_MODEL': 'nomic-embed-text:latest',
        'EVAL_MODEL': 'claude-sonnet-4-20250514',
        'ENABLE_RERANKER': 'true',
        'RERANKER_MODEL': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        'ENABLE_HYBRID_SEARCH': 'true',
        'RRF_K': '60',
        'RETRIEVAL_TOP_K': '10',
        'ENABLE_CONTEXTUAL_RETRIEVAL': 'false',
        'OLLAMA_URL': 'http://localhost:11434',
        'CHROMADB_URL': 'http://localhost:8000',
        'REDIS_URL': 'redis://localhost:6379/0',
    }
    with patch.dict('os.environ', env_vars):
        yield env_vars


# ============================================================================
# Models Endpoint Tests
# ============================================================================

def test_models_endpoint_returns_200(mock_ollama, mock_env_vars):
    """GET /metrics/models should return 200."""
    response = client.get("/metrics/models")
    assert response.status_code == 200


def test_models_endpoint_returns_llm_info(mock_ollama, mock_env_vars):
    """GET /metrics/models should include LLM model info."""
    response = client.get("/metrics/models")
    data = response.json()

    assert "llm" in data
    assert data["llm"]["name"] == "gemma3:4b"
    assert data["llm"]["provider"] == "Ollama"
    assert data["llm"]["model_type"] == "llm"
    assert data["llm"]["is_local"] is True


def test_models_endpoint_returns_embedding_info(mock_ollama, mock_env_vars):
    """GET /metrics/models should include embedding model info."""
    response = client.get("/metrics/models")
    data = response.json()

    assert "embedding" in data
    assert data["embedding"]["name"] == "nomic-embed-text:latest"
    assert data["embedding"]["provider"] == "Ollama"
    assert data["embedding"]["model_type"] == "embedding"


def test_models_endpoint_returns_reranker_info(mock_ollama, mock_env_vars):
    """GET /metrics/models should include reranker model info when enabled."""
    response = client.get("/metrics/models")
    data = response.json()

    assert "reranker" in data
    assert data["reranker"]["name"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert data["reranker"]["provider"] == "HuggingFace"


def test_models_endpoint_returns_eval_info(mock_ollama, mock_env_vars):
    """GET /metrics/models should include eval model info."""
    response = client.get("/metrics/models")
    data = response.json()

    assert "eval" in data
    assert data["eval"]["name"] == "claude-sonnet-4-20250514"
    assert data["eval"]["provider"] == "Anthropic"


def test_models_endpoint_includes_reference_urls(mock_ollama, mock_env_vars):
    """GET /metrics/models should include reference URLs for models."""
    response = client.get("/metrics/models")
    data = response.json()

    assert data["llm"]["reference_url"] is not None
    assert "ollama.com" in data["llm"]["reference_url"]


# ============================================================================
# Retrieval Config Endpoint Tests
# ============================================================================

def test_retrieval_endpoint_returns_200(mock_env_vars):
    """GET /metrics/retrieval should return 200."""
    response = client.get("/metrics/retrieval")
    assert response.status_code == 200


def test_retrieval_endpoint_returns_hybrid_config(mock_env_vars):
    """GET /metrics/retrieval should include hybrid search config."""
    response = client.get("/metrics/retrieval")
    data = response.json()

    assert "hybrid_search" in data
    assert data["hybrid_search"]["enabled"] is True
    assert data["hybrid_search"]["rrf_k"] == 60
    assert data["hybrid_search"]["fusion_method"] == "reciprocal_rank_fusion"


def test_retrieval_endpoint_returns_bm25_config(mock_env_vars):
    """GET /metrics/retrieval should include BM25 config."""
    response = client.get("/metrics/retrieval")
    data = response.json()

    assert "hybrid_search" in data
    assert "bm25" in data["hybrid_search"]
    assert data["hybrid_search"]["bm25"]["enabled"] is True


def test_retrieval_endpoint_returns_reranker_config(mock_env_vars):
    """GET /metrics/retrieval should include reranker config."""
    response = client.get("/metrics/retrieval")
    data = response.json()

    assert "reranker" in data
    assert data["reranker"]["enabled"] is True
    assert data["reranker"]["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_retrieval_endpoint_returns_top_k(mock_env_vars):
    """GET /metrics/retrieval should include top-k settings."""
    response = client.get("/metrics/retrieval")
    data = response.json()

    assert data["retrieval_top_k"] == 10
    assert "final_top_n" in data


def test_retrieval_endpoint_includes_research_references(mock_env_vars):
    """GET /metrics/retrieval should include research references."""
    response = client.get("/metrics/retrieval")
    data = response.json()

    assert "research_reference" in data["hybrid_search"]
    assert "improvement_claim" in data["hybrid_search"]


# ============================================================================
# Evaluation Definitions Endpoint Tests
# ============================================================================

def test_eval_definitions_endpoint_returns_200():
    """GET /metrics/eval/definitions should return 200."""
    response = client.get("/metrics/eval/definitions")
    assert response.status_code == 200


def test_eval_definitions_returns_list():
    """GET /metrics/eval/definitions should return a list of metrics."""
    response = client.get("/metrics/eval/definitions")
    data = response.json()

    assert isinstance(data, list)
    assert len(data) >= 5  # We have 5 core metrics


def test_eval_definitions_includes_required_fields():
    """Each metric definition should have required fields."""
    response = client.get("/metrics/eval/definitions")
    data = response.json()

    for metric in data:
        assert "name" in metric
        assert "category" in metric
        assert "description" in metric
        assert "threshold" in metric
        assert "interpretation" in metric


def test_eval_definitions_includes_all_categories():
    """Metric definitions should cover all categories."""
    response = client.get("/metrics/eval/definitions")
    data = response.json()

    categories = {m["category"] for m in data}
    assert "retrieval" in categories
    assert "generation" in categories
    assert "safety" in categories


def test_eval_definitions_includes_expected_metrics():
    """Metric definitions should include core RAG metrics."""
    response = client.get("/metrics/eval/definitions")
    data = response.json()

    metric_names = {m["name"] for m in data}
    assert "precision_at_k" in metric_names
    assert "recall_at_k" in metric_names
    assert "citation_precision" in metric_names
    assert "faithfulness" in metric_names
    assert "answer_relevancy" in metric_names
    assert "hallucination" in metric_names


# ============================================================================
# Evaluation Runs Endpoint Tests
# ============================================================================

@pytest.fixture
def mock_eval_progress():
    """Mock Redis-based eval progress functions."""
    mock_runs = [
        {
            "run_id": "test-run-1",
            "name": "Test Run 1",
            "status": "completed",
            "created_at": "2025-01-01T00:00:00",
            "completed_at": "2025-01-01T00:05:00",
            "groups": ["retrieval"],
            "datasets": ["golden"],
            "total_questions": 10,
            "results": {"weighted_score": 0.85},
        }
    ]

    with patch('api.routes.eval.list_eval_runs', return_value=mock_runs):
        yield mock_runs


def test_eval_runs_endpoint_returns_200(mock_eval_progress):
    """GET /metrics/eval/runs should return 200."""
    response = client.get("/metrics/eval/runs")
    assert response.status_code == 200


def test_eval_runs_returns_structure(mock_eval_progress):
    """GET /metrics/eval/runs should return expected structure."""
    response = client.get("/metrics/eval/runs")
    data = response.json()

    assert "runs" in data
    assert "total" in data
    assert "limit" in data
    assert "offset" in data
    assert isinstance(data["runs"], list)


def test_eval_runs_supports_limit_param(mock_eval_progress):
    """GET /metrics/eval/runs should support limit parameter."""
    response = client.get("/metrics/eval/runs?limit=5")
    assert response.status_code == 200


# ============================================================================
# Evaluation Summary Endpoint Tests
# ============================================================================

def test_eval_summary_endpoint_returns_200():
    """GET /metrics/eval/summary should return 200."""
    response = client.get("/metrics/eval/summary")
    assert response.status_code == 200


def test_eval_summary_returns_structure():
    """GET /metrics/eval/summary should return expected structure."""
    response = client.get("/metrics/eval/summary")
    data = response.json()

    assert "total_runs" in data
    assert "metric_trends" in data
    assert isinstance(data["metric_trends"], list)


# ============================================================================
# System Metrics Endpoint Tests
# ============================================================================

def test_system_metrics_endpoint_returns_200(mock_system_metrics):
    """GET /metrics/system should return 200."""
    response = client.get("/metrics/system")
    assert response.status_code == 200


def test_system_metrics_returns_models(mock_system_metrics):
    """GET /metrics/system should include models configuration."""
    response = client.get("/metrics/system")
    data = response.json()

    assert "models" in data
    assert "llm" in data["models"]
    assert "embedding" in data["models"]


def test_system_metrics_returns_retrieval(mock_system_metrics):
    """GET /metrics/system should include retrieval configuration."""
    response = client.get("/metrics/system")
    data = response.json()

    assert "retrieval" in data
    assert "hybrid_search" in data["retrieval"]


def test_system_metrics_returns_document_count(mock_system_metrics):
    """GET /metrics/system should include document count."""
    response = client.get("/metrics/system")
    data = response.json()

    assert "document_count" in data
    assert isinstance(data["document_count"], int)


def test_system_metrics_returns_document_stats(mock_system_metrics):
    """GET /metrics/system should include document statistics."""
    response = client.get("/metrics/system")
    data = response.json()

    assert "document_count" in data
    assert "chunk_count" in data


def test_system_metrics_returns_health_status(mock_system_metrics):
    """GET /metrics/system should include health status."""
    response = client.get("/metrics/system")
    data = response.json()

    assert "health_status" in data
    assert "component_status" in data


def test_system_metrics_returns_timestamp(mock_system_metrics):
    """GET /metrics/system should include a timestamp."""
    response = client.get("/metrics/system")
    data = response.json()

    assert "timestamp" in data
    assert "system_name" in data


# ============================================================================
# Edge Cases
# ============================================================================

def test_models_with_reranker_disabled(mock_ollama):
    """GET /metrics/models should handle disabled reranker."""
    disabled_reranker_config = create_mock_models_config()
    disabled_reranker_config.reranker.enabled = False

    with patch(
        "infrastructure.config.models_config.get_models_config",
        return_value=disabled_reranker_config,
    ):
        with patch(
            "infrastructure.config.models_config._default_manager._config",
            disabled_reranker_config,
        ):
            response = client.get("/metrics/models")
            data = response.json()

            # Reranker should be None when disabled
            assert data["reranker"] is None


def test_retrieval_with_hybrid_disabled():
    """GET /metrics/retrieval should handle disabled hybrid search."""
    disabled_hybrid_config = create_mock_models_config()
    disabled_hybrid_config.retrieval.enable_hybrid_search = False

    with patch(
        "infrastructure.config.models_config.get_models_config",
        return_value=disabled_hybrid_config,
    ):
        with patch(
            "infrastructure.config.models_config._default_manager._config",
            disabled_hybrid_config,
        ):
            response = client.get("/metrics/retrieval")
            data = response.json()

            assert data["hybrid_search"]["enabled"] is False
