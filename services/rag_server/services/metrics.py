"""Service for gathering RAG system metrics and configuration.

Provides methods to:
- Query model information from Ollama/HuggingFace
- Gather retrieval configuration
- Get system overview metrics

Note: Evaluation-related functions have been moved to services/eval/.
"""

import logging

import httpx

from schemas.metrics import (
    ModelInfo,
    ModelSize,
    ModelsConfig,
    VectorSearchConfig,
    BM25Config,
    HybridSearchConfig,
    ContextualRetrievalConfig,
    RerankerConfig,
    RetrievalConfig,
    SystemMetrics,
)
from core.config import get_optional_env
from app.settings import has_anthropic_key
from pipelines.inference import get_inference_config
from pipelines.ingestion import get_ingestion_config

logger = logging.getLogger(__name__)

# Model reference URLs
MODEL_REFERENCES = {
    # Ollama models
    "gemma3:4b": {
        "url": "https://ollama.com/library/gemma3",
        "description": "Google Gemma 3 4B - Lightweight, efficient LLM for text generation",
        "parameters": "4B",
    },
    "gemma2:9b": {
        "url": "https://ollama.com/library/gemma2",
        "description": "Google Gemma 2 9B - Mid-size LLM with strong reasoning",
        "parameters": "9B",
    },
    "llama3.2:3b": {
        "url": "https://ollama.com/library/llama3.2",
        "description": "Meta Llama 3.2 3B - Efficient instruction-following model",
        "parameters": "3B",
    },
    "llama3.1:8b": {
        "url": "https://ollama.com/library/llama3.1",
        "description": "Meta Llama 3.1 8B - Powerful open-source LLM",
        "parameters": "8B",
    },
    "mistral:7b": {
        "url": "https://ollama.com/library/mistral",
        "description": "Mistral 7B - Fast, efficient open-source model",
        "parameters": "7B",
    },
    "nomic-embed-text:latest": {
        "url": "https://ollama.com/library/nomic-embed-text",
        "description": "Nomic Embed Text - High-quality text embeddings (768 dims)",
        "parameters": "137M",
        "context_window": 8192,
    },
    "nomic-embed-text": {
        "url": "https://ollama.com/library/nomic-embed-text",
        "description": "Nomic Embed Text - High-quality text embeddings (768 dims)",
        "parameters": "137M",
        "context_window": 8192,
    },
    "mxbai-embed-large": {
        "url": "https://ollama.com/library/mxbai-embed-large",
        "description": "MixedBread Embed Large - State-of-the-art embeddings (1024 dims)",
        "parameters": "335M",
        "context_window": 512,
    },
    # HuggingFace reranker models
    "cross-encoder/ms-marco-MiniLM-L-6-v2": {
        "url": "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2",
        "description": "MS MARCO MiniLM L-6 - Fast cross-encoder for passage reranking",
        "parameters": "22M",
        "disk_size_mb": 80,
    },
    "BAAI/bge-reranker-base": {
        "url": "https://huggingface.co/BAAI/bge-reranker-base",
        "description": "BGE Reranker Base - High-quality Chinese/English reranker",
        "parameters": "278M",
        "disk_size_mb": 1100,
    },
    # Anthropic eval models
    "claude-sonnet-4-20250514": {
        "url": "https://docs.anthropic.com/en/docs/about-claude/models",
        "description": "Claude Sonnet 4 - Fast, intelligent model for evaluation tasks",
        "parameters": "Unknown",
        "context_window": 200000,
    },
    "claude-3-5-sonnet-20241022": {
        "url": "https://docs.anthropic.com/en/docs/about-claude/models",
        "description": "Claude 3.5 Sonnet - Balanced performance and cost for evals",
        "parameters": "Unknown",
        "context_window": 200000,
    },
}


async def get_ollama_model_info(model_name: str) -> dict | None:
    """Query Ollama API for model details."""
    ollama_url = get_optional_env("OLLAMA_URL", "http://host.docker.internal:11434")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{ollama_url}/api/show", json={"name": model_name}
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.warning(f"Failed to get Ollama model info for {model_name}: {e}")

    return None


async def check_ollama_model_loaded(model_name: str) -> bool:
    """Check if a model is currently loaded in Ollama."""
    ollama_url = get_optional_env("OLLAMA_URL", "http://host.docker.internal:11434")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ollama_url}/api/ps")
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                return any(
                    m.get("name", "").startswith(model_name.split(":")[0])
                    for m in models
                )
    except Exception as e:
        logger.warning(f"Failed to check Ollama model status: {e}")

    return False


def get_model_reference(model_name: str) -> dict:
    """Get reference information for a model."""
    if model_name in MODEL_REFERENCES:
        return MODEL_REFERENCES[model_name]

    base_name = model_name.split(":")[0]
    for key, value in MODEL_REFERENCES.items():
        if key.startswith(base_name):
            return value

    return {
        "url": None,
        "description": f"Model: {model_name}",
        "parameters": "Unknown",
    }


async def get_models_config() -> ModelsConfig:
    """Get complete models configuration with details."""
    from infrastructure.config.models_config import get_models_config as get_config

    config = get_config()
    llm_model = config.llm.model
    embedding_model = config.embedding.model
    eval_model = config.eval.model

    inference_config = get_inference_config()

    # Get LLM info
    llm_ref = get_model_reference(llm_model)
    llm_ollama_info = await get_ollama_model_info(llm_model)
    llm_loaded = await check_ollama_model_loaded(llm_model)

    llm_size = ModelSize(
        parameters=llm_ref.get("parameters"),
        disk_size_mb=(
            llm_ollama_info.get("size", 0) / 1024 / 1024 if llm_ollama_info else None
        ),
        context_window=llm_ref.get("context_window"),
    )

    llm_info = ModelInfo(
        name=llm_model,
        provider="Ollama",
        model_type="llm",
        is_local=True,
        size=llm_size,
        reference_url=llm_ref.get("url"),
        description=llm_ref.get("description"),
        status="loaded" if llm_loaded else "available",
    )

    # Get embedding info
    embed_ref = get_model_reference(embedding_model)
    embed_ollama_info = await get_ollama_model_info(embedding_model)

    embed_size = ModelSize(
        parameters=embed_ref.get("parameters"),
        disk_size_mb=(
            embed_ollama_info.get("size", 0) / 1024 / 1024 if embed_ollama_info else None
        ),
        context_window=embed_ref.get("context_window"),
    )

    embedding_info = ModelInfo(
        name=embedding_model,
        provider="Ollama",
        model_type="embedding",
        is_local=True,
        size=embed_size,
        reference_url=embed_ref.get("url"),
        description=embed_ref.get("description"),
        status="available",
    )

    # Get reranker info (if enabled)
    reranker_info = None
    if inference_config["reranker_enabled"]:
        reranker_model = inference_config["reranker_model"]
        reranker_ref = get_model_reference(reranker_model)

        reranker_size = ModelSize(
            parameters=reranker_ref.get("parameters"),
            disk_size_mb=reranker_ref.get("disk_size_mb"),
        )

        reranker_info = ModelInfo(
            name=reranker_model,
            provider="HuggingFace",
            model_type="reranker",
            is_local=True,
            size=reranker_size,
            reference_url=reranker_ref.get("url"),
            description=reranker_ref.get("description"),
            status="available",
        )

    # Get eval model info
    eval_ref = get_model_reference(eval_model)
    eval_size = ModelSize(
        parameters=eval_ref.get("parameters"),
        context_window=eval_ref.get("context_window"),
    )

    eval_info = ModelInfo(
        name=eval_model,
        provider="Anthropic",
        model_type="eval",
        is_local=False,
        size=eval_size,
        reference_url=eval_ref.get("url"),
        description=eval_ref.get("description"),
        status="available" if has_anthropic_key() else "unavailable",
    )

    return ModelsConfig(
        llm=llm_info,
        embedding=embedding_info,
        reranker=reranker_info,
        eval=eval_info,
    )


def get_retrieval_config() -> RetrievalConfig:
    """Get complete retrieval pipeline configuration."""
    inference_config = get_inference_config()
    ingestion_config = get_ingestion_config()

    vector_config = VectorSearchConfig(
        enabled=True,
        chunk_size=500,
        chunk_overlap=50,
        vector_store="PostgreSQL (pgvector)",
        collection_name="documents",
    )

    bm25_config = BM25Config(
        enabled=inference_config["hybrid_search_enabled"],
    )

    hybrid_search_config = HybridSearchConfig(
        enabled=inference_config["hybrid_search_enabled"],
        bm25=bm25_config,
        vector=vector_config,
        fusion_method="reciprocal_rank_fusion",
        rrf_k=inference_config["rrf_k"],
    )

    contextual_retrieval_config = ContextualRetrievalConfig(
        enabled=ingestion_config["contextual_retrieval_enabled"],
    )

    top_k = inference_config["retrieval_top_k"]
    top_n = max(5, top_k // 2) if inference_config["reranker_enabled"] else top_k

    reranker_cfg = RerankerConfig(
        enabled=inference_config["reranker_enabled"],
        model=(
            inference_config["reranker_model"]
            if inference_config["reranker_enabled"]
            else None
        ),
        top_n=top_n if inference_config["reranker_enabled"] else None,
    )

    return RetrievalConfig(
        retrieval_top_k=top_k,
        final_top_n=top_n,
        hybrid_search=hybrid_search_config,
        contextual_retrieval=contextual_retrieval_config,
        reranker=reranker_cfg,
    )


async def get_system_metrics() -> SystemMetrics:
    """Get complete system metrics overview."""
    from infrastructure.database.postgres import get_session
    from infrastructure.database import documents as db_docs

    models = await get_models_config()
    retrieval = get_retrieval_config()

    try:
        async with get_session() as session:
            documents = await db_docs.list_documents(session)
        doc_count = len(documents)
        chunk_count = sum(d.get("chunks", 0) for d in documents)
    except Exception as e:
        logger.warning(f"Failed to get document stats: {e}")
        doc_count = 0
        chunk_count = 0

    component_status = {}

    # Check PostgreSQL
    try:
        from sqlalchemy import text
        from infrastructure.database.postgres import get_engine
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        component_status["postgres"] = "healthy"
    except Exception as e:
        logger.warning(f"PostgreSQL health check error: {e}")
        component_status["postgres"] = "unavailable"

    # Check Ollama
    ollama_url = get_optional_env("OLLAMA_URL", "http://host.docker.internal:11434")
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            if resp.status_code == 200:
                component_status["ollama"] = "healthy"
            else:
                logger.warning(f"Ollama health check failed: status={resp.status_code}")
                component_status["ollama"] = "unhealthy"
    except Exception as e:
        logger.warning(f"Ollama health check error: {e}")
        component_status["ollama"] = "unavailable"

    health_status = (
        "healthy"
        if all(s == "healthy" for s in component_status.values())
        else "degraded"
    )

    return SystemMetrics(
        models=models,
        retrieval=retrieval,
        document_count=doc_count,
        chunk_count=chunk_count,
        health_status=health_status,
        component_status=component_status,
    )
