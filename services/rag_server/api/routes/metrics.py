"""System metrics and configuration API routes (/metrics/*).

Provides endpoints for:
- System overview and health
- Model configuration (LLM, embedding, reranker)
- Retrieval pipeline configuration

Note: Evaluation-specific endpoints have been moved to /metrics/eval/*.
See api/routes/eval.py for evaluation runs, baseline, comparison, and recommendations.
"""

import logging

from fastapi import APIRouter, HTTPException

from schemas.metrics import (
    SystemMetrics,
    ModelsConfig,
    RetrievalConfig,
)
from services.metrics import (
    get_system_metrics as fetch_system_metrics,
    get_models_config,
    get_retrieval_config,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/metrics/system", response_model=SystemMetrics)
async def get_system_metrics():
    """Get complete system metrics and configuration overview.

    Returns comprehensive information about:
    - All models (LLM, embedding, reranker) with sizes and references
    - Retrieval pipeline configuration (hybrid search, BM25, reranking)
    - Document statistics
    - Component health status
    """
    try:
        return await fetch_system_metrics()
    except Exception as e:
        logger.error(f"[METRICS] Error fetching system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/models", response_model=ModelsConfig)
async def get_detailed_models_info():
    """Get detailed information about all models used in the RAG system.

    Returns for each model (LLM, embedding, reranker):
    - Model name and provider
    - Size information (parameters, disk size, context window)
    - Reference URL to model documentation
    - Current status (loaded, available, unavailable)
    """
    try:
        return await get_models_config()
    except Exception as e:
        logger.error(f"[METRICS] Error fetching models config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/retrieval", response_model=RetrievalConfig)
async def get_retrieval_configuration():
    """Get retrieval pipeline configuration.

    Returns configuration for:
    - Hybrid search (BM25 + Vector + RRF fusion)
    - Contextual retrieval (Anthropic method)
    - Reranking settings
    - Top-K and Top-N parameters
    - Research references and improvement claims
    """
    try:
        return get_retrieval_config()
    except Exception as e:
        logger.error(f"[METRICS] Error fetching retrieval config: {e}")
        raise HTTPException(status_code=500, detail=str(e))
