import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from schemas.metrics import (
    SystemMetrics,
    ModelsConfig,
    RetrievalConfig,
    MetricDefinition,
    EvaluationHistory,
    EvaluationSummary,
    GoldenBaseline,
    ComparisonResult,
    Recommendation,
    EvaluationRun,
)
from services.metrics import (
    get_system_metrics as fetch_system_metrics,
    get_models_config,
    get_retrieval_config,
    get_metric_definitions,
    load_evaluation_history,
    get_evaluation_summary as fetch_eval_summary,
    get_evaluation_run_by_id,
    delete_evaluation_run,
)
from services.baseline import get_baseline_service
from services.comparison import compare_runs as compare_eval_runs
from services.recommendation import get_recommendation

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/metrics/system", response_model=SystemMetrics)
async def get_system_metrics():
    """Get complete system metrics and configuration overview.

    Returns comprehensive information about:
    - All models (LLM, embedding, reranker, eval) with sizes and references
    - Retrieval pipeline configuration (hybrid search, BM25, reranking)
    - Evaluation metrics definitions
    - Latest evaluation results
    - Document statistics
    - Component health status
    """
    try:
        return await fetch_system_metrics()
    except Exception as e:
        logger.error(f"[METRICS] Error fetching system metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/models", response_model=ModelsConfig)
async def get_detailed_models_info():
    """Get detailed information about all models used in the RAG system.

    Returns for each model (LLM, embedding, reranker, eval):
    - Model name and provider
    - Size information (parameters, disk size, context window)
    - Reference URL to model documentation
    - Current status (loaded, available, unavailable)
    """
    try:
        return await get_models_config()
    except Exception as e:
        logger.error(f"[METRICS] Error fetching models config: {str(e)}")
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
        logger.error(f"[METRICS] Error fetching retrieval config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/evaluation/definitions", response_model=list[MetricDefinition])
async def get_evaluation_metric_definitions():
    """Get definitions for all evaluation metrics.

    Returns for each metric:
    - Name and category (retrieval, generation, safety)
    - Description of what it measures
    - Pass/fail threshold
    - Interpretation guide
    - Reference documentation URL
    """
    try:
        return get_metric_definitions()
    except Exception as e:
        logger.error(f"[METRICS] Error fetching metric definitions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/evaluation/history", response_model=EvaluationHistory)
async def get_evaluation_history(limit: int = 20):
    """Get historical evaluation runs.

    Args:
        limit: Maximum number of runs to return (default: 20)

    Returns:
        List of evaluation runs with detailed results and metrics
    """
    try:
        return load_evaluation_history(limit=limit)
    except Exception as e:
        logger.error(f"[METRICS] Error fetching evaluation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/evaluation/summary", response_model=EvaluationSummary)
async def get_evaluation_summary():
    """Get evaluation summary with trends.

    Returns:
    - Latest evaluation run
    - Total number of runs
    - Metric trends over time (improving/declining/stable)
    - Best performing run
    """
    try:
        return fetch_eval_summary()
    except Exception as e:
        logger.error(f"[METRICS] Error fetching evaluation summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# NOTE: Parameterized routes must come AFTER specific routes like /summary
@router.get("/metrics/evaluation/{run_id}", response_model=EvaluationRun)
async def get_evaluation_run(run_id: str):
    """Get a specific evaluation run by ID.

    Args:
        run_id: The evaluation run ID

    Returns:
        The evaluation run details
    """
    try:
        run = get_evaluation_run_by_id(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Evaluation run '{run_id}' not found")
        return run
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[METRICS] Error fetching evaluation run: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/metrics/evaluation/{run_id}", status_code=204)
async def delete_eval_run(run_id: str):
    """Delete a specific evaluation run by ID.

    Args:
        run_id: The evaluation run ID to delete
    """
    try:
        deleted = delete_evaluation_run(run_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Evaluation run '{run_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[METRICS] Error deleting evaluation run: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Baseline Management Endpoints
# ============================================================================


@router.get("/metrics/baseline", response_model=Optional[GoldenBaseline])
async def get_golden_baseline():
    """Get the current golden baseline.

    Returns:
        The golden baseline if set, null otherwise
    """
    try:
        baseline_service = get_baseline_service()
        return baseline_service.get_baseline()
    except Exception as e:
        logger.error(f"[METRICS] Error fetching baseline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/baseline/{run_id}", response_model=GoldenBaseline)
async def set_golden_baseline(run_id: str, set_by: Optional[str] = None):
    """Set a specific evaluation run as the golden baseline.

    The baseline's metric scores become the thresholds to beat.
    New evaluation runs will be compared against this baseline.

    Args:
        run_id: ID of the evaluation run to set as baseline
        set_by: Optional identifier for who set the baseline
    """
    try:
        # Get the evaluation run
        run = get_evaluation_run_by_id(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Evaluation run '{run_id}' not found")

        baseline_service = get_baseline_service()
        return baseline_service.set_baseline(run, set_by=set_by)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[METRICS] Error setting baseline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/metrics/baseline", status_code=204)
async def clear_golden_baseline():
    """Clear the current golden baseline."""
    try:
        baseline_service = get_baseline_service()
        baseline_service.clear_baseline()
    except Exception as e:
        logger.error(f"[METRICS] Error clearing baseline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Comparison Endpoints
# ============================================================================


@router.get("/metrics/compare/{run_a_id}/{run_b_id}", response_model=ComparisonResult)
async def compare_runs(run_a_id: str, run_b_id: str):
    """Compare two evaluation runs side-by-side.

    Returns metric deltas, latency/cost comparison, and winner determination.

    Args:
        run_a_id: ID of first evaluation run
        run_b_id: ID of second evaluation run
    """
    try:
        # Get both runs
        run_a = get_evaluation_run_by_id(run_a_id)
        run_b = get_evaluation_run_by_id(run_b_id)

        if run_a is None:
            raise HTTPException(status_code=404, detail=f"Evaluation run '{run_a_id}' not found")
        if run_b is None:
            raise HTTPException(status_code=404, detail=f"Evaluation run '{run_b_id}' not found")

        return compare_eval_runs(run_a, run_b)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[METRICS] Error comparing runs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/compare-to-baseline/{run_id}", response_model=ComparisonResult)
async def compare_to_baseline(run_id: str):
    """Compare a run against the golden baseline.

    Args:
        run_id: ID of the evaluation run to compare
    """
    try:
        # Get the baseline
        baseline_service = get_baseline_service()
        baseline = baseline_service.get_baseline()

        if baseline is None:
            raise HTTPException(status_code=404, detail="No golden baseline set")

        # Get the runs
        run = get_evaluation_run_by_id(run_id)
        baseline_run = get_evaluation_run_by_id(baseline.run_id)

        if run is None:
            raise HTTPException(status_code=404, detail=f"Evaluation run '{run_id}' not found")
        if baseline_run is None:
            raise HTTPException(status_code=404, detail=f"Baseline run '{baseline.run_id}' not found")

        return compare_eval_runs(run, baseline_run)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[METRICS] Error comparing to baseline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Recommendation Endpoint
# ============================================================================


@router.post("/metrics/recommend", response_model=Recommendation)
async def get_recommendation(
    accuracy_weight: float = Query(0.5, ge=0, le=1, description="Weight for accuracy (0-1)"),
    speed_weight: float = Query(0.3, ge=0, le=1, description="Weight for speed (0-1)"),
    cost_weight: float = Query(0.2, ge=0, le=1, description="Weight for cost efficiency (0-1)"),
    limit_to_runs: int = Query(10, ge=1, le=100, description="Max runs to consider"),
):
    """Get recommended configuration based on weighted preferences.

    Analyzes historical evaluation runs and recommends the optimal
    configuration based on your priorities for accuracy, speed, and cost.

    Args:
        accuracy_weight: How much to prioritize accuracy (default 0.5)
        speed_weight: How much to prioritize speed (default 0.3)
        cost_weight: How much to prioritize cost efficiency (default 0.2)
        limit_to_runs: Maximum number of historical runs to analyze
    """
    try:
        result = get_recommendation(
            accuracy_weight=accuracy_weight,
            speed_weight=speed_weight,
            cost_weight=cost_weight,
            limit_to_runs=limit_to_runs,
        )

        if result is None:
            raise HTTPException(
                status_code=400,
                detail="Insufficient evaluation data. Run at least one evaluation with config snapshots.",
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[METRICS] Error getting recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
