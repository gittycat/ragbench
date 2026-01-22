"""Evaluation API routes (/metrics/eval/).

Provides endpoints for:
- Discovery: metric groups and datasets
- Execution: start, monitor, and cancel evaluation runs
- Results: retrieve completed run results
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from schemas.eval import (
    MetricGroupResponse,
    MetricGroupsResponse,
    MetricInfo,
    DatasetResponse,
    DatasetsResponse,
    EvalRunRequest,
    EvalRunResponse,
    EvalRunListResponse,
    EvalRunListItem,
    EvalRunProgress,
    EvalRunConfig,
    EvalRunResults,
    GroupResults,
    MetricValue,
    PerformanceResults,
    EvaluationSummary,
    GoldenBaseline,
    ComparisonResult,
    Recommendation,
    EvaluationRun,
    MetricDefinition,
)
from infrastructure.tasks.eval_progress import (
    create_eval_run,
    get_eval_progress,
    delete_eval_progress,
    list_eval_runs,
    cancel_eval_run,
)
from services.eval import (
    # History
    load_evaluation_history,
    get_evaluation_summary,
    get_evaluation_run_by_id,
    get_metric_definitions,
    # Baseline
    get_baseline,
    set_baseline,
    clear_baseline,
    # Comparison
    compare_runs,
    # Recommendation
    get_recommendation,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Static Metric Group Definitions
# ============================================================================

METRIC_GROUPS = [
    MetricGroupResponse(
        id="retrieval",
        name="Retrieval Quality",
        description="Measures how well the system retrieves relevant documents",
        metrics=[
            MetricInfo(
                id="recall_at_k",
                name="Recall@K",
                description="Fraction of relevant documents in top K results",
                parameters={"k": [1, 3, 5, 10]},
                requires_judge=False,
            ),
            MetricInfo(
                id="precision_at_k",
                name="Precision@K",
                description="Fraction of top K results that are relevant",
                parameters={"k": [1, 3, 5]},
                requires_judge=False,
            ),
            MetricInfo(
                id="mrr",
                name="Mean Reciprocal Rank",
                description="Rank of first relevant result",
                requires_judge=False,
            ),
            MetricInfo(
                id="ndcg",
                name="NDCG",
                description="Normalized Discounted Cumulative Gain",
                parameters={"k": 10},
                requires_judge=False,
            ),
        ],
        estimated_duration_per_sample_ms=100,
        requires_judge=False,
        recommended_datasets=["ragbench", "msmarco", "hotpotqa"],
    ),
    MetricGroupResponse(
        id="generation",
        name="Answer Quality",
        description="Evaluates generated answer quality using LLM-as-judge",
        metrics=[
            MetricInfo(
                id="faithfulness",
                name="Faithfulness",
                description="Is the answer grounded in the retrieved context?",
                requires_judge=True,
            ),
            MetricInfo(
                id="answer_correctness",
                name="Answer Correctness",
                description="Semantic match with expected answer",
                requires_judge=True,
            ),
            MetricInfo(
                id="answer_relevancy",
                name="Answer Relevancy",
                description="Does the answer address the question?",
                requires_judge=True,
            ),
        ],
        estimated_duration_per_sample_ms=3000,
        requires_judge=True,
        recommended_datasets=["ragbench", "qasper", "hotpotqa"],
    ),
    MetricGroupResponse(
        id="citation",
        name="Citation Quality",
        description="Measures source attribution accuracy",
        metrics=[
            MetricInfo(
                id="citation_precision",
                name="Citation Precision",
                description="Fraction of citations that are relevant",
                requires_judge=False,
            ),
            MetricInfo(
                id="citation_recall",
                name="Citation Recall",
                description="Fraction of relevant passages that are cited",
                requires_judge=False,
            ),
            MetricInfo(
                id="section_accuracy",
                name="Section Accuracy",
                description="Accuracy at document+section level",
                requires_judge=False,
            ),
        ],
        estimated_duration_per_sample_ms=100,
        requires_judge=False,
        recommended_datasets=["qasper"],
    ),
    MetricGroupResponse(
        id="abstention",
        name="Abstention Quality",
        description="Handles unanswerable question detection",
        metrics=[
            MetricInfo(
                id="unanswerable_accuracy",
                name="Unanswerable Accuracy",
                description="Correct abstention on unanswerable questions",
                requires_judge=False,
            ),
            MetricInfo(
                id="false_positive_rate",
                name="False Positive Rate",
                description="Incorrectly abstaining on answerable questions",
                requires_judge=False,
            ),
            MetricInfo(
                id="false_negative_rate",
                name="False Negative Rate",
                description="Incorrectly answering unanswerable questions",
                requires_judge=False,
            ),
        ],
        estimated_duration_per_sample_ms=100,
        requires_judge=False,
        recommended_datasets=["squad_v2"],
    ),
    MetricGroupResponse(
        id="performance",
        name="Performance Metrics",
        description="Latency and cost tracking",
        metrics=[
            MetricInfo(
                id="latency_p50",
                name="Latency P50",
                description="Median query latency",
                requires_judge=False,
            ),
            MetricInfo(
                id="latency_p95",
                name="Latency P95",
                description="95th percentile latency",
                requires_judge=False,
            ),
            MetricInfo(
                id="cost_per_query",
                name="Cost Per Query",
                description="Cost in USD per query",
                requires_judge=False,
            ),
        ],
        estimated_duration_per_sample_ms=0,
        requires_judge=False,
        recommended_datasets=["ragbench"],
    ),
]

VALID_GROUPS = {g.id for g in METRIC_GROUPS}


# ============================================================================
# Discovery Endpoints
# ============================================================================


@router.get("/metrics/eval/groups", response_model=MetricGroupsResponse)
async def get_metric_groups():
    """Get available evaluation metric groups with their metrics."""
    return MetricGroupsResponse(groups=METRIC_GROUPS)


@router.get("/metrics/eval/datasets", response_model=DatasetsResponse)
async def get_datasets():
    """Get available evaluation datasets."""
    # Lazy import to avoid loading heavy dependencies at module load time
    from evaluation_cc.datasets.registry import list_available, get_metadata

    datasets = []

    for name in list_available():
        try:
            metadata = get_metadata(name)
            datasets.append(DatasetResponse(
                id=metadata.get("id", name.value),
                name=metadata.get("name", name.value.replace("_", " ").title()),
                description=metadata.get("description", ""),
                size=metadata.get("size", 0),
                domains=metadata.get("domains", []),
                primary_aspects=metadata.get("primary_aspects", []),
                requires_download=metadata.get("requires_download", True),
                download_size_mb=metadata.get("download_size_mb", 0),
            ))
        except Exception as e:
            logger.warning(f"[EVAL] Failed to get metadata for {name}: {e}")

    return DatasetsResponse(datasets=datasets)


# ============================================================================
# Run Management Endpoints
# ============================================================================


@router.post("/metrics/eval/runs", response_model=EvalRunResponse)
async def start_evaluation_run(request: EvalRunRequest):
    """Start a new evaluation run."""
    # Lazy imports to avoid loading heavy dependencies at module load time
    from evaluation_cc.datasets.registry import list_available
    from infrastructure.tasks.eval_worker import run_evaluation_task

    try:
        # Validate groups
        invalid_groups = set(request.groups) - VALID_GROUPS
        if invalid_groups:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid groups: {invalid_groups}. Valid groups: {VALID_GROUPS}",
            )

        # Validate datasets
        valid_datasets = {ds.value for ds in list_available()}
        invalid_datasets = set(request.datasets) - valid_datasets
        if invalid_datasets:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid datasets: {invalid_datasets}. Valid datasets: {valid_datasets}",
            )

        # Validate judge requirement
        needs_judge = "generation" in request.groups
        if needs_judge and (not request.judge or not request.judge.enabled):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "validation_error",
                    "message": "Generation metrics require LLM judge to be enabled",
                    "details": {
                        "field": "judge.enabled",
                        "required_by": ["faithfulness", "answer_correctness", "answer_relevancy"],
                    },
                },
            )

        # Generate run ID
        run_id = str(uuid.uuid4())[:8]
        run_name = request.name or f"eval-{run_id}"

        # Estimate total questions
        total_questions = len(request.datasets) * request.samples_per_dataset

        # Create config for response
        config = EvalRunConfig(
            groups=request.groups,
            datasets=request.datasets,
            samples_per_dataset=request.samples_per_dataset,
            total_samples=total_questions,
            judge_enabled=request.judge.enabled if request.judge else False,
        )

        # Create progress entry in Redis
        create_eval_run(
            run_id=run_id,
            name=run_name,
            groups=request.groups,
            datasets=request.datasets,
            total_questions=total_questions,
            config=config.model_dump(),
        )

        # Queue Celery task
        run_evaluation_task.apply_async(
            args=[
                run_id,
                run_name,
                request.groups,
                request.datasets,
                request.samples_per_dataset,
                request.judge.model_dump() if request.judge else None,
                request.metrics,
                request.seed,
            ],
            task_id=f"eval-{run_id}",
        )

        logger.info(f"[EVAL] Started evaluation run {run_id}: {run_name}")

        return EvalRunResponse(
            run_id=run_id,
            name=run_name,
            status="pending",
            created_at=datetime.now(),
            config=config,
            question_count=total_questions,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[EVAL] Error starting evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/eval/runs/{run_id}", response_model=EvalRunResponse)
async def get_evaluation_run(run_id: str):
    """Get evaluation run details and current status."""
    # Check Redis for run progress
    progress_data = get_eval_progress(run_id)

    if not progress_data:
        raise HTTPException(status_code=404, detail=f"Evaluation run '{run_id}' not found")

    # Build progress object
    progress = EvalRunProgress(
        phase=progress_data.get("phase", "unknown"),
        total_questions=progress_data.get("total_questions", 0),
        completed_questions=progress_data.get("completed_questions", 0),
        current_dataset=progress_data.get("current_dataset"),
        percent_complete=_calculate_percent(
            progress_data.get("completed_questions", 0),
            progress_data.get("total_questions", 1),
        ),
        metrics_computed=progress_data.get("metrics_computed", []),
        metrics_pending=progress_data.get("metrics_pending", []),
    )

    # Build config from stored data
    config_data = progress_data.get("config", {})
    config = EvalRunConfig(
        groups=config_data.get("groups", progress_data.get("groups", [])),
        datasets=config_data.get("datasets", progress_data.get("datasets", [])),
        samples_per_dataset=config_data.get("samples_per_dataset", 0),
        total_samples=config_data.get("total_samples", progress_data.get("total_questions", 0)),
        judge_enabled=config_data.get("judge_enabled", False),
    )

    # Build results if available
    results = None
    results_data = progress_data.get("results")
    if results_data:
        results = _parse_results(results_data)

    # Calculate duration if completed
    duration = None
    completed_at = None
    if progress_data.get("completed_at"):
        completed_at = datetime.fromisoformat(progress_data["completed_at"])
        created_at = datetime.fromisoformat(progress_data["created_at"])
        duration = (completed_at - created_at).total_seconds()

    return EvalRunResponse(
        run_id=run_id,
        name=progress_data.get("name", f"eval-{run_id}"),
        status=progress_data.get("status", "unknown"),
        created_at=datetime.fromisoformat(progress_data["created_at"]),
        completed_at=completed_at,
        duration_seconds=duration,
        progress=progress,
        config=config,
        results=results,
        question_count=progress_data.get("total_questions", 0),
        error_count=len(progress_data.get("errors", [])),
    )


@router.get("/metrics/eval/runs", response_model=EvalRunListResponse)
async def list_evaluation_runs(
    limit: int = Query(20, ge=1, le=100, description="Maximum runs to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    status: str | None = Query(None, description="Filter by status"),
):
    """List evaluation runs with pagination."""
    # Get runs from Redis
    all_runs = list_eval_runs(status=status, limit=limit + offset + 100)

    # Apply pagination
    paginated = all_runs[offset:offset + limit]

    # Convert to response items
    items = []
    for run_data in paginated:
        results_data = run_data.get("results", {})
        weighted_score = results_data.get("weighted_score") if results_data else None

        items.append(EvalRunListItem(
            run_id=run_data["run_id"],
            name=run_data.get("name", f"eval-{run_data['run_id']}"),
            status=run_data.get("status", "unknown"),
            created_at=datetime.fromisoformat(run_data["created_at"]),
            completed_at=datetime.fromisoformat(run_data["completed_at"]) if run_data.get("completed_at") else None,
            weighted_score=weighted_score,
            groups=run_data.get("groups", []),
            datasets=run_data.get("datasets", []),
            question_count=run_data.get("total_questions", 0),
        ))

    return EvalRunListResponse(
        runs=items,
        total=len(all_runs),
        limit=limit,
        offset=offset,
    )


@router.delete("/metrics/eval/runs/{run_id}", status_code=204)
async def delete_evaluation_run_endpoint(run_id: str):
    """Delete or cancel an evaluation run."""
    # Check if run exists
    progress_data = get_eval_progress(run_id)
    if not progress_data:
        raise HTTPException(status_code=404, detail=f"Evaluation run '{run_id}' not found")

    # If running, cancel it
    status = progress_data.get("status")
    if status in ("pending", "running"):
        cancel_eval_run(run_id)
        # TODO: Also revoke the Celery task
        logger.info(f"[EVAL] Cancelled evaluation run {run_id}")
    else:
        # Delete completed/failed run
        delete_eval_progress(run_id)
        logger.info(f"[EVAL] Deleted evaluation run {run_id}")


# ============================================================================
# SSE Progress Endpoint
# ============================================================================


@router.get("/metrics/eval/runs/{run_id}/progress")
async def stream_evaluation_progress(run_id: str):
    """Stream evaluation progress via Server-Sent Events."""

    async def event_generator():
        last_completed = -1
        last_phase = None
        poll_count = 0
        max_polls = 3600  # 1 hour at 1 poll/second

        while poll_count < max_polls:
            poll_count += 1

            progress_data = get_eval_progress(run_id)

            if not progress_data:
                yield f"event: error\ndata: {json.dumps({'error': 'Run not found', 'recoverable': False})}\n\n"
                break

            status = progress_data.get("status")
            phase = progress_data.get("phase")
            completed = progress_data.get("completed_questions", 0)
            total = progress_data.get("total_questions", 1)

            # Send update if changed
            if completed != last_completed or phase != last_phase:
                event_data = {
                    "phase": phase,
                    "completed": completed,
                    "total": total,
                    "percent": _calculate_percent(completed, total),
                    "current_dataset": progress_data.get("current_dataset"),
                    "metrics_computed": progress_data.get("metrics_computed", []),
                }
                yield f"event: progress\ndata: {json.dumps(event_data)}\n\n"
                last_completed = completed
                last_phase = phase

            # Check for completion
            if status == "completed":
                results_data = progress_data.get("results", {})
                yield f"event: complete\ndata: {json.dumps({'run_id': run_id, 'weighted_score': results_data.get('weighted_score')})}\n\n"
                break
            elif status == "failed":
                errors = progress_data.get("errors", [])
                error_msg = errors[-1]["error"] if errors else "Unknown error"
                yield f"event: error\ndata: {json.dumps({'error': error_msg, 'recoverable': False})}\n\n"
                break
            elif status == "cancelled":
                yield f"event: error\ndata: {json.dumps({'error': 'Run was cancelled', 'recoverable': False})}\n\n"
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# Helper Functions
# ============================================================================


def _calculate_percent(completed: int, total: int) -> int:
    """Calculate completion percentage."""
    if total <= 0:
        return 0
    return min(100, int(completed / total * 100))


def _parse_results(results_data: dict) -> EvalRunResults:
    """Parse results data into EvalRunResults."""
    groups = {}
    for group_id, group_data in results_data.get("groups", {}).items():
        metrics = [
            MetricValue(
                name=m["name"],
                value=m["value"],
                sample_size=m.get("sample_size", 0),
            )
            for m in group_data.get("metrics", [])
        ]
        groups[group_id] = GroupResults(
            average=group_data.get("average", 0.0),
            metrics=metrics,
        )

    performance = None
    perf_data = results_data.get("performance")
    if perf_data:
        performance = PerformanceResults(
            latency_p50_ms=perf_data.get("latency_p50_ms", 0),
            latency_p95_ms=perf_data.get("latency_p95_ms", 0),
            latency_avg_ms=perf_data.get("latency_avg_ms", 0),
            cost_total_usd=perf_data.get("cost_total_usd", 0),
        )

    return EvalRunResults(
        weighted_score=results_data.get("weighted_score"),
        groups=groups,
        performance=performance,
    )


# ============================================================================
# Analysis Endpoints (migrated from /metrics/*)
# ============================================================================


@router.get("/metrics/eval/definitions", response_model=list[MetricDefinition])
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
        logger.error(f"[EVAL] Error fetching metric definitions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/eval/summary", response_model=EvaluationSummary)
async def get_eval_summary_endpoint():
    """Get evaluation summary with trends.

    Returns:
    - Latest evaluation run
    - Total number of runs
    - Metric trends over time (improving/declining/stable)
    - Best performing run
    """
    try:
        return get_evaluation_summary()
    except Exception as e:
        logger.error(f"[EVAL] Error fetching evaluation summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Baseline Management Endpoints
# ============================================================================


@router.get("/metrics/eval/baseline", response_model=GoldenBaseline | None)
async def get_golden_baseline():
    """Get the current golden baseline.

    Returns:
        The golden baseline if set, null otherwise
    """
    try:
        return get_baseline()
    except Exception as e:
        logger.error(f"[EVAL] Error fetching baseline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/eval/baseline/{run_id}", response_model=GoldenBaseline)
async def set_golden_baseline(run_id: str, set_by: str | None = None):
    """Set a specific evaluation run as the golden baseline.

    The baseline's metric scores become the thresholds to beat.
    New evaluation runs will be compared against this baseline.

    Args:
        run_id: ID of the evaluation run to set as baseline
        set_by: Optional identifier for who set the baseline
    """
    try:
        run = get_evaluation_run_by_id(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Evaluation run '{run_id}' not found")

        return set_baseline(run, set_by=set_by)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[EVAL] Error setting baseline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/metrics/eval/baseline", status_code=204)
async def clear_golden_baseline_endpoint():
    """Clear the current golden baseline."""
    try:
        clear_baseline()
    except Exception as e:
        logger.error(f"[EVAL] Error clearing baseline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Comparison Endpoints
# ============================================================================


@router.get("/metrics/eval/compare/{run_a_id}/{run_b_id}", response_model=ComparisonResult)
async def compare_runs_endpoint(run_a_id: str, run_b_id: str):
    """Compare two evaluation runs side-by-side.

    Returns metric deltas, latency/cost comparison, and winner determination.

    Args:
        run_a_id: ID of first evaluation run
        run_b_id: ID of second evaluation run
    """
    try:
        run_a = get_evaluation_run_by_id(run_a_id)
        run_b = get_evaluation_run_by_id(run_b_id)

        if run_a is None:
            raise HTTPException(status_code=404, detail=f"Evaluation run '{run_a_id}' not found")
        if run_b is None:
            raise HTTPException(status_code=404, detail=f"Evaluation run '{run_b_id}' not found")

        return compare_runs(run_a, run_b)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[EVAL] Error comparing runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/eval/compare-to-baseline/{run_id}", response_model=ComparisonResult)
async def compare_to_baseline(run_id: str):
    """Compare a run against the golden baseline.

    Args:
        run_id: ID of the evaluation run to compare
    """
    try:
        baseline = get_baseline()

        if baseline is None:
            raise HTTPException(status_code=404, detail="No golden baseline set")

        run = get_evaluation_run_by_id(run_id)
        baseline_run = get_evaluation_run_by_id(baseline.run_id)

        if run is None:
            raise HTTPException(status_code=404, detail=f"Evaluation run '{run_id}' not found")
        if baseline_run is None:
            raise HTTPException(status_code=404, detail=f"Baseline run '{baseline.run_id}' not found")

        return compare_runs(run, baseline_run)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[EVAL] Error comparing to baseline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Recommendation Endpoint
# ============================================================================


@router.post("/metrics/eval/recommend", response_model=Recommendation)
async def get_recommendation_endpoint(
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
        logger.error(f"[EVAL] Error getting recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
