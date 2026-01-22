"""Celery task for running evaluations asynchronously.

Uses lazy imports to avoid loading heavy evaluation dependencies
(HuggingFace datasets, etc.) during API startup.
"""

import logging
import traceback
from typing import Any, TYPE_CHECKING

from infrastructure.tasks.celery_app import celery_app
from infrastructure.tasks.eval_progress import (
    update_eval_progress,
    complete_eval_run,
    fail_eval_run,
)

if TYPE_CHECKING:
    from evaluation_cc.config import EvalConfig
    from evaluation_cc.runner import EvaluationRunner
    from evaluation_cc.schemas import MetricGroup

logger = logging.getLogger(__name__)


def _format_results(eval_run, MetricGroup) -> dict[str, Any]:
    """Format EvalRun results for API response."""
    results = {
        "weighted_score": None,
        "groups": {},
        "performance": None,
    }

    if eval_run.weighted_score:
        results["weighted_score"] = eval_run.weighted_score.score

    if eval_run.scorecard:
        # Group metrics by their group
        for group, metrics in eval_run.scorecard.by_group.items():
            group_metrics = []
            total_value = 0.0
            for metric in metrics:
                group_metrics.append({
                    "name": metric.name,
                    "value": metric.value,
                    "sample_size": metric.sample_size,
                })
                total_value += metric.value

            avg_value = total_value / len(metrics) if metrics else 0.0
            results["groups"][group.value] = {
                "average": avg_value,
                "metrics": group_metrics,
            }

        # Extract performance metrics
        perf_metrics = eval_run.scorecard.by_group.get(MetricGroup.PERFORMANCE, [])
        if perf_metrics:
            perf_dict = {m.name: m.value for m in perf_metrics}
            results["performance"] = {
                "latency_p50_ms": perf_dict.get("latency_p50_ms", 0),
                "latency_p95_ms": perf_dict.get("latency_p95_ms", 0),
                "latency_avg_ms": perf_dict.get("latency_avg_ms", 0),
                "cost_total_usd": 0.0,  # Would need token tracking
            }

    return results


@celery_app.task(
    bind=True,
    name="infrastructure.tasks.eval_worker.run_evaluation_task",
    queue="eval",
    soft_time_limit=3600,  # 1 hour soft limit
    time_limit=3900,  # 1 hour 5 min hard limit
)
def run_evaluation_task(
    self,
    run_id: str,
    name: str,
    groups: list[str],
    datasets: list[str],
    samples_per_dataset: int,
    judge_config: dict[str, Any] | None,
    metrics_selection: dict[str, list[str]] | None,
    seed: int | None,
) -> dict[str, Any]:
    """Celery task for running evaluations asynchronously.

    Args:
        run_id: Unique run identifier
        name: Run name
        groups: Metric groups to evaluate
        datasets: Dataset names to use
        samples_per_dataset: Number of samples per dataset
        judge_config: Judge configuration dict
        metrics_selection: Optional per-group metric selection
        seed: Random seed for reproducibility

    Returns:
        Result dict with run_id and status
    """
    logger.info(f"[EVAL_TASK {run_id}] Starting evaluation: {name}")

    try:
        # Lazy imports to avoid loading heavy dependencies at module load time
        from evaluation_cc.config import (
            EvalConfig,
            DatasetName,
            MetricConfig,
            JudgeConfig as EvalJudgeConfig,
        )
        from evaluation_cc.runner import EvaluationRunner
        from evaluation_cc.schemas import MetricGroup

        # Build MetricConfig from groups
        metric_config = MetricConfig(
            retrieval="retrieval" in groups,
            generation="generation" in groups,
            citation="citation" in groups,
            abstention="abstention" in groups,
            performance="performance" in groups,
        )

        # Build JudgeConfig
        if judge_config and judge_config.get("enabled"):
            judge = EvalJudgeConfig(
                enabled=True,
                provider=judge_config.get("provider", "anthropic"),
                model=judge_config.get("model", "claude-sonnet-4-20250514"),
            )
        else:
            judge = EvalJudgeConfig(enabled=False)

        # Convert dataset names to enum
        dataset_enums = []
        for ds_name in datasets:
            try:
                dataset_enums.append(DatasetName(ds_name))
            except ValueError:
                logger.warning(f"[EVAL_TASK {run_id}] Unknown dataset: {ds_name}")
                update_eval_progress(run_id, error=f"Unknown dataset: {ds_name}")

        if not dataset_enums:
            raise ValueError("No valid datasets specified")

        # Build EvalConfig
        config = EvalConfig(
            datasets=dataset_enums,
            samples_per_dataset=samples_per_dataset,
            metrics=metric_config,
            judge=judge,
            seed=seed,
            rag_server_url="http://localhost:8001",
        )

        # Update status to running
        all_metrics = []
        if metric_config.retrieval:
            all_metrics.extend(["recall_at_k", "precision_at_k", "mrr", "ndcg"])
        if metric_config.generation:
            all_metrics.extend(["faithfulness", "answer_correctness", "answer_relevancy"])
        if metric_config.citation:
            all_metrics.extend(["citation_precision", "citation_recall", "section_accuracy"])
        if metric_config.abstention:
            all_metrics.extend(["unanswerable_accuracy", "false_positive_rate", "false_negative_rate"])
        if metric_config.performance:
            all_metrics.extend(["latency_p50", "latency_p95", "cost_per_query"])

        update_eval_progress(
            run_id,
            status="running",
            phase="loading",
            metrics_pending=all_metrics,
        )

        # Create and run evaluation
        runner = EvaluationRunner(config)

        try:
            # Update to querying phase
            update_eval_progress(run_id, phase="querying")

            # Run the evaluation
            eval_run = runner.run(name=name)

            # Update to computing_metrics phase
            update_eval_progress(
                run_id,
                phase="computing_metrics",
                completed_questions=eval_run.question_count - eval_run.error_count,
                metrics_computed=all_metrics,
                metrics_pending=[],
            )

            # Format results for API
            results = _format_results(eval_run, MetricGroup)

            # Complete the run
            complete_eval_run(run_id, results)

            logger.info(f"[EVAL_TASK {run_id}] Completed successfully")
            return {
                "run_id": run_id,
                "status": "completed",
                "weighted_score": results.get("weighted_score"),
            }

        finally:
            runner.close()

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[EVAL_TASK {run_id}] Failed: {error_msg}")
        logger.error(traceback.format_exc())
        fail_eval_run(run_id, error_msg)
        raise
