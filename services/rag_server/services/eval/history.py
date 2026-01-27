"""Evaluation history management.

Provides functions to:
- Save and load evaluation runs from disk
- Query evaluation history
- Calculate evaluation summaries and trends
- Get metric definitions
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from schemas.eval import (
    EvaluationRun,
    EvaluationHistory,
    EvaluationSummary,
    MetricTrend,
    MetricDefinition,
)

logger = logging.getLogger(__name__)

# Use Docker path if available, otherwise local development path
EVAL_RESULTS_DIR = (
    Path("/app/evals/data/results")
    if Path("/app/evals/data").exists()
    else Path("evals/data/results")
)


def ensure_eval_results_dir():
    """Ensure evaluation results directory exists."""
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_evaluation_run(run: EvaluationRun) -> Path:
    """Save an evaluation run to disk."""
    ensure_eval_results_dir()

    filename = f"eval_run_{run.run_id}_{run.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    filepath = EVAL_RESULTS_DIR / filename

    with open(filepath, "w") as f:
        json.dump(run.model_dump(), f, indent=2, default=str)

    logger.info(f"Saved evaluation run to {filepath}")
    return filepath


def load_evaluation_history(limit: int = 20) -> EvaluationHistory:
    """Load evaluation history from disk."""
    ensure_eval_results_dir()

    runs = []
    result_files = sorted(
        EVAL_RESULTS_DIR.glob("eval_run_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]

    for filepath in result_files:
        try:
            with open(filepath) as f:
                data = json.load(f)
                if isinstance(data.get("timestamp"), str):
                    data["timestamp"] = datetime.fromisoformat(
                        data["timestamp"].replace("Z", "+00:00")
                    )
                runs.append(EvaluationRun(**data))
        except Exception as e:
            logger.warning(f"Failed to load evaluation run from {filepath}: {e}")

    return EvaluationHistory(runs=runs)


def get_evaluation_run_by_id(run_id: str) -> EvaluationRun | None:
    """Load a specific evaluation run by ID."""
    ensure_eval_results_dir()

    for filepath in EVAL_RESULTS_DIR.glob("eval_run_*.json"):
        try:
            with open(filepath) as f:
                data = json.load(f)
                if data.get("run_id") == run_id:
                    if isinstance(data.get("timestamp"), str):
                        data["timestamp"] = datetime.fromisoformat(
                            data["timestamp"].replace("Z", "+00:00")
                        )
                    return EvaluationRun(**data)
        except Exception as e:
            logger.warning(f"Failed to load evaluation run from {filepath}: {e}")

    return None


def delete_evaluation_run(run_id: str) -> bool:
    """Delete a specific evaluation run by ID. Returns True if deleted."""
    ensure_eval_results_dir()

    for filepath in EVAL_RESULTS_DIR.glob("eval_run_*.json"):
        try:
            with open(filepath) as f:
                data = json.load(f)
                if data.get("run_id") == run_id:
                    filepath.unlink()
                    logger.info(f"Deleted evaluation run {run_id} from {filepath}")
                    return True
        except Exception as e:
            logger.warning(f"Failed to check evaluation run from {filepath}: {e}")

    return False


def get_evaluation_summary() -> EvaluationSummary:
    """Get summary of evaluation history with trends."""
    history = load_evaluation_history()

    if not history.runs:
        return EvaluationSummary(
            latest_run=None,
            total_runs=0,
            metric_trends=[],
            best_run=None,
        )

    sorted_runs = sorted(history.runs, key=lambda r: r.timestamp)

    metric_trends = []
    metric_names = [
        "contextual_precision",
        "contextual_recall",
        "faithfulness",
        "answer_relevancy",
        "hallucination",
    ]

    for metric_name in metric_names:
        values = []
        timestamps = []

        for run in sorted_runs:
            if metric_name in run.metric_averages:
                values.append(run.metric_averages[metric_name])
                timestamps.append(run.timestamp)

        if values:
            if len(values) >= 2:
                recent_avg = sum(values[-3:]) / len(values[-3:])
                older_avg = sum(values[:3]) / len(values[:3])
                if recent_avg > older_avg + 0.05:
                    trend = "improving"
                elif recent_avg < older_avg - 0.05:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            metric_trends.append(
                MetricTrend(
                    metric_name=metric_name,
                    values=values,
                    timestamps=timestamps,
                    trend_direction=trend,
                    latest_value=values[-1],
                    average_value=sum(values) / len(values),
                )
            )

    # Find best run (highest average of non-hallucination metrics)
    best_run = None
    best_score = -1
    for run in history.runs:
        if run.metric_averages:
            scores = []
            for m in [
                "contextual_precision",
                "contextual_recall",
                "faithfulness",
                "answer_relevancy",
            ]:
                if m in run.metric_averages:
                    scores.append(run.metric_averages[m])
            if "hallucination" in run.metric_averages:
                scores.append(1 - run.metric_averages["hallucination"])

            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_run = run

    return EvaluationSummary(
        latest_run=sorted_runs[-1] if sorted_runs else None,
        total_runs=len(history.runs),
        metric_trends=metric_trends,
        best_run=best_run,
    )


def get_metric_definitions() -> list[MetricDefinition]:
    """Get definitions for all evaluation metrics."""
    return [
        MetricDefinition(
            name="precision_at_k",
            category="retrieval",
            description="Fraction of top-K retrieved chunks that are relevant.",
            threshold=0.5,
            interpretation="Score 0-1. Higher is better. Measures how much noise is in top-K retrieval.",
            reference_url="https://en.wikipedia.org/wiki/Precision_and_recall",
        ),
        MetricDefinition(
            name="recall_at_k",
            category="retrieval",
            description="Fraction of gold evidence covered by top-K retrieved chunks.",
            threshold=0.5,
            interpretation="Score 0-1. Higher is better. Measures coverage of relevant evidence.",
            reference_url="https://en.wikipedia.org/wiki/Precision_and_recall",
        ),
        MetricDefinition(
            name="mrr",
            category="retrieval",
            description="Mean reciprocal rank of the first relevant chunk.",
            threshold=0.3,
            interpretation="Score 0-1. Higher is better. Measures how early the first relevant chunk appears.",
            reference_url="https://en.wikipedia.org/wiki/Mean_reciprocal_rank",
        ),
        MetricDefinition(
            name="ndcg",
            category="retrieval",
            description="Normalized Discounted Cumulative Gain of ranked retrieval results.",
            threshold=0.5,
            interpretation="Score 0-1. Higher is better. Measures ranking quality with position discounts.",
            reference_url="https://en.wikipedia.org/wiki/Discounted_cumulative_gain",
        ),
        MetricDefinition(
            name="citation_precision",
            category="retrieval",
            description="Fraction of cited chunks that match gold evidence or gold documents.",
            threshold=0.6,
            interpretation="Score 0-1. Higher is better. Measures correctness of citations.",
        ),
        MetricDefinition(
            name="citation_recall",
            category="retrieval",
            description="Fraction of gold evidence that is covered by cited chunks.",
            threshold=0.6,
            interpretation="Score 0-1. Higher is better. Measures citation coverage.",
        ),
        MetricDefinition(
            name="faithfulness",
            category="generation",
            description="Measures whether the answer is grounded in the retrieved context without adding unsupported claims.",
            threshold=0.7,
            interpretation="Score 0-1. Above 0.7 is good. Measures: Is the answer supported by the context?",
            reference_url="https://docs.confident-ai.com/docs/metrics-faithfulness",
        ),
        MetricDefinition(
            name="answer_relevancy",
            category="generation",
            description="Measures whether the generated answer actually addresses the user's question.",
            threshold=0.7,
            interpretation="Score 0-1. Above 0.7 is good. Measures: Does the answer address the question asked?",
            reference_url="https://docs.confident-ai.com/docs/metrics-answer-relevancy",
        ),
        MetricDefinition(
            name="hallucination",
            category="safety",
            description="Measures the proportion of the answer that contains hallucinated (unsupported) information.",
            threshold=0.5,
            interpretation="Score 0-1. Below 0.5 is good (lower = less hallucination). Measures: How much is made up?",
            reference_url="https://docs.confident-ai.com/docs/metrics-hallucination",
        ),
        MetricDefinition(
            name="unanswerable_accuracy",
            category="safety",
            description="Measures how often the system correctly abstains on unanswerable questions.",
            threshold=0.7,
            interpretation="Score 0-1. Higher is better. Measures abstention correctness on unanswerable inputs.",
        ),
        MetricDefinition(
            name="answerable_abstain_rate",
            category="safety",
            description="Measures how often the system abstains on answerable questions.",
            threshold=0.2,
            interpretation="Score 0-1. Lower is better. Measures false abstentions on answerable inputs.",
        ),
        MetricDefinition(
            name="long_form_completeness",
            category="generation",
            description="Measures how much of the gold evidence is covered in long-form answers.",
            threshold=0.5,
            interpretation="Score 0-1. Higher is better. Measures evidence coverage for long-form tasks.",
        ),
    ]
