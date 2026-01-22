"""Comparison functions for evaluation runs.

Provides side-by-side comparison of two evaluation runs,
calculating deltas for metrics, latency, and cost.
"""

import logging

from schemas.metrics import LatencyMetrics, CostMetrics
from schemas.eval import ComparisonResult, EvaluationRun

logger = logging.getLogger(__name__)

# Metrics where lower is better
LOWER_IS_BETTER = {"hallucination"}

# Weights for winner determination
DEFAULT_WEIGHTS = {
    "metrics": 0.6,
    "latency": 0.2,
    "cost": 0.2,
}


def compare_runs(
    run_a: EvaluationRun,
    run_b: EvaluationRun,
) -> ComparisonResult:
    """Compare two evaluation runs."""
    metric_deltas = _compare_metrics(
        run_a.metric_averages,
        run_b.metric_averages,
    )

    latency_delta, latency_pct = _compare_latency(
        run_a.latency,
        run_b.latency,
    )

    cost_delta, cost_pct = _compare_cost(
        run_a.cost,
        run_b.cost,
    )

    winner, reason = _determine_winner(
        metric_deltas,
        latency_delta,
        cost_delta,
        run_a,
        run_b,
    )

    return ComparisonResult(
        run_a_id=run_a.run_id,
        run_b_id=run_b.run_id,
        run_a_config=run_a.config_snapshot,
        run_b_config=run_b.config_snapshot,
        metric_deltas=metric_deltas,
        latency_delta_ms=latency_delta,
        latency_improvement_pct=latency_pct,
        cost_delta_usd=cost_delta,
        cost_improvement_pct=cost_pct,
        winner=winner,
        winner_reason=reason,
    )


def _compare_metrics(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
) -> dict[str, float]:
    """Returns dict of metric deltas (positive = A is better)."""
    deltas = {}
    all_metrics = set(metrics_a.keys()) | set(metrics_b.keys())

    for metric in all_metrics:
        val_a = metrics_a.get(metric)
        val_b = metrics_b.get(metric)

        if val_a is None or val_b is None:
            continue

        if metric in LOWER_IS_BETTER:
            delta = val_b - val_a  # Positive = A has lower (better) value
        else:
            delta = val_a - val_b  # Positive = A has higher (better) value

        deltas[metric] = round(delta, 4)

    return deltas


def _compare_latency(
    latency_a: LatencyMetrics | None,
    latency_b: LatencyMetrics | None,
) -> tuple[float | None, float | None]:
    """Returns (delta_ms, improvement_pct). Positive delta = A is faster."""
    if not latency_a or not latency_b:
        return None, None

    delta = latency_b.p95_query_time_ms - latency_a.p95_query_time_ms

    if latency_b.p95_query_time_ms > 0:
        pct = (delta / latency_b.p95_query_time_ms) * 100
    else:
        pct = 0.0

    return round(delta, 2), round(pct, 1)


def _compare_cost(
    cost_a: CostMetrics | None,
    cost_b: CostMetrics | None,
) -> tuple[float | None, float | None]:
    """Returns (delta_usd, improvement_pct). Positive delta = A is cheaper."""
    if not cost_a or not cost_b:
        return None, None

    delta = cost_b.cost_per_query_usd - cost_a.cost_per_query_usd

    if cost_b.cost_per_query_usd > 0:
        pct = (delta / cost_b.cost_per_query_usd) * 100
    else:
        pct = 0.0

    return round(delta, 6), round(pct, 1)


def _determine_winner(
    metric_deltas: dict[str, float],
    latency_delta: float | None,
    cost_delta: float | None,
    run_a: EvaluationRun,
    run_b: EvaluationRun,
) -> tuple[str, str]:
    """Returns (winner, reason)."""
    if metric_deltas:
        avg_delta = sum(metric_deltas.values()) / len(metric_deltas)
    else:
        avg_delta = 0.0

    reasons = []

    a_better_metrics = sum(1 for d in metric_deltas.values() if d > 0.01)
    b_better_metrics = sum(1 for d in metric_deltas.values() if d < -0.01)

    if a_better_metrics > b_better_metrics:
        reasons.append(f"A better on {a_better_metrics} metrics")
    elif b_better_metrics > a_better_metrics:
        reasons.append(f"B better on {b_better_metrics} metrics")

    if latency_delta is not None:
        if latency_delta > 50:
            reasons.append("A is faster")
        elif latency_delta < -50:
            reasons.append("B is faster")

    if cost_delta is not None:
        if cost_delta > 0.001:
            reasons.append("A is cheaper")
        elif cost_delta < -0.001:
            reasons.append("B is cheaper")

    score_a = 0.0
    score_b = 0.0

    if avg_delta > 0.01:
        score_a += DEFAULT_WEIGHTS["metrics"]
    elif avg_delta < -0.01:
        score_b += DEFAULT_WEIGHTS["metrics"]

    if latency_delta is not None:
        if latency_delta > 50:
            score_a += DEFAULT_WEIGHTS["latency"]
        elif latency_delta < -50:
            score_b += DEFAULT_WEIGHTS["latency"]

    if cost_delta is not None:
        if cost_delta > 0.001:
            score_a += DEFAULT_WEIGHTS["cost"]
        elif cost_delta < -0.001:
            score_b += DEFAULT_WEIGHTS["cost"]

    if score_a > score_b + 0.1:
        winner = "run_a"
        model_name = (
            run_a.config_snapshot.llm_model if run_a.config_snapshot else run_a.run_id
        )
        reason = (
            f"{model_name}: " + ", ".join(reasons) if reasons else "Higher overall score"
        )
    elif score_b > score_a + 0.1:
        winner = "run_b"
        model_name = (
            run_b.config_snapshot.llm_model if run_b.config_snapshot else run_b.run_id
        )
        reason = (
            f"{model_name}: " + ", ".join(reasons) if reasons else "Higher overall score"
        )
    else:
        winner = "tie"
        reason = "Similar performance across metrics, latency, and cost"

    return winner, reason
