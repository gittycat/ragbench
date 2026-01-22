"""Golden baseline management for evaluation runs.

Provides functions to:
- Get/set/clear the golden baseline
- Check runs against the baseline
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from schemas.metrics import ConfigSnapshot
from schemas.eval import (
    BaselineCheckResult,
    EvaluationRun,
    GoldenBaseline,
)

logger = logging.getLogger(__name__)

# Baseline file path (Docker or local)
_BASELINE_FILE = (
    Path("/app/eval_data/golden_baseline.json")
    if Path("/app/eval_data").exists()
    else Path("eval_data/golden_baseline.json")
)


def get_baseline(baseline_path: Path | None = None) -> GoldenBaseline | None:
    """Load the current golden baseline."""
    path = baseline_path or _BASELINE_FILE
    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)
        return GoldenBaseline(**data)
    except Exception as e:
        logger.error(f"Failed to load baseline: {e}")
        return None


def set_baseline(
    run: EvaluationRun,
    set_by: str | None = None,
    baseline_path: Path | None = None,
) -> GoldenBaseline:
    """Set an evaluation run as the golden baseline."""
    path = baseline_path or _BASELINE_FILE

    if run.config_snapshot:
        config_snapshot = run.config_snapshot
    else:
        config_snapshot = _create_config_snapshot_from_legacy(run.retrieval_config)

    baseline = GoldenBaseline(
        run_id=run.run_id,
        set_at=datetime.utcnow(),
        set_by=set_by,
        target_metrics=run.metric_averages,
        config_snapshot=config_snapshot,
        target_latency_p95_ms=run.latency.p95_query_time_ms if run.latency else None,
        target_cost_per_query_usd=run.cost.cost_per_query_usd if run.cost else None,
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(baseline.model_dump(mode="json"), f, indent=2, default=str)

    logger.info(f"Golden baseline set to run {run.run_id}")
    return baseline


def clear_baseline(baseline_path: Path | None = None) -> bool:
    """Clear the current golden baseline. Returns True if cleared."""
    path = baseline_path or _BASELINE_FILE
    if not path.exists():
        return False

    path.unlink()
    logger.info("Golden baseline cleared")
    return True


def check_against_baseline(
    run: EvaluationRun,
    baseline_path: Path | None = None,
) -> BaselineCheckResult | None:
    """Check if a run passes the golden baseline."""
    baseline = get_baseline(baseline_path)
    if baseline is None:
        return None

    metrics_pass = []
    metrics_fail = []
    metric_deltas = {}

    for metric_name, target_value in baseline.target_metrics.items():
        actual_value = run.metric_averages.get(metric_name)
        if actual_value is None:
            continue

        # For hallucination, lower is better
        if metric_name == "hallucination":
            passed = actual_value <= target_value
            delta = target_value - actual_value
        else:
            passed = actual_value >= target_value
            delta = actual_value - target_value

        if passed:
            metrics_pass.append(metric_name)
        else:
            metrics_fail.append(metric_name)

        metric_deltas[metric_name] = round(delta, 4)

    return BaselineCheckResult(
        baseline_run_id=baseline.run_id,
        checked_run_id=run.run_id,
        metrics_pass=metrics_pass,
        metrics_fail=metrics_fail,
        overall_pass=len(metrics_fail) == 0,
        metric_deltas=metric_deltas,
    )


def _create_config_snapshot_from_legacy(retrieval_config: dict | None) -> ConfigSnapshot:
    """Convert legacy retrieval_config dict to ConfigSnapshot."""
    if not retrieval_config:
        return ConfigSnapshot(
            llm_provider="unknown",
            llm_model="unknown",
            embedding_provider="ollama",
            embedding_model="nomic-embed-text:latest",
            retrieval_top_k=10,
            hybrid_search_enabled=True,
            rrf_k=60,
            contextual_retrieval_enabled=False,
            reranker_enabled=True,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            reranker_top_n=5,
        )

    hybrid = retrieval_config.get("hybrid_search", {})
    reranker = retrieval_config.get("reranker", {})
    contextual = retrieval_config.get("contextual_retrieval", {})

    return ConfigSnapshot(
        llm_provider="unknown",
        llm_model="unknown",
        embedding_provider="ollama",
        embedding_model="nomic-embed-text:latest",
        retrieval_top_k=retrieval_config.get("retrieval_top_k", 10),
        hybrid_search_enabled=hybrid.get("enabled", True),
        rrf_k=hybrid.get("rrf_k", 60),
        contextual_retrieval_enabled=contextual.get("enabled", False),
        reranker_enabled=reranker.get("enabled", True),
        reranker_model=reranker.get("model"),
        reranker_top_n=reranker.get("top_n", 5),
    )
