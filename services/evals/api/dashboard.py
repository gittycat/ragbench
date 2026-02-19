"""Map a scorecard (from JSON run file) to 5 high-level dashboard metrics."""

from api.schemas import DashboardMetrics


def compute_dashboard_metrics(scorecard: dict | None, tier: str = "") -> DashboardMetrics | None:
    """Derive dashboard metrics from a raw scorecard dict.

    Args:
        scorecard: The "scorecard" dict as stored in the JSON run file.
        tier: "generation" or "end_to_end". Controls whether retrieval metrics exist.

    Returns:
        DashboardMetrics or None if no scorecard data.
    """
    if not scorecard:
        return None

    metrics_list = scorecard.get("metrics", [])
    if not metrics_list:
        return None

    lookup: dict[str, float] = {m["name"]: m["value"] for m in metrics_list}

    # Retrieval Relevance = avg(recall_at_5, mrr) — null for generation-only
    retrieval_relevance = None
    if tier != "generation":
        recall = lookup.get("recall_at_5")
        mrr = lookup.get("mrr")
        vals = [v for v in (recall, mrr) if v is not None]
        if vals:
            retrieval_relevance = sum(vals) / len(vals)

    faithfulness = lookup.get("faithfulness")
    answer_completeness = lookup.get("answer_correctness")
    answer_relevance = lookup.get("answer_relevancy")

    latency_p50 = lookup.get("latency_p50_ms")
    latency_p95 = lookup.get("latency_p95_ms")

    return DashboardMetrics(
        retrieval_relevance=retrieval_relevance,
        faithfulness=faithfulness,
        answer_completeness=answer_completeness,
        answer_relevance=answer_relevance,
        latency_p50_seconds=latency_p50 / 1000 if latency_p50 is not None else None,
        latency_p95_seconds=latency_p95 / 1000 if latency_p95 is not None else None,
    )
