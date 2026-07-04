"""Unit tests for the additive evals API extensions (Part 1 of the dashboard redesign).

Covers: flat metrics map + groups on RunSummary, telemetry derivation from
cost_per_query details, widened compare deltas (union of metrics, null-safe),
and backward compat with legacy run dicts missing the new fields.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.dashboard import compute_dashboard_metrics
from api.job_manager import JobManager


def _scorecard(*, tier="end_to_end", include_retrieval=True, cost_details=None):
    metrics = []
    by_group = {}

    def add(name, value, group, details=None, sample_size=10):
        metrics.append({"name": name, "value": value, "group": group, "sample_size": sample_size, "details": details or {}})
        by_group.setdefault(group, []).append(name)

    if include_retrieval:
        add("recall_at_5", 0.8, "retrieval")
        add("mrr", 0.7, "retrieval")

    add("faithfulness", 0.9, "generation")
    add("answer_correctness", 0.85, "generation")
    add("answer_relevancy", 0.75, "generation")
    add("latency_p50_ms", 120.0, "performance")
    add("latency_p95_ms", 300.0, "performance")
    add("latency_avg_ms", 150.0, "performance")
    add(
        "cost_per_query",
        0.002,
        "performance",
        details=cost_details
        if cost_details is not None
        else {
            "avg_cost_usd": 0.002,
            "total_cost_usd": 0.2,
            "total_prompt_tokens": 1000,
            "total_completion_tokens": 200,
            "model": "claude-sonnet-5",
        },
    )

    return {"metrics": metrics, "by_group": by_group}


def _run_dict(run_id="run1", tier="end_to_end", **overrides):
    data = {
        "id": run_id,
        "name": f"eval-{run_id}",
        "created_at": "2026-07-01T00:00:00",
        "completed_at": "2026-07-01T00:10:00",
        "config": {},
        "scorecard": _scorecard(tier=tier),
        "weighted_score": {"score": 0.82, "weights": {}, "contributions": {}, "objectives": {}},
        "question_count": 100,
        "error_count": 0,
        "duration_seconds": 600.0,
        "metadata": {"tier": tier},
    }
    data.update(overrides)
    return data


@pytest.fixture
def jm(tmp_path):
    return JobManager(runs_dir=tmp_path)


class TestRunSummaryMetricsMap:
    def test_flat_metrics_and_groups(self, jm):
        summary = jm.run_to_summary(_run_dict())
        assert summary.metrics["faithfulness"] == 0.9
        assert summary.metrics["recall_at_5"] == 0.8
        assert summary.metrics["latency_p50_ms"] == 120.0
        assert set(summary.groups["retrieval"]) == {"recall_at_5", "mrr"}
        assert "cost_per_query" in summary.groups["performance"]

    def test_empty_scorecard_defaults(self, jm):
        data = _run_dict(scorecard=None)
        summary = jm.run_to_summary(data)
        assert summary.metrics == {}
        assert summary.groups == {}
        assert summary.dashboard_metrics is None


class TestTelemetryDerivation:
    def test_cost_and_latency_fields(self):
        scorecard = _scorecard()
        dm = compute_dashboard_metrics(scorecard, tier="end_to_end")
        assert dm.avg_cost_usd == 0.002
        assert dm.total_cost_usd == 0.2
        assert dm.total_prompt_tokens == 1000
        assert dm.total_completion_tokens == 200
        assert dm.cost_model == "claude-sonnet-5"
        assert dm.latency_avg_seconds == pytest.approx(0.15)

    def test_missing_cost_details_are_none(self):
        scorecard = _scorecard(cost_details={})
        dm = compute_dashboard_metrics(scorecard, tier="end_to_end")
        assert dm.avg_cost_usd is None
        assert dm.total_cost_usd is None
        assert dm.cost_model is None

    def test_retrieval_null_for_generation_tier(self):
        scorecard = _scorecard(tier="generation", include_retrieval=False)
        dm = compute_dashboard_metrics(scorecard, tier="generation")
        assert dm.retrieval_relevance is None


class TestCompareDeltas:
    def test_union_of_metrics_with_missing_side(self, jm):
        run_a = _run_dict(run_id="a", tier="end_to_end")
        run_b_scorecard = _scorecard(tier="generation", include_retrieval=False)
        run_b = _run_dict(run_id="b", tier="generation", scorecard=run_b_scorecard, metadata={"tier": "generation"})

        jm._run_index["a"] = (jm.runs_dir / "a.json", run_a)
        jm._run_index["b"] = (jm.runs_dir / "b.json", run_b)

        summary_a = jm.run_to_summary(run_a)
        summary_b = jm.run_to_summary(run_b)

        # recall_at_5 present in a, absent in b -> null delta
        metrics_union = set(summary_a.metrics) | set(summary_b.metrics)
        assert "recall_at_5" in metrics_union
        assert "recall_at_5" not in summary_b.metrics

    def test_weighted_score_present(self, jm):
        summary = jm.run_to_summary(_run_dict())
        assert summary.weighted_score == 0.82


class TestLegacyRunCompat:
    def test_legacy_run_without_new_fields_still_parses(self, jm):
        legacy = {
            "id": "legacy1",
            "name": "old-run",
            "created_at": "2026-01-01T00:00:00",
            "completed_at": "2026-01-01T00:05:00",
            "scorecard": {
                "metrics": [
                    {"name": "faithfulness", "value": 0.9, "group": "generation", "sample_size": 5, "details": {}},
                ],
                "by_group": {"generation": ["faithfulness"]},
            },
            "weighted_score": {"score": 0.7},
            "question_count": 5,
            "error_count": 0,
            "metadata": {"tier": "generation"},
        }
        summary = jm.run_to_summary(legacy)
        assert summary.metrics["faithfulness"] == 0.9
        assert summary.dashboard_metrics.avg_cost_usd is None
        assert summary.dashboard_metrics.total_prompt_tokens is None
