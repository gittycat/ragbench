"""Pydantic models for the Evaluation API (/metrics/eval/).

This module defines all request/response models for the evaluation API:
- Discovery: metric groups and datasets
- Execution: evaluation runs with progress tracking
- Results: run status and results
- Analysis: baseline, comparison, recommendations

## Schema Architecture

Three schema layers exist in this codebase:

1. **schemas/eval.py** (this file) - Pydantic models for all /metrics/eval/* endpoints
   - Used for: API request/response validation, OpenAPI docs, frontend integration
   - Contains: All eval-specific schemas

2. **schemas/metrics.py** - Pydantic models for system metrics endpoints
   - Used for: /metrics/system, /metrics/models, /metrics/retrieval
   - Contains: ConfigSnapshot, LatencyMetrics, CostMetrics (shared with eval)

3. **evals/schemas/** - Dataclasses for internal evaluation
   - Used for: EvaluationRunner, metric computation, result storage
   - Contains: EvalQuestion, EvalResponse, EvalRun, Scorecard
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# Reuse shared models from metrics.py (config, latency, cost)
from schemas.metrics import (
    ConfigSnapshot,
    LatencyMetrics,
    CostMetrics,
)


# ============================================================================
# Discovery Models
# ============================================================================


class MetricInfo(BaseModel):
    """Information about a single evaluation metric."""

    id: str = Field(..., description="Metric identifier (e.g., 'recall_at_k')")
    name: str = Field(..., description="Display name (e.g., 'Recall@K')")
    description: str = Field(..., description="What the metric measures")
    parameters: dict[str, Any] | None = Field(
        None, description="Configurable parameters (e.g., {'k': [1, 3, 5, 10]})"
    )
    requires_judge: bool = Field(
        False, description="Whether this metric requires LLM-as-judge"
    )


class MetricGroupResponse(BaseModel):
    """A group of related evaluation metrics."""

    id: str = Field(..., description="Group identifier (e.g., 'retrieval', 'generation')")
    name: str = Field(..., description="Display name (e.g., 'Retrieval Quality')")
    description: str = Field(..., description="What this group evaluates")
    metrics: list[MetricInfo] = Field(..., description="Metrics in this group")
    estimated_duration_per_sample_ms: int = Field(
        ..., description="Estimated time per sample in milliseconds"
    )
    requires_judge: bool = Field(
        ..., description="Whether any metric in this group requires LLM-as-judge"
    )
    recommended_datasets: list[str] = Field(
        ..., description="Datasets recommended for this group"
    )


class MetricGroupsResponse(BaseModel):
    """Response containing all available metric groups."""

    groups: list[MetricGroupResponse] = Field(..., description="Available metric groups")


class DatasetResponse(BaseModel):
    """Information about an evaluation dataset."""

    id: str = Field(..., description="Dataset identifier")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="What the dataset contains")
    size: int = Field(..., description="Number of samples in the dataset")
    domains: list[str] = Field(..., description="Domains covered by the dataset")
    primary_aspects: list[str] = Field(
        ..., description="Primary evaluation aspects (e.g., 'retrieval', 'generation')"
    )
    requires_download: bool = Field(
        ..., description="Whether dataset needs to be downloaded"
    )
    download_size_mb: int = Field(..., description="Download size in megabytes")


class DatasetsResponse(BaseModel):
    """Response containing all available datasets."""

    datasets: list[DatasetResponse] = Field(..., description="Available datasets")


# ============================================================================
# Execution Request Models
# ============================================================================


class JudgeConfig(BaseModel):
    """Configuration for LLM-as-judge evaluation."""

    enabled: bool = Field(True, description="Whether to enable LLM-as-judge")
    provider: str = Field("anthropic", description="LLM provider for judge")
    model: str = Field(
        "claude-sonnet-4-20250514", description="Model to use for judge"
    )


class EvalRunRequest(BaseModel):
    """Request body for starting an evaluation run."""

    name: str | None = Field(None, description="Optional name for the run")
    groups: list[str] = Field(
        ..., description="Metric groups to run (e.g., ['retrieval', 'generation'])"
    )
    metrics: dict[str, list[str]] | None = Field(
        None,
        description="Optional: specific metrics per group. If omitted, all metrics in selected groups run.",
    )
    datasets: list[str] = Field(
        default=["golden"], description="Datasets to evaluate on"
    )
    samples_per_dataset: int = Field(
        default=100, ge=1, le=10000, description="Number of samples per dataset"
    )
    judge: JudgeConfig | None = Field(
        None, description="Judge configuration. Required if generation group is selected."
    )
    seed: int | None = Field(42, description="Random seed for reproducibility")


# ============================================================================
# Progress and Response Models
# ============================================================================


class EvalRunProgress(BaseModel):
    """Progress information for an evaluation run."""

    phase: str = Field(
        ...,
        description="Current phase: 'loading', 'querying', 'computing_metrics', 'completed', 'failed'",
    )
    total_questions: int = Field(..., description="Total number of questions to evaluate")
    completed_questions: int = Field(..., description="Number of questions completed")
    current_dataset: str | None = Field(None, description="Dataset currently being processed")
    percent_complete: int = Field(..., ge=0, le=100, description="Completion percentage")
    metrics_computed: list[str] = Field(
        default_factory=list, description="Metrics that have been computed"
    )
    metrics_pending: list[str] = Field(
        default_factory=list, description="Metrics still pending computation"
    )


class EvalRunConfig(BaseModel):
    """Configuration snapshot for an evaluation run."""

    groups: list[str] = Field(..., description="Metric groups being evaluated")
    datasets: list[str] = Field(..., description="Datasets being used")
    samples_per_dataset: int = Field(..., description="Samples per dataset")
    total_samples: int = Field(..., description="Total samples across all datasets")
    judge_enabled: bool = Field(..., description="Whether LLM-as-judge is enabled")


class MetricValue(BaseModel):
    """A single metric result."""

    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value (0-1)")
    sample_size: int = Field(..., description="Number of samples used")


class GroupResults(BaseModel):
    """Results for a metric group."""

    average: float = Field(..., description="Average score across metrics in the group")
    metrics: list[MetricValue] = Field(..., description="Individual metric results")


class PerformanceResults(BaseModel):
    """Performance metrics from the run."""

    latency_p50_ms: float = Field(..., description="Median query latency in ms")
    latency_p95_ms: float = Field(..., description="95th percentile latency in ms")
    latency_avg_ms: float = Field(..., description="Average query latency in ms")
    cost_total_usd: float = Field(..., description="Total cost in USD")


class EvalRunResults(BaseModel):
    """Complete results from an evaluation run."""

    weighted_score: float | None = Field(
        None, description="Overall weighted score (0-1)"
    )
    groups: dict[str, GroupResults] = Field(
        default_factory=dict, description="Results per metric group"
    )
    performance: PerformanceResults | None = Field(
        None, description="Performance metrics"
    )


class EvalRunResponse(BaseModel):
    """Response for a single evaluation run."""

    run_id: str = Field(..., description="Unique run identifier")
    name: str = Field(..., description="Run name")
    status: str = Field(
        ..., description="Status: 'pending', 'running', 'completed', 'failed', 'cancelled'"
    )
    created_at: datetime = Field(..., description="When the run was created")
    completed_at: datetime | None = Field(None, description="When the run completed")
    duration_seconds: float | None = Field(None, description="Run duration in seconds")
    progress: EvalRunProgress | None = Field(None, description="Progress information")
    config: EvalRunConfig | None = Field(None, description="Run configuration")
    results: EvalRunResults | None = Field(
        None, description="Results (populated when completed)"
    )
    question_count: int = Field(0, description="Number of questions evaluated")
    error_count: int = Field(0, description="Number of errors encountered")


class EvalRunListItem(BaseModel):
    """Summary item for list of evaluation runs."""

    run_id: str = Field(..., description="Unique run identifier")
    name: str = Field(..., description="Run name")
    status: str = Field(..., description="Run status")
    created_at: datetime = Field(..., description="When the run was created")
    completed_at: datetime | None = Field(None, description="When the run completed")
    weighted_score: float | None = Field(None, description="Overall score if completed")
    groups: list[str] = Field(..., description="Metric groups evaluated")
    datasets: list[str] = Field(..., description="Datasets used")
    question_count: int = Field(0, description="Number of questions")


class EvalRunListResponse(BaseModel):
    """Paginated list of evaluation runs."""

    runs: list[EvalRunListItem] = Field(..., description="List of runs")
    total: int = Field(..., description="Total number of runs")
    limit: int = Field(..., description="Page size limit")
    offset: int = Field(..., description="Current offset")


# ============================================================================
# Error Response Model
# ============================================================================


class EvalErrorDetail(BaseModel):
    """Details about a validation error."""

    field: str | None = Field(None, description="Field that caused the error")
    required_by: list[str] | None = Field(
        None, description="Features that require this field"
    )


class EvalErrorResponse(BaseModel):
    """Error response format for eval API."""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: EvalErrorDetail | None = Field(None, description="Additional error details")


# ============================================================================
# Metric Definitions (for discovery)
# ============================================================================


class MetricDefinition(BaseModel):
    """Definition of an evaluation metric."""

    name: str = Field(..., description="Metric name")
    category: str = Field(..., description="Category: 'retrieval', 'generation', 'safety'")
    description: str = Field(..., description="What the metric measures")
    threshold: float = Field(..., description="Pass/fail threshold")
    interpretation: str = Field(..., description="How to interpret the score")
    reference_url: str | None = Field(None, description="Documentation URL")


# ============================================================================
# Test Case Results (detailed per-question results)
# ============================================================================


class MetricResult(BaseModel):
    """Result for a single metric on a test case."""

    metric_name: str = Field(..., description="Metric name")
    score: float = Field(..., description="Score (0-1)")
    passed: bool = Field(..., description="Whether score meets threshold")
    threshold: float = Field(..., description="Threshold used")
    reason: str | None = Field(None, description="Explanation for the score")


class TestCaseResult(BaseModel):
    """Result for a single test case."""

    test_id: str = Field(..., description="Test case identifier")
    question: str = Field(..., description="Question asked")
    expected_answer: str | None = Field(None, description="Expected answer")
    actual_answer: str = Field(..., description="RAG system answer")
    metrics: list[MetricResult] = Field(..., description="Per-metric results")
    passed: bool = Field(..., description="Whether all metrics passed")
    retrieval_context_count: int = Field(..., description="Number of retrieved chunks")


# ============================================================================
# Evaluation Run (complete run with results)
# ============================================================================


class BaselineCheckResult(BaseModel):
    """Result of checking a run against the golden baseline."""

    baseline_run_id: str = Field(..., description="ID of the baseline run")
    checked_run_id: str = Field(..., description="ID of the run being checked")
    metrics_pass: list[str] = Field(..., description="Metrics that passed baseline")
    metrics_fail: list[str] = Field(..., description="Metrics that failed baseline")
    overall_pass: bool = Field(..., description="Whether all metrics passed")
    metric_deltas: dict[str, float] = Field(
        ..., description="Delta from baseline per metric (positive = better)"
    )


class EvaluationRun(BaseModel):
    """Complete evaluation run results."""

    run_id: str = Field(..., description="Unique run identifier")
    timestamp: datetime = Field(..., description="When evaluation was run")
    framework: str = Field("DeepEval", description="Evaluation framework used")
    eval_model: str = Field(..., description="Model used for evaluation")

    # Summary statistics
    total_tests: int = Field(..., description="Total number of test cases")
    passed_tests: int = Field(..., description="Number of passing tests")
    pass_rate: float = Field(..., description="Percentage of tests passed")

    # Per-metric averages
    metric_averages: dict[str, float] = Field(..., description="Average score per metric")
    metric_pass_rates: dict[str, float] = Field(..., description="Pass rate per metric")

    # Configuration snapshot (legacy format, kept for backward compatibility)
    retrieval_config: dict | None = Field(None, description="Retrieval config at time of eval")

    # Enhanced configuration snapshot
    config_snapshot: ConfigSnapshot | None = Field(
        None, description="Full configuration snapshot at evaluation time"
    )

    # Latency metrics
    latency: LatencyMetrics | None = Field(
        None, description="Query latency statistics from this run"
    )

    # Cost metrics
    cost: CostMetrics | None = Field(
        None, description="Token usage and cost tracking for this run"
    )

    # Golden baseline flag
    is_golden_baseline: bool = Field(
        False, description="Whether this run is the golden baseline"
    )

    # Baseline comparison result
    compared_to_baseline: BaselineCheckResult | None = Field(
        None, description="Comparison result against golden baseline"
    )

    # Detailed results (optional, can be large)
    test_cases: list[TestCaseResult] | None = Field(None, description="Detailed per-test results")

    # Notes
    notes: str | None = Field(None, description="Notes about this evaluation run")


# ============================================================================
# History and Summary
# ============================================================================


class EvaluationHistory(BaseModel):
    """Historical evaluation runs for comparison."""

    runs: list[EvaluationRun] = Field(..., description="List of evaluation runs")
    comparison_metrics: list[str] = Field(
        default_factory=lambda: [
            "contextual_precision",
            "contextual_recall",
            "faithfulness",
            "answer_relevancy",
            "hallucination",
        ],
        description="Metrics to compare across runs",
    )


class MetricTrend(BaseModel):
    """Trend data for a single metric across evaluations."""

    metric_name: str = Field(..., description="Metric name")
    values: list[float] = Field(..., description="Score values over time")
    timestamps: list[datetime] = Field(..., description="Timestamps for each value")
    trend_direction: str = Field(..., description="'improving', 'declining', 'stable'")
    latest_value: float = Field(..., description="Most recent score")
    average_value: float = Field(..., description="Average across all runs")


class EvaluationSummary(BaseModel):
    """Summary of evaluation metrics with trends."""

    latest_run: EvaluationRun | None = Field(None, description="Most recent evaluation")
    total_runs: int = Field(..., description="Total number of evaluation runs")
    metric_trends: list[MetricTrend] = Field(..., description="Trends per metric")
    best_run: EvaluationRun | None = Field(None, description="Best performing run")
    configuration_impact: dict | None = Field(
        None, description="Analysis of how config changes affected metrics"
    )


# ============================================================================
# Baseline Management
# ============================================================================


class GoldenBaseline(BaseModel):
    """Golden baseline configuration and thresholds.

    Represents the target performance to beat. New evaluation runs
    are compared against this baseline for pass/fail determination.
    """

    run_id: str = Field(..., description="ID of the baseline evaluation run")
    set_at: datetime = Field(..., description="When the baseline was set")
    set_by: str | None = Field(None, description="Who set the baseline")

    # Target thresholds (from baseline run's scores)
    target_metrics: dict[str, float] = Field(
        ..., description="Metric thresholds to beat (metric_name -> threshold)"
    )

    # Reference configuration
    config_snapshot: ConfigSnapshot = Field(..., description="Configuration of the baseline run")

    # Optional performance targets
    target_latency_p95_ms: float | None = Field(
        None, description="Target P95 latency to beat (lower is better)"
    )
    target_cost_per_query_usd: float | None = Field(
        None, description="Target cost per query to beat (lower is better)"
    )


# ============================================================================
# Comparison
# ============================================================================


class ComparisonResult(BaseModel):
    """Result of comparing two evaluation runs side-by-side.

    Provides detailed analysis of differences between runs
    to help identify which configuration performs better.
    """

    run_a_id: str = Field(..., description="ID of first run")
    run_b_id: str = Field(..., description="ID of second run")

    # Configuration comparison
    run_a_config: ConfigSnapshot | None = Field(None, description="Config of run A")
    run_b_config: ConfigSnapshot | None = Field(None, description="Config of run B")

    # Metric deltas (positive = run A is better)
    metric_deltas: dict[str, float] = Field(
        ..., description="Score delta per metric (positive = A better)"
    )

    # Latency comparison
    latency_delta_ms: float | None = Field(
        None, description="Latency delta in ms (positive = A faster)"
    )
    latency_improvement_pct: float | None = Field(
        None, description="Latency improvement percentage"
    )

    # Cost comparison
    cost_delta_usd: float | None = Field(
        None, description="Cost delta in USD (positive = A cheaper)"
    )
    cost_improvement_pct: float | None = Field(
        None, description="Cost improvement percentage"
    )

    # Winner determination
    winner: Literal["run_a", "run_b", "tie"] = Field(
        ..., description="Which run is better overall"
    )
    winner_reason: str = Field(..., description="Explanation for winner determination")


# ============================================================================
# Recommendation
# ============================================================================


class Recommendation(BaseModel):
    """Configuration recommendation based on historical analysis.

    Suggests optimal configuration based on user preferences
    for accuracy, speed, and cost tradeoffs.
    """

    recommended_config: ConfigSnapshot = Field(..., description="Recommended configuration")
    source_run_id: str = Field(..., description="ID of the run this recommendation is based on")

    reasoning: str = Field(..., description="Human-readable explanation for recommendation")

    # Normalized scores (0-1)
    accuracy_score: float = Field(..., description="Accuracy score (0-1)")
    speed_score: float = Field(..., description="Speed score (0-1, higher = faster)")
    cost_score: float = Field(..., description="Cost efficiency score (0-1, higher = cheaper)")

    # Composite score
    composite_score: float = Field(..., description="Weighted composite score")

    # Weights used for this recommendation
    weights: dict[str, float] = Field(..., description="Weights used (accuracy, speed, cost)")

    # Alternative options
    alternatives: list[dict] = Field(
        default_factory=list, description="Alternative configurations with their scores"
    )


# ============================================================================
# Trend Annotation (for charts)
# ============================================================================


class TrendAnnotation(BaseModel):
    """Annotation on a trend chart point.

    Marks significant events like configuration changes,
    baseline updates, or manual notes.
    """

    timestamp: datetime = Field(..., description="When the event occurred")
    run_id: str = Field(..., description="Associated evaluation run ID")
    annotation_type: Literal[
        "config_change", "baseline_set", "regression", "improvement", "note"
    ] = Field(..., description="Type of annotation")
    title: str = Field(..., description="Short title for the annotation")
    description: str | None = Field(None, description="Detailed description")
    config_diff: dict | None = Field(None, description="What changed from previous run")
