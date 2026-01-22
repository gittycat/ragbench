"""Pydantic models for the Evaluation API (/metrics/eval/).

This module defines request/response models for:
- Discovery: metric groups and datasets
- Execution: evaluation runs with progress tracking
- Results: run status and results

## Schema Architecture

Three schema layers exist in this codebase:

1. **schemas/eval.py** (this file) - Pydantic models for the new Eval API
   - Used for: /metrics/eval/* endpoint request/response validation
   - Optimized for: API contracts, OpenAPI docs, frontend integration

2. **schemas/metrics.py** - Pydantic models for existing metrics endpoints
   - Used for: /metrics/system, /metrics/models, /metrics/retrieval
   - Contains: ConfigSnapshot, LatencyMetrics, CostMetrics (reusable)

3. **evaluation_cc/schemas/** - Dataclasses for internal evaluation
   - Used for: EvaluationRunner, metric computation, result storage
   - Contains: EvalQuestion, EvalResponse, EvalRun, Scorecard

Reuses models from schemas.metrics where applicable to avoid duplication.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# Reuse existing models from metrics.py
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
