"""Pydantic request/response models for the eval API."""

from datetime import datetime

from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────


class TriggerRunRequest(BaseModel):
    name: str | None = None
    tier: str = "generation"
    datasets: list[str] = Field(default_factory=lambda: ["ragbench"])
    samples: int = 100
    seed: int | None = 42
    judge_enabled: bool = True


# ── Responses ─────────────────────────────────────────────────────────────────


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str = "queued"
    created_at: datetime


class ProgressInfo(BaseModel):
    current_question: int = 0
    total_questions: int = 0
    current_dataset: str = ""
    phase: str = "initializing"
    elapsed_seconds: float = 0.0


class ActiveJobResponse(BaseModel):
    job_id: str
    status: str
    progress: ProgressInfo


class DashboardMetrics(BaseModel):
    retrieval_relevance: float | None = None
    faithfulness: float | None = None
    answer_completeness: float | None = None
    answer_relevance: float | None = None
    latency_p50_seconds: float | None = None
    latency_p95_seconds: float | None = None


class RunSummary(BaseModel):
    id: str
    name: str
    created_at: datetime
    completed_at: datetime | None = None
    tier: str = ""
    datasets: list[str] = Field(default_factory=list)
    question_count: int = 0
    error_count: int = 0
    duration_seconds: float | None = None
    weighted_score: float | None = None
    dashboard_metrics: DashboardMetrics | None = None


class RunListResponse(BaseModel):
    runs: list[RunSummary]
    total: int


class RunDetailResponse(BaseModel):
    """Full run detail — includes raw scorecard data."""

    id: str
    name: str
    created_at: datetime
    completed_at: datetime | None = None
    tier: str = ""
    datasets: list[str] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)
    scorecard: dict | None = None
    weighted_score: dict | None = None
    question_count: int = 0
    error_count: int = 0
    duration_seconds: float | None = None
    metadata: dict = Field(default_factory=dict)
    dashboard_metrics: DashboardMetrics | None = None


class CompareRunsResponse(BaseModel):
    runs: list[RunDetailResponse]
    deltas: dict[str, float | None] = Field(default_factory=dict)


class DatasetInfo(BaseModel):
    name: str
    description: str = ""
    source_url: str = ""
    supported_tiers: list[str] = Field(default_factory=list)


class DashboardResponse(BaseModel):
    latest_run: RunSummary | None = None
    total_runs: int = 0
    active_job: ActiveJobResponse | None = None
