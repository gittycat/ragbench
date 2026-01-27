"""Results schemas for metrics, scorecards, and evaluation runs."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MetricGroup(str, Enum):
    """Groups of related metrics."""

    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    CITATION = "citation"
    ABSTENTION = "abstention"
    PERFORMANCE = "performance"


@dataclass
class MetricResult:
    """Result of a single metric computation.

    Attributes:
        name: Metric name (e.g., "recall_at_5", "faithfulness")
        value: Computed metric value (typically 0-1 for ratios)
        group: Which metric group this belongs to
        details: Additional metric-specific details
        sample_size: Number of samples used to compute this metric
    """

    name: str
    value: float
    group: MetricGroup
    details: dict[str, Any] = field(default_factory=dict)
    sample_size: int = 0

    def __post_init__(self):
        if not self.name:
            raise ValueError("name cannot be empty")


@dataclass
class Scorecard:
    """Complete scorecard with all metrics for an evaluation run.

    Attributes:
        metrics: All computed metrics
        by_group: Metrics organized by group
        by_domain: Metrics broken down by domain (if applicable)
        by_query_type: Metrics broken down by query type (if applicable)
    """

    metrics: list[MetricResult] = field(default_factory=list)
    by_group: dict[MetricGroup, list[MetricResult]] = field(default_factory=dict)
    by_domain: dict[str, list[MetricResult]] = field(default_factory=dict)
    by_query_type: dict[str, list[MetricResult]] = field(default_factory=dict)

    def get_metric(self, name: str) -> MetricResult | None:
        """Get a specific metric by name."""
        for m in self.metrics:
            if m.name == name:
                return m
        return None

    def get_group_average(self, group: MetricGroup) -> float | None:
        """Get average value for all metrics in a group."""
        group_metrics = self.by_group.get(group, [])
        if not group_metrics:
            return None
        return sum(m.value for m in group_metrics) / len(group_metrics)

    def add_metric(self, metric: MetricResult) -> None:
        """Add a metric to the scorecard."""
        self.metrics.append(metric)
        if metric.group not in self.by_group:
            self.by_group[metric.group] = []
        self.by_group[metric.group].append(metric)


@dataclass
class WeightedScore:
    """A weighted overall score combining multiple metrics.

    Attributes:
        score: The final weighted score (0-1)
        weights: The weights used for each metric/group
        contributions: How much each metric contributed to the final score
        objectives: The objective names and their importance
    """

    score: float
    weights: dict[str, float] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)
    objectives: dict[str, float] = field(default_factory=dict)

    def explain(self) -> str:
        """Generate a human-readable explanation of the score."""
        lines = [f"Weighted Score: {self.score:.3f}"]
        lines.append("Contributions:")
        for name, contrib in sorted(
            self.contributions.items(), key=lambda x: -x[1]
        ):
            weight = self.weights.get(name, 0)
            lines.append(f"  {name}: {contrib:.3f} (weight: {weight:.2f})")
        return "\n".join(lines)


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier for multi-objective optimization.

    Attributes:
        run_id: ID of the evaluation run
        config_name: Human-readable config description
        objectives: Objective name -> value mapping
        is_dominated: Whether this point is dominated by another
        dominates: List of run IDs this point dominates
    """

    run_id: str
    config_name: str
    objectives: dict[str, float] = field(default_factory=dict)
    is_dominated: bool = False
    dominates: list[str] = field(default_factory=list)


@dataclass
class ConfigSnapshot:
    """Snapshot of the RAG configuration used for an evaluation run.

    Attributes:
        llm_model: LLM model name/identifier
        llm_provider: LLM provider (ollama, anthropic, etc.)
        embedding_model: Embedding model name
        reranker_model: Reranker model (if enabled)
        retrieval_top_k: Number of chunks to retrieve
        hybrid_search_enabled: Whether hybrid search was used
        contextual_retrieval_enabled: Whether contextual retrieval was used
        additional: Any additional config parameters
    """

    llm_model: str
    llm_provider: str
    embedding_model: str
    reranker_model: str | None = None
    retrieval_top_k: int = 10
    hybrid_search_enabled: bool = True
    contextual_retrieval_enabled: bool = False
    additional: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRun:
    """A complete evaluation run with all results.

    Attributes:
        id: Unique run identifier
        name: Human-readable run name
        created_at: When the run was created
        completed_at: When the run completed (None if still running)
        config: Configuration snapshot
        datasets: Names of datasets used
        scorecard: Complete metric scorecard
        weighted_score: Overall weighted score
        question_count: Total questions evaluated
        error_count: Number of questions that errored
        metadata: Additional run metadata
    """

    id: str
    name: str
    created_at: datetime
    config: ConfigSnapshot
    datasets: list[str] = field(default_factory=list)
    scorecard: Scorecard | None = None
    weighted_score: WeightedScore | None = None
    completed_at: datetime | None = None
    question_count: int = 0
    error_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if the run has completed."""
        return self.completed_at is not None

    @property
    def success_rate(self) -> float:
        """Calculate the success rate (non-error questions)."""
        if self.question_count == 0:
            return 0.0
        return (self.question_count - self.error_count) / self.question_count

    @property
    def duration_seconds(self) -> float | None:
        """Calculate run duration in seconds."""
        if not self.completed_at:
            return None
        return (self.completed_at - self.created_at).total_seconds()
