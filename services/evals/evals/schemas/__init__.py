"""Evaluation schemas for type-safe data contracts."""

from evals.schemas.dataset import (
    EvalQuestion,
    GoldPassage,
    EvalDataset,
    QueryType,
    Difficulty,
)
from evals.schemas.response import (
    EvalResponse,
    Citation,
    RetrievedChunk,
    TokenUsage,
    QueryMetrics,
)
from evals.schemas.results import (
    MetricResult,
    MetricGroup,
    Scorecard,
    WeightedScore,
    ParetoPoint,
    ConfigSnapshot,
    EvalRun,
)

__all__ = [
    # Dataset schemas
    "EvalQuestion",
    "GoldPassage",
    "EvalDataset",
    "QueryType",
    "Difficulty",
    # Response schemas
    "EvalResponse",
    "Citation",
    "RetrievedChunk",
    "TokenUsage",
    "QueryMetrics",
    # Result schemas
    "MetricResult",
    "MetricGroup",
    "Scorecard",
    "WeightedScore",
    "ParetoPoint",
    "ConfigSnapshot",
    "EvalRun",
]
