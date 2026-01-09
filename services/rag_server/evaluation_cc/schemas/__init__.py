"""Evaluation schemas for type-safe data contracts."""

from evaluation_cc.schemas.dataset import (
    EvalQuestion,
    GoldPassage,
    EvalDataset,
    QueryType,
    Difficulty,
)
from evaluation_cc.schemas.response import (
    EvalResponse,
    Citation,
    RetrievedChunk,
    TokenUsage,
    QueryMetrics,
)
from evaluation_cc.schemas.results import (
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
