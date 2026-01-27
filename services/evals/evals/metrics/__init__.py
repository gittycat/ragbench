"""Evaluation metrics for RAG systems."""

from evals.metrics.base import BaseMetric
from evals.metrics.retrieval import (
    RecallAtK,
    PrecisionAtK,
    MRR,
    NDCG,
)
from evals.metrics.generation import (
    Faithfulness,
    AnswerCorrectness,
    AnswerRelevancy,
)
from evals.metrics.citation import (
    CitationPrecision,
    CitationRecall,
    SectionAccuracy,
)
from evals.metrics.abstention import (
    UnanswerableAccuracy,
    FalsePositiveRate,
    FalseNegativeRate,
)
from evals.metrics.performance import (
    LatencyP50,
    LatencyP95,
    CostPerQuery,
)
from evals.schemas.results import MetricGroup

# Metric groups for easy selection
METRIC_GROUPS = {
    MetricGroup.RETRIEVAL: [RecallAtK, PrecisionAtK, MRR, NDCG],
    MetricGroup.GENERATION: [Faithfulness, AnswerCorrectness, AnswerRelevancy],
    MetricGroup.CITATION: [CitationPrecision, CitationRecall, SectionAccuracy],
    MetricGroup.ABSTENTION: [UnanswerableAccuracy, FalsePositiveRate, FalseNegativeRate],
    MetricGroup.PERFORMANCE: [LatencyP50, LatencyP95, CostPerQuery],
}

__all__ = [
    # Base
    "BaseMetric",
    # Retrieval
    "RecallAtK",
    "PrecisionAtK",
    "MRR",
    "NDCG",
    # Generation
    "Faithfulness",
    "AnswerCorrectness",
    "AnswerRelevancy",
    # Citation
    "CitationPrecision",
    "CitationRecall",
    "SectionAccuracy",
    # Abstention
    "UnanswerableAccuracy",
    "FalsePositiveRate",
    "FalseNegativeRate",
    # Performance
    "LatencyP50",
    "LatencyP95",
    "CostPerQuery",
    # Groups
    "METRIC_GROUPS",
]
