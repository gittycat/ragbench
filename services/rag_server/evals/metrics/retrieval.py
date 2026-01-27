"""Retrieval quality metrics.

Measures how well the RAG system retrieves relevant documents/chunks.
"""

import math
from typing import Any

from evals.metrics.base import BaseMetric
from evals.schemas import (
    EvalQuestion,
    EvalResponse,
    MetricResult,
    MetricGroup,
)


class RecallAtK(BaseMetric):
    """Recall@K measures the fraction of relevant documents retrieved in top K.

    Recall@K = |Retrieved ∩ Relevant| / |Relevant|

    Higher is better. 1.0 means all relevant documents were retrieved.
    """

    def __init__(self, k: int = 5):
        self.k = k

    @property
    def name(self) -> str:
        return f"recall_at_{self.k}"

    @property
    def group(self) -> MetricGroup:
        return MetricGroup.RETRIEVAL

    @property
    def description(self) -> str:
        return f"Fraction of relevant documents retrieved in top {self.k}"

    def compute(
        self,
        question: EvalQuestion,
        response: EvalResponse,
        **kwargs: Any,
    ) -> MetricResult:
        # Get gold chunk IDs
        gold_chunk_ids = {p.chunk_id for p in question.gold_passages}

        if not gold_chunk_ids:
            return MetricResult(
                name=self.name,
                value=1.0,  # No gold = perfect recall by default
                group=self.group,
                sample_size=1,
                details={"note": "No gold passages defined"},
            )

        # Get retrieved chunk IDs (top K)
        retrieved_chunks = response.retrieved_chunks[: self.k]
        retrieved_chunk_ids = {c.chunk_id for c in retrieved_chunks}

        # Calculate recall
        hits = len(gold_chunk_ids & retrieved_chunk_ids)
        recall = hits / len(gold_chunk_ids)

        return MetricResult(
            name=self.name,
            value=recall,
            group=self.group,
            sample_size=1,
            details={
                "hits": hits,
                "gold_count": len(gold_chunk_ids),
                "retrieved_count": len(retrieved_chunk_ids),
            },
        )


class PrecisionAtK(BaseMetric):
    """Precision@K measures the fraction of retrieved documents that are relevant.

    Precision@K = |Retrieved ∩ Relevant| / K

    Higher is better. 1.0 means all retrieved documents are relevant.
    """

    def __init__(self, k: int = 5):
        self.k = k

    @property
    def name(self) -> str:
        return f"precision_at_{self.k}"

    @property
    def group(self) -> MetricGroup:
        return MetricGroup.RETRIEVAL

    @property
    def description(self) -> str:
        return f"Fraction of top {self.k} retrieved documents that are relevant"

    def compute(
        self,
        question: EvalQuestion,
        response: EvalResponse,
        **kwargs: Any,
    ) -> MetricResult:
        # Get gold chunk IDs
        gold_chunk_ids = {p.chunk_id for p in question.gold_passages}

        # Get retrieved chunk IDs (top K)
        retrieved_chunks = response.retrieved_chunks[: self.k]
        retrieved_chunk_ids = {c.chunk_id for c in retrieved_chunks}

        if not retrieved_chunk_ids:
            return MetricResult(
                name=self.name,
                value=0.0,
                group=self.group,
                sample_size=1,
                details={"note": "No chunks retrieved"},
            )

        # Calculate precision
        hits = len(gold_chunk_ids & retrieved_chunk_ids)
        precision = hits / min(self.k, len(retrieved_chunk_ids))

        return MetricResult(
            name=self.name,
            value=precision,
            group=self.group,
            sample_size=1,
            details={
                "hits": hits,
                "gold_count": len(gold_chunk_ids),
                "retrieved_count": len(retrieved_chunk_ids),
            },
        )


class MRR(BaseMetric):
    """Mean Reciprocal Rank measures rank of the first relevant result.

    MRR = 1 / rank_of_first_relevant

    Higher is better. 1.0 means the first result is relevant.
    """

    @property
    def name(self) -> str:
        return "mrr"

    @property
    def group(self) -> MetricGroup:
        return MetricGroup.RETRIEVAL

    @property
    def description(self) -> str:
        return "Reciprocal of the rank of the first relevant result"

    def compute(
        self,
        question: EvalQuestion,
        response: EvalResponse,
        **kwargs: Any,
    ) -> MetricResult:
        # Get gold chunk IDs
        gold_chunk_ids = {p.chunk_id for p in question.gold_passages}

        if not gold_chunk_ids:
            return MetricResult(
                name=self.name,
                value=1.0,
                group=self.group,
                sample_size=1,
                details={"note": "No gold passages defined"},
            )

        # Find rank of first relevant result
        for rank, chunk in enumerate(response.retrieved_chunks, start=1):
            if chunk.chunk_id in gold_chunk_ids:
                return MetricResult(
                    name=self.name,
                    value=1.0 / rank,
                    group=self.group,
                    sample_size=1,
                    details={"first_relevant_rank": rank},
                )

        # No relevant result found
        return MetricResult(
            name=self.name,
            value=0.0,
            group=self.group,
            sample_size=1,
            details={"first_relevant_rank": None},
        )


class NDCG(BaseMetric):
    """Normalized Discounted Cumulative Gain measures ranking quality.

    NDCG accounts for position of relevant results (earlier is better)
    and can handle graded relevance.

    Higher is better. 1.0 means perfect ranking.
    """

    def __init__(self, k: int = 10):
        self.k = k

    @property
    def name(self) -> str:
        return f"ndcg_at_{self.k}"

    @property
    def group(self) -> MetricGroup:
        return MetricGroup.RETRIEVAL

    @property
    def description(self) -> str:
        return f"Normalized discounted cumulative gain at {self.k}"

    def compute(
        self,
        question: EvalQuestion,
        response: EvalResponse,
        **kwargs: Any,
    ) -> MetricResult:
        # Build relevance map: chunk_id -> relevance score
        relevance_map = {
            p.chunk_id: p.relevance_score for p in question.gold_passages
        }

        if not relevance_map:
            return MetricResult(
                name=self.name,
                value=1.0,
                group=self.group,
                sample_size=1,
                details={"note": "No gold passages defined"},
            )

        # Get relevance scores for retrieved chunks
        retrieved_relevances = []
        for chunk in response.retrieved_chunks[: self.k]:
            rel = relevance_map.get(chunk.chunk_id, 0.0)
            retrieved_relevances.append(rel)

        # Compute DCG
        dcg = self._compute_dcg(retrieved_relevances)

        # Compute ideal DCG (sorted relevances)
        ideal_relevances = sorted(relevance_map.values(), reverse=True)[: self.k]
        # Pad with zeros if needed
        ideal_relevances.extend([0.0] * (self.k - len(ideal_relevances)))
        idcg = self._compute_dcg(ideal_relevances)

        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=ndcg,
            group=self.group,
            sample_size=1,
            details={
                "dcg": dcg,
                "idcg": idcg,
                "retrieved_relevances": retrieved_relevances,
            },
        )

    def _compute_dcg(self, relevances: list[float]) -> float:
        """Compute Discounted Cumulative Gain."""
        dcg = 0.0
        for i, rel in enumerate(relevances):
            # DCG formula: rel_i / log2(i + 2)
            # Using i + 2 because positions are 1-indexed
            dcg += rel / math.log2(i + 2)
        return dcg
