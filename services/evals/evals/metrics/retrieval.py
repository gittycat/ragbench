"""Retrieval quality metrics.

Measures how well the RAG system retrieves relevant documents/chunks.
"""

import math
from typing import Any

from evals.metrics.base import BaseMetric
from evals.metrics.text_match import match_retrieved_to_gold, _token_overlap
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
        if not question.gold_passages:
            return MetricResult(
                name=self.name,
                value=0.0,
                group=self.group,
                sample_size=1,
                details={"note": "No gold passages defined"},
            )

        retrieved_chunks = response.retrieved_chunks[: self.k]
        matched = match_retrieved_to_gold(retrieved_chunks, question.gold_passages)
        recall = len(matched) / len(question.gold_passages)

        return MetricResult(
            name=self.name,
            value=recall,
            group=self.group,
            sample_size=1,
            details={
                "hits": len(matched),
                "gold_count": len(question.gold_passages),
                "retrieved_count": len(retrieved_chunks),
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
        if not question.gold_passages:
            return MetricResult(
                name=self.name,
                value=0.0,
                group=self.group,
                sample_size=1,
                details={"note": "No gold passages defined"},
            )

        retrieved_chunks = response.retrieved_chunks[: self.k]
        if not retrieved_chunks:
            return MetricResult(
                name=self.name,
                value=0.0,
                group=self.group,
                sample_size=1,
                details={"note": "No chunks retrieved"},
            )

        matched = match_retrieved_to_gold(retrieved_chunks, question.gold_passages)
        precision = len(matched) / min(self.k, len(retrieved_chunks))

        return MetricResult(
            name=self.name,
            value=precision,
            group=self.group,
            sample_size=1,
            details={
                "hits": len(matched),
                "gold_count": len(question.gold_passages),
                "retrieved_count": len(retrieved_chunks),
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
        if not question.gold_passages:
            return MetricResult(
                name=self.name,
                value=0.0,
                group=self.group,
                sample_size=1,
                details={"note": "No gold passages defined"},
            )

        gold_chunk_ids = {p.chunk_id for p in question.gold_passages}

        for i, chunk in enumerate(response.retrieved_chunks):
            rank = i + 1
            # Exact ID match
            if chunk.chunk_id in gold_chunk_ids:
                return MetricResult(
                    name=self.name,
                    value=1.0 / rank,
                    group=self.group,
                    sample_size=1,
                    details={"first_relevant_rank": rank},
                )
            # Text overlap fallback
            if chunk.text:
                for gold in question.gold_passages:
                    if gold.text and _token_overlap(chunk.text, gold.text) >= 0.3:
                        return MetricResult(
                            name=self.name,
                            value=1.0 / rank,
                            group=self.group,
                            sample_size=1,
                            details={"first_relevant_rank": rank},
                        )

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
        if not question.gold_passages:
            return MetricResult(
                name=self.name,
                value=0.0,
                group=self.group,
                sample_size=1,
                details={"note": "No gold passages defined"},
            )

        gold_id_to_score = {p.chunk_id: p.relevance_score for p in question.gold_passages}

        # Build relevance vector using exact ID match then text overlap
        retrieved_relevances = []
        for chunk in response.retrieved_chunks[: self.k]:
            rel = gold_id_to_score.get(chunk.chunk_id, 0.0)
            if rel == 0.0 and chunk.text:
                for gold in question.gold_passages:
                    if gold.text and _token_overlap(chunk.text, gold.text) >= 0.3:
                        rel = gold.relevance_score
                        break
            retrieved_relevances.append(rel)

        # Compute DCG
        dcg = self._compute_dcg(retrieved_relevances)

        # Compute ideal DCG (sorted relevances)
        ideal_relevances = sorted(gold_id_to_score.values(), reverse=True)[: self.k]
        ideal_relevances.extend([0.0] * (self.k - len(ideal_relevances)))
        idcg = self._compute_dcg(ideal_relevances)

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
            dcg += rel / math.log2(i + 2)
        return dcg
