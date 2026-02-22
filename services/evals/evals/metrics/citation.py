"""Citation quality metrics.

Measures how well the RAG system cites sources for its claims.
"""

from typing import Any

from evals.metrics.base import BaseMetric
from evals.metrics.text_match import _token_overlap
from evals.schemas import (
    EvalQuestion,
    EvalResponse,
    MetricResult,
    MetricGroup,
)


def _chunk_by_rank(response: EvalResponse) -> dict[int, Any]:
    """Build lookup from 1-based rank to RetrievedChunk."""
    return {c.rank: c for c in response.retrieved_chunks if c.rank is not None}


class CitationPrecision(BaseMetric):
    """Citation precision measures the fraction of citations that are relevant.

    Citation Precision = |Cited ∩ Relevant| / |Cited|

    Higher is better. 1.0 means all citations are relevant.
    """

    @property
    def name(self) -> str:
        return "citation_precision"

    @property
    def group(self) -> MetricGroup:
        return MetricGroup.CITATION

    @property
    def description(self) -> str:
        return "Fraction of citations that point to relevant passages"

    def compute(
        self,
        question: EvalQuestion,
        response: EvalResponse,
        **kwargs: Any,
    ) -> MetricResult:
        citations = response.citations
        if not citations:
            return MetricResult(
                name=self.name,
                value=0.0,
                group=self.group,
                sample_size=1,
                details={"note": "No citations in answer"},
            )

        if not question.gold_passages:
            return MetricResult(
                name=self.name,
                value=1.0,
                group=self.group,
                sample_size=1,
                details={"note": "No gold passages defined"},
            )

        gold_chunk_ids = {p.chunk_id for p in question.gold_passages}
        chunk_by_rank = _chunk_by_rank(response)

        hits = 0
        for citation in citations:
            # Exact chunk_id match
            if citation.chunk_id and citation.chunk_id in gold_chunk_ids:
                hits += 1
                continue
            # Look up the source chunk by rank and do text overlap
            retrieved = chunk_by_rank.get(citation.source_index)
            if retrieved and retrieved.text:
                for gold in question.gold_passages:
                    if gold.text and _token_overlap(retrieved.text, gold.text) >= 0.3:
                        hits += 1
                        break

        precision = hits / len(citations)

        return MetricResult(
            name=self.name,
            value=precision,
            group=self.group,
            sample_size=1,
            details={
                "hits": hits,
                "cited_count": len(citations),
                "gold_count": len(question.gold_passages),
            },
        )


class CitationRecall(BaseMetric):
    """Citation recall measures the fraction of relevant passages that are cited.

    Citation Recall = |Cited ∩ Relevant| / |Relevant|

    Higher is better. 1.0 means all relevant passages are cited.
    """

    @property
    def name(self) -> str:
        return "citation_recall"

    @property
    def group(self) -> MetricGroup:
        return MetricGroup.CITATION

    @property
    def description(self) -> str:
        return "Fraction of relevant passages that are cited"

    def compute(
        self,
        question: EvalQuestion,
        response: EvalResponse,
        **kwargs: Any,
    ) -> MetricResult:
        if not question.gold_passages:
            return MetricResult(
                name=self.name,
                value=1.0,
                group=self.group,
                sample_size=1,
                details={"note": "No gold passages defined"},
            )

        citations = response.citations
        gold_chunk_ids = {p.chunk_id for p in question.gold_passages}
        chunk_by_rank = _chunk_by_rank(response)

        # Collect texts of cited chunks (exact id + text fallback)
        cited_texts: list[str] = []
        cited_exact_ids: set[str] = set()
        for citation in citations:
            if citation.chunk_id:
                cited_exact_ids.add(citation.chunk_id)
            retrieved = chunk_by_rank.get(citation.source_index)
            if retrieved and retrieved.text:
                cited_texts.append(retrieved.text)

        hits = 0
        for gold in question.gold_passages:
            if gold.chunk_id in cited_exact_ids:
                hits += 1
                continue
            if gold.text:
                for cited_text in cited_texts:
                    if _token_overlap(cited_text, gold.text) >= 0.3:
                        hits += 1
                        break

        recall = hits / len(question.gold_passages)

        return MetricResult(
            name=self.name,
            value=recall,
            group=self.group,
            sample_size=1,
            details={
                "hits": hits,
                "cited_count": len(citations),
                "gold_count": len(question.gold_passages),
            },
        )


class SectionAccuracy(BaseMetric):
    """Section accuracy measures accuracy at the document+section level.

    Checks if citations point to the correct document AND the correct
    section/chunk within that document.

    Higher is better. 1.0 means perfect section-level accuracy.
    """

    @property
    def name(self) -> str:
        return "section_accuracy"

    @property
    def group(self) -> MetricGroup:
        return MetricGroup.CITATION

    @property
    def description(self) -> str:
        return "Accuracy of citations at document+section level"

    def compute(
        self,
        question: EvalQuestion,
        response: EvalResponse,
        **kwargs: Any,
    ) -> MetricResult:
        gold_passages = {(p.doc_id, p.chunk_id) for p in question.gold_passages}
        gold_doc_ids = {p.doc_id for p in question.gold_passages}

        if not gold_passages:
            return MetricResult(
                name=self.name,
                value=1.0,
                group=self.group,
                sample_size=1,
                details={"note": "No gold passages defined"},
            )

        citations = response.citations
        if not citations:
            return MetricResult(
                name=self.name,
                value=0.0,
                group=self.group,
                sample_size=1,
                details={"note": "No citations in answer"},
            )

        chunk_by_rank = _chunk_by_rank(response)
        doc_correct = 0
        section_correct = 0

        for citation in citations:
            # Exact (doc_id, chunk_id) match
            if citation.doc_id and citation.chunk_id:
                if citation.doc_id in gold_doc_ids:
                    doc_correct += 1
                    if (citation.doc_id, citation.chunk_id) in gold_passages:
                        section_correct += 1
                    continue

            # Text-based fallback via retrieved chunk
            retrieved = chunk_by_rank.get(citation.source_index)
            if retrieved and retrieved.text:
                for gold in question.gold_passages:
                    if gold.text and _token_overlap(retrieved.text, gold.text) >= 0.3:
                        doc_correct += 1
                        section_correct += 1  # text match implies section match
                        break

        total_citations = len(citations)
        doc_accuracy = doc_correct / total_citations
        section_accuracy = section_correct / total_citations

        return MetricResult(
            name=self.name,
            value=section_accuracy,
            group=self.group,
            sample_size=1,
            details={
                "doc_accuracy": doc_accuracy,
                "section_accuracy": section_accuracy,
                "doc_correct": doc_correct,
                "section_correct": section_correct,
                "total_citations": total_citations,
            },
        )
