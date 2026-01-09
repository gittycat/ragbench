"""Citation quality metrics.

Measures how well the RAG system cites sources for its claims.
"""

from typing import Any

from evaluation_cc.metrics.base import BaseMetric
from evaluation_cc.schemas import (
    EvalQuestion,
    EvalResponse,
    MetricResult,
    MetricGroup,
)


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
        # Get cited chunk IDs
        cited_chunk_ids = response.cited_chunk_ids

        if not cited_chunk_ids:
            # No citations - could be 0 or N/A depending on interpretation
            return MetricResult(
                name=self.name,
                value=0.0,
                group=self.group,
                sample_size=1,
                details={"note": "No citations in answer"},
            )

        # Get gold chunk IDs
        gold_chunk_ids = {p.chunk_id for p in question.gold_passages}

        if not gold_chunk_ids:
            # No gold data - can't evaluate precision
            return MetricResult(
                name=self.name,
                value=1.0,  # Assume citations are relevant if no gold
                group=self.group,
                sample_size=1,
                details={"note": "No gold passages defined"},
            )

        # Calculate precision
        hits = len(cited_chunk_ids & gold_chunk_ids)
        precision = hits / len(cited_chunk_ids)

        return MetricResult(
            name=self.name,
            value=precision,
            group=self.group,
            sample_size=1,
            details={
                "hits": hits,
                "cited_count": len(cited_chunk_ids),
                "gold_count": len(gold_chunk_ids),
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
        # Get gold chunk IDs
        gold_chunk_ids = {p.chunk_id for p in question.gold_passages}

        if not gold_chunk_ids:
            return MetricResult(
                name=self.name,
                value=1.0,  # No gold = perfect recall
                group=self.group,
                sample_size=1,
                details={"note": "No gold passages defined"},
            )

        # Get cited chunk IDs
        cited_chunk_ids = response.cited_chunk_ids

        # Calculate recall
        hits = len(cited_chunk_ids & gold_chunk_ids)
        recall = hits / len(gold_chunk_ids)

        return MetricResult(
            name=self.name,
            value=recall,
            group=self.group,
            sample_size=1,
            details={
                "hits": hits,
                "cited_count": len(cited_chunk_ids),
                "gold_count": len(gold_chunk_ids),
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
        # Get gold passage information
        gold_passages = {
            (p.doc_id, p.chunk_id) for p in question.gold_passages
        }
        gold_doc_ids = {p.doc_id for p in question.gold_passages}

        if not gold_passages:
            return MetricResult(
                name=self.name,
                value=1.0,
                group=self.group,
                sample_size=1,
                details={"note": "No gold passages defined"},
            )

        # Get citations
        citations = response.citations
        if not citations:
            return MetricResult(
                name=self.name,
                value=0.0,
                group=self.group,
                sample_size=1,
                details={"note": "No citations in answer"},
            )

        # Evaluate each citation
        doc_correct = 0
        section_correct = 0

        for citation in citations:
            # Check document-level accuracy
            if citation.doc_id and citation.doc_id in gold_doc_ids:
                doc_correct += 1

                # Check section-level accuracy
                if citation.chunk_id:
                    if (citation.doc_id, citation.chunk_id) in gold_passages:
                        section_correct += 1

        # Calculate accuracy
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
