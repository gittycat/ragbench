"""
Tests for RAG evaluation framework.

Tests cover:
- Retrieval metrics (Recall@K, Precision@K, MRR, NDCG)
- Citation metrics (CitationPrecision, CitationRecall, SectionAccuracy)
- Citation extraction from LLM answers
- Query endpoint with include_chunks parameter

Run with: pytest tests/test_rag_eval.py --run-eval
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# RETRIEVAL METRICS TESTS
# =============================================================================


class TestRecallAtK:
    """Tests for Recall@K retrieval metric."""

    def test_perfect_recall(self):
        """All gold chunks retrieved should yield recall of 1.0."""
        from evaluation_cc.metrics.retrieval import RecallAtK
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage, RetrievedChunk

        metric = RecallAtK(k=5)

        question = EvalQuestion(
            id="q1",
            question="What is the capital of France?",
            expected_answer="Paris",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="chunk1", text="Paris is the capital"),
                GoldPassage(doc_id="doc1", chunk_id="chunk2", text="France's capital is Paris"),
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Paris",
            retrieved_chunks=[
                RetrievedChunk(doc_id="doc1", chunk_id="chunk1", text="Paris is the capital", rank=1),
                RetrievedChunk(doc_id="doc1", chunk_id="chunk2", text="France's capital is Paris", rank=2),
                RetrievedChunk(doc_id="doc2", chunk_id="chunk3", text="London is big", rank=3),
            ],
        )

        result = metric.compute(question, response)

        assert result.name == "recall_at_5"
        assert result.value == 1.0
        assert result.details["hits"] == 2

    def test_partial_recall(self):
        """Only some gold chunks retrieved should yield partial recall."""
        from evaluation_cc.metrics.retrieval import RecallAtK
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage, RetrievedChunk

        metric = RecallAtK(k=5)

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="chunk1", text="chunk 1"),
                GoldPassage(doc_id="doc1", chunk_id="chunk2", text="chunk 2"),
                GoldPassage(doc_id="doc1", chunk_id="chunk3", text="chunk 3"),
                GoldPassage(doc_id="doc1", chunk_id="chunk4", text="chunk 4"),
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y",
            retrieved_chunks=[
                RetrievedChunk(doc_id="doc1", chunk_id="chunk1", text="chunk 1", rank=1),
                RetrievedChunk(doc_id="doc1", chunk_id="chunk2", text="chunk 2", rank=2),
            ],
        )

        result = metric.compute(question, response)

        assert result.value == 0.5  # 2 of 4 gold chunks retrieved
        assert result.details["hits"] == 2
        assert result.details["gold_count"] == 4

    def test_no_gold_passages(self):
        """No gold passages should default to recall of 1.0."""
        from evaluation_cc.metrics.retrieval import RecallAtK
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, RetrievedChunk

        metric = RecallAtK(k=5)

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y",
            retrieved_chunks=[
                RetrievedChunk(doc_id="doc1", chunk_id="chunk1", text="text", rank=1),
            ],
        )

        result = metric.compute(question, response)

        assert result.value == 1.0
        assert "No gold passages" in result.details.get("note", "")

    def test_k_limit_applied(self):
        """Only top K retrieved chunks should be considered."""
        from evaluation_cc.metrics.retrieval import RecallAtK
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage, RetrievedChunk

        metric = RecallAtK(k=2)

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="chunk3", text="chunk 3"),  # At rank 3, outside k=2
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y",
            retrieved_chunks=[
                RetrievedChunk(doc_id="doc1", chunk_id="chunk1", text="chunk 1", rank=1),
                RetrievedChunk(doc_id="doc1", chunk_id="chunk2", text="chunk 2", rank=2),
                RetrievedChunk(doc_id="doc1", chunk_id="chunk3", text="chunk 3", rank=3),
            ],
        )

        result = metric.compute(question, response)

        assert result.value == 0.0  # Gold chunk is at rank 3, but k=2
        assert result.details["retrieved_count"] == 2


class TestPrecisionAtK:
    """Tests for Precision@K retrieval metric."""

    def test_perfect_precision(self):
        """All retrieved chunks are relevant should yield precision of 1.0."""
        from evaluation_cc.metrics.retrieval import PrecisionAtK
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage, RetrievedChunk

        metric = PrecisionAtK(k=3)

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="chunk1", text="text1"),
                GoldPassage(doc_id="doc1", chunk_id="chunk2", text="text2"),
                GoldPassage(doc_id="doc1", chunk_id="chunk3", text="text3"),
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y",
            retrieved_chunks=[
                RetrievedChunk(doc_id="doc1", chunk_id="chunk1", text="text1", rank=1),
                RetrievedChunk(doc_id="doc1", chunk_id="chunk2", text="text2", rank=2),
                RetrievedChunk(doc_id="doc1", chunk_id="chunk3", text="text3", rank=3),
            ],
        )

        result = metric.compute(question, response)

        assert result.name == "precision_at_3"
        assert result.value == 1.0

    def test_partial_precision(self):
        """Some irrelevant chunks should yield partial precision."""
        from evaluation_cc.metrics.retrieval import PrecisionAtK
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage, RetrievedChunk

        metric = PrecisionAtK(k=4)

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="chunk1", text="text1"),
                GoldPassage(doc_id="doc1", chunk_id="chunk2", text="text2"),
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y",
            retrieved_chunks=[
                RetrievedChunk(doc_id="doc1", chunk_id="chunk1", text="text1", rank=1),
                RetrievedChunk(doc_id="doc1", chunk_id="chunk2", text="text2", rank=2),
                RetrievedChunk(doc_id="doc2", chunk_id="chunk3", text="irrelevant", rank=3),
                RetrievedChunk(doc_id="doc2", chunk_id="chunk4", text="irrelevant", rank=4),
            ],
        )

        result = metric.compute(question, response)

        assert result.value == 0.5  # 2 of 4 retrieved are relevant

    def test_no_chunks_retrieved(self):
        """No retrieved chunks should yield precision of 0.0."""
        from evaluation_cc.metrics.retrieval import PrecisionAtK
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage

        metric = PrecisionAtK(k=5)

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="chunk1", text="text1"),
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y",
            retrieved_chunks=[],
        )

        result = metric.compute(question, response)

        assert result.value == 0.0
        assert "No chunks retrieved" in result.details.get("note", "")


class TestMRR:
    """Tests for Mean Reciprocal Rank metric."""

    def test_first_result_relevant(self):
        """First result relevant should yield MRR of 1.0."""
        from evaluation_cc.metrics.retrieval import MRR
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage, RetrievedChunk

        metric = MRR()

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="chunk1", text="text1"),
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y",
            retrieved_chunks=[
                RetrievedChunk(doc_id="doc1", chunk_id="chunk1", text="text1", rank=1),
                RetrievedChunk(doc_id="doc2", chunk_id="chunk2", text="text2", rank=2),
            ],
        )

        result = metric.compute(question, response)

        assert result.name == "mrr"
        assert result.value == 1.0
        assert result.details["first_relevant_rank"] == 1

    def test_second_result_relevant(self):
        """Second result relevant should yield MRR of 0.5."""
        from evaluation_cc.metrics.retrieval import MRR
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage, RetrievedChunk

        metric = MRR()

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="chunk2", text="text2"),
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y",
            retrieved_chunks=[
                RetrievedChunk(doc_id="doc2", chunk_id="chunk1", text="irrelevant", rank=1),
                RetrievedChunk(doc_id="doc1", chunk_id="chunk2", text="text2", rank=2),
            ],
        )

        result = metric.compute(question, response)

        assert result.value == 0.5
        assert result.details["first_relevant_rank"] == 2

    def test_no_relevant_result(self):
        """No relevant results should yield MRR of 0.0."""
        from evaluation_cc.metrics.retrieval import MRR
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage, RetrievedChunk

        metric = MRR()

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="gold_chunk", text="gold text"),
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y",
            retrieved_chunks=[
                RetrievedChunk(doc_id="doc2", chunk_id="chunk1", text="irrelevant", rank=1),
                RetrievedChunk(doc_id="doc2", chunk_id="chunk2", text="also irrelevant", rank=2),
            ],
        )

        result = metric.compute(question, response)

        assert result.value == 0.0
        assert result.details["first_relevant_rank"] is None


class TestNDCG:
    """Tests for Normalized Discounted Cumulative Gain metric."""

    def test_perfect_ranking(self):
        """Perfect ranking should yield NDCG of 1.0."""
        from evaluation_cc.metrics.retrieval import NDCG
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage, RetrievedChunk

        metric = NDCG(k=3)

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="chunk1", text="text1", relevance_score=1.0),
                GoldPassage(doc_id="doc1", chunk_id="chunk2", text="text2", relevance_score=0.8),
                GoldPassage(doc_id="doc1", chunk_id="chunk3", text="text3", relevance_score=0.5),
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y",
            retrieved_chunks=[
                RetrievedChunk(doc_id="doc1", chunk_id="chunk1", text="text1", rank=1),  # rel=1.0
                RetrievedChunk(doc_id="doc1", chunk_id="chunk2", text="text2", rank=2),  # rel=0.8
                RetrievedChunk(doc_id="doc1", chunk_id="chunk3", text="text3", rank=3),  # rel=0.5
            ],
        )

        result = metric.compute(question, response)

        assert result.name == "ndcg_at_3"
        assert result.value == pytest.approx(1.0, abs=0.001)

    def test_inverted_ranking(self):
        """Inverted ranking should yield NDCG less than 1.0."""
        from evaluation_cc.metrics.retrieval import NDCG
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage, RetrievedChunk

        metric = NDCG(k=3)

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="chunk1", text="text1", relevance_score=1.0),
                GoldPassage(doc_id="doc1", chunk_id="chunk2", text="text2", relevance_score=0.5),
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y",
            retrieved_chunks=[
                RetrievedChunk(doc_id="doc1", chunk_id="chunk2", text="text2", rank=1),  # rel=0.5
                RetrievedChunk(doc_id="doc1", chunk_id="chunk1", text="text1", rank=2),  # rel=1.0
            ],
        )

        result = metric.compute(question, response)

        # NDCG should be less than 1.0 since ranking is suboptimal
        assert result.value < 1.0
        assert result.value > 0.0

    def test_no_gold_passages(self):
        """No gold passages should default to NDCG of 1.0."""
        from evaluation_cc.metrics.retrieval import NDCG
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, RetrievedChunk

        metric = NDCG(k=3)

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y",
            retrieved_chunks=[
                RetrievedChunk(doc_id="doc1", chunk_id="chunk1", text="text1", rank=1),
            ],
        )

        result = metric.compute(question, response)

        assert result.value == 1.0


# =============================================================================
# CITATION METRICS TESTS
# =============================================================================


class TestCitationPrecision:
    """Tests for citation precision metric."""

    def test_perfect_citation_precision(self):
        """All citations pointing to gold chunks should yield precision of 1.0."""
        from evaluation_cc.metrics.citation import CitationPrecision
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage, Citation

        metric = CitationPrecision()

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="chunk1", text="text1"),
                GoldPassage(doc_id="doc1", chunk_id="chunk2", text="text2"),
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y [1][2]",
            citations=[
                Citation(source_index=1, doc_id="doc1", chunk_id="chunk1"),
                Citation(source_index=2, doc_id="doc1", chunk_id="chunk2"),
            ],
        )

        result = metric.compute(question, response)

        assert result.name == "citation_precision"
        assert result.value == 1.0

    def test_partial_citation_precision(self):
        """Some citations to non-gold chunks should yield partial precision."""
        from evaluation_cc.metrics.citation import CitationPrecision
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage, Citation

        metric = CitationPrecision()

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="chunk1", text="text1"),
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y [1][2]",
            citations=[
                Citation(source_index=1, doc_id="doc1", chunk_id="chunk1"),  # Gold
                Citation(source_index=2, doc_id="doc2", chunk_id="chunk_other"),  # Not gold
            ],
        )

        result = metric.compute(question, response)

        assert result.value == 0.5

    def test_no_citations(self):
        """No citations should yield precision of 0.0."""
        from evaluation_cc.metrics.citation import CitationPrecision
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage

        metric = CitationPrecision()

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="chunk1", text="text1"),
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y",
            citations=[],
        )

        result = metric.compute(question, response)

        assert result.value == 0.0
        assert "No citations" in result.details.get("note", "")


class TestCitationRecall:
    """Tests for citation recall metric."""

    def test_perfect_citation_recall(self):
        """All gold chunks cited should yield recall of 1.0."""
        from evaluation_cc.metrics.citation import CitationRecall
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage, Citation

        metric = CitationRecall()

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="chunk1", text="text1"),
                GoldPassage(doc_id="doc1", chunk_id="chunk2", text="text2"),
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y [1][2]",
            citations=[
                Citation(source_index=1, doc_id="doc1", chunk_id="chunk1"),
                Citation(source_index=2, doc_id="doc1", chunk_id="chunk2"),
            ],
        )

        result = metric.compute(question, response)

        assert result.name == "citation_recall"
        assert result.value == 1.0

    def test_partial_citation_recall(self):
        """Only some gold chunks cited should yield partial recall."""
        from evaluation_cc.metrics.citation import CitationRecall
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage, Citation

        metric = CitationRecall()

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="chunk1", text="text1"),
                GoldPassage(doc_id="doc1", chunk_id="chunk2", text="text2"),
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y [1]",
            citations=[
                Citation(source_index=1, doc_id="doc1", chunk_id="chunk1"),
            ],
        )

        result = metric.compute(question, response)

        assert result.value == 0.5

    def test_no_gold_passages(self):
        """No gold passages should default to recall of 1.0."""
        from evaluation_cc.metrics.citation import CitationRecall
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, Citation

        metric = CitationRecall()

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y [1]",
            citations=[
                Citation(source_index=1, doc_id="doc1", chunk_id="chunk1"),
            ],
        )

        result = metric.compute(question, response)

        assert result.value == 1.0


class TestSectionAccuracy:
    """Tests for section-level citation accuracy metric."""

    def test_perfect_section_accuracy(self):
        """All citations to correct doc+section should yield accuracy of 1.0."""
        from evaluation_cc.metrics.citation import SectionAccuracy
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage, Citation

        metric = SectionAccuracy()

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="section1", text="text1"),
                GoldPassage(doc_id="doc1", chunk_id="section2", text="text2"),
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y [1][2]",
            citations=[
                Citation(source_index=1, doc_id="doc1", chunk_id="section1"),
                Citation(source_index=2, doc_id="doc1", chunk_id="section2"),
            ],
        )

        result = metric.compute(question, response)

        assert result.name == "section_accuracy"
        assert result.value == 1.0
        assert result.details["doc_accuracy"] == 1.0

    def test_doc_correct_but_section_wrong(self):
        """Citation to right doc but wrong section should show in details."""
        from evaluation_cc.metrics.citation import SectionAccuracy
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage, Citation

        metric = SectionAccuracy()

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="section1", text="text1"),
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y [1]",
            citations=[
                Citation(source_index=1, doc_id="doc1", chunk_id="wrong_section"),
            ],
        )

        result = metric.compute(question, response)

        assert result.value == 0.0  # Section wrong
        assert result.details["doc_accuracy"] == 1.0  # Doc correct
        assert result.details["section_accuracy"] == 0.0

    def test_no_citations(self):
        """No citations should yield section accuracy of 0.0."""
        from evaluation_cc.metrics.citation import SectionAccuracy
        from evaluation_cc.schemas import EvalQuestion, EvalResponse, GoldPassage

        metric = SectionAccuracy()

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
            gold_passages=[
                GoldPassage(doc_id="doc1", chunk_id="section1", text="text1"),
            ],
        )

        response = EvalResponse(
            question_id="q1",
            answer="Y",
            citations=[],
        )

        result = metric.compute(question, response)

        assert result.value == 0.0
        assert "No citations" in result.details.get("note", "")


# =============================================================================
# CITATION EXTRACTION TESTS
# =============================================================================


class TestCitationExtraction:
    """Tests for extracting numeric citations from LLM answers."""

    def test_extract_single_citation(self):
        """Single bracket citation should be extracted."""
        from pipelines.inference import extract_numeric_citations

        sources = [
            {"document_id": "doc1", "chunk_id": "chunk1"},
            {"document_id": "doc2", "chunk_id": "chunk2"},
        ]

        citations = extract_numeric_citations("The answer is Paris [1].", sources)

        assert len(citations) == 1
        assert citations[0]["source_index"] == 1
        assert citations[0]["document_id"] == "doc1"
        assert citations[0]["chunk_id"] == "chunk1"

    def test_extract_multiple_citations(self):
        """Multiple bracket citations should be extracted."""
        from pipelines.inference import extract_numeric_citations

        sources = [
            {"document_id": "doc1", "chunk_id": "chunk1"},
            {"document_id": "doc2", "chunk_id": "chunk2"},
            {"document_id": "doc3", "chunk_id": "chunk3"},
        ]

        citations = extract_numeric_citations("According to [1] and [2], also [3].", sources)

        assert len(citations) == 3
        assert [c["source_index"] for c in citations] == [1, 2, 3]

    def test_extract_comma_separated_citations(self):
        """Comma-separated citations like [1,2] should be expanded."""
        from pipelines.inference import extract_numeric_citations

        sources = [
            {"document_id": "doc1", "chunk_id": "chunk1"},
            {"document_id": "doc2", "chunk_id": "chunk2"},
            {"document_id": "doc3", "chunk_id": "chunk3"},
        ]

        citations = extract_numeric_citations("See sources [1, 2, 3].", sources)

        assert len(citations) == 3
        assert [c["source_index"] for c in citations] == [1, 2, 3]

    def test_extract_range_citations(self):
        """Range citations like [1-3] should be expanded."""
        from pipelines.inference import extract_numeric_citations

        sources = [
            {"document_id": "doc1", "chunk_id": "chunk1"},
            {"document_id": "doc2", "chunk_id": "chunk2"},
            {"document_id": "doc3", "chunk_id": "chunk3"},
        ]

        citations = extract_numeric_citations("Sources [1-3] agree.", sources)

        assert len(citations) == 3
        assert [c["source_index"] for c in citations] == [1, 2, 3]

    def test_dedupe_citations(self):
        """Duplicate citations should be deduplicated."""
        from pipelines.inference import extract_numeric_citations

        sources = [
            {"document_id": "doc1", "chunk_id": "chunk1"},
        ]

        citations = extract_numeric_citations("See [1] and again [1].", sources)

        assert len(citations) == 1
        assert citations[0]["source_index"] == 1

    def test_citation_out_of_range(self):
        """Citations beyond source count should be ignored."""
        from pipelines.inference import extract_numeric_citations

        sources = [
            {"document_id": "doc1", "chunk_id": "chunk1"},
        ]

        citations = extract_numeric_citations("See [1] and [5].", sources)

        assert len(citations) == 1
        assert citations[0]["source_index"] == 1

    def test_no_citations(self):
        """Answer without citations should return empty list."""
        from pipelines.inference import extract_numeric_citations

        sources = [
            {"document_id": "doc1", "chunk_id": "chunk1"},
        ]

        citations = extract_numeric_citations("The answer is Paris.", sources)

        assert citations == []

    def test_parenthesis_citations(self):
        """Parenthesis citations like (1) should also be extracted."""
        from pipelines.inference import extract_numeric_citations

        sources = [
            {"document_id": "doc1", "chunk_id": "chunk1"},
            {"document_id": "doc2", "chunk_id": "chunk2"},
        ]

        citations = extract_numeric_citations("According to (1) and (2).", sources)

        assert len(citations) == 2


# =============================================================================
# QUERY ENDPOINT TESTS (include_chunks)
# =============================================================================


class TestQueryEndpointIncludeChunks:
    """Tests for /query endpoint with include_chunks parameter."""

    def test_include_chunks_false_no_chunk_fields(self):
        """When include_chunks=False, chunk_id/chunk_index should not be in sources."""
        from pipelines.inference import extract_sources
        from unittest.mock import MagicMock

        # Create mock source nodes
        node1 = MagicMock()
        node1.id_ = "node-123"
        node1.metadata = {
            "document_id": "doc1",
            "file_name": "test.pdf",
            "chunk_index": 0,
            "path": "/path/to/test.pdf",
        }
        node1.get_content.return_value = "This is the text content"
        node1.score = 0.95

        sources = extract_sources([node1], include_chunks=False)

        assert len(sources) == 1
        assert "document_id" in sources[0]
        assert "chunk_id" not in sources[0]
        assert "chunk_index" not in sources[0]

    def test_include_chunks_true_has_chunk_fields(self):
        """When include_chunks=True, chunk_id/chunk_index should be in sources."""
        from pipelines.inference import extract_sources
        from unittest.mock import MagicMock

        # Create mock source nodes
        node1 = MagicMock()
        node1.id_ = "node-123"
        node1.metadata = {
            "document_id": "doc1",
            "file_name": "test.pdf",
            "chunk_index": 5,
            "path": "/path/to/test.pdf",
        }
        node1.get_content.return_value = "This is the text content"
        node1.score = 0.95

        sources = extract_sources([node1], include_chunks=True, dedupe_by_document=False)

        assert len(sources) == 1
        assert sources[0]["chunk_id"] == "node-123"
        assert sources[0]["chunk_index"] == 5

    def test_include_chunks_disables_deduplication(self):
        """When include_chunks=True, multiple chunks from same doc should be returned."""
        from pipelines.inference import extract_sources
        from unittest.mock import MagicMock

        # Create mock source nodes from same document
        node1 = MagicMock()
        node1.id_ = "chunk-1"
        node1.metadata = {
            "document_id": "doc1",
            "file_name": "test.pdf",
            "chunk_index": 0,
        }
        node1.get_content.return_value = "First chunk"
        node1.score = 0.95

        node2 = MagicMock()
        node2.id_ = "chunk-2"
        node2.metadata = {
            "document_id": "doc1",  # Same document
            "file_name": "test.pdf",
            "chunk_index": 1,
        }
        node2.get_content.return_value = "Second chunk"
        node2.score = 0.90

        sources = extract_sources([node1, node2], include_chunks=True, dedupe_by_document=False)

        assert len(sources) == 2
        assert sources[0]["chunk_id"] == "chunk-1"
        assert sources[1]["chunk_id"] == "chunk-2"


# =============================================================================
# INTEGRATION TESTS (require running services)
# =============================================================================


# =============================================================================
# DATASET LOADER TESTS
# =============================================================================


@pytest.mark.eval
class TestDatasetRegistry:
    """Tests for dataset registry functionality."""

    def test_list_available_returns_all_datasets(self):
        """Registry should list all registered datasets."""
        from evaluation_cc.datasets.registry import list_available
        from evaluation_cc.config import DatasetName

        available = list_available()

        assert DatasetName.RAGBENCH in available
        assert DatasetName.QASPER in available
        assert DatasetName.SQUAD_V2 in available
        assert DatasetName.HOTPOTQA in available
        assert DatasetName.MSMARCO in available

    def test_get_loader_returns_instance(self):
        """get_loader should return cached loader instances."""
        from evaluation_cc.datasets.registry import get_loader
        from evaluation_cc.config import DatasetName

        loader1 = get_loader(DatasetName.RAGBENCH)
        loader2 = get_loader(DatasetName.RAGBENCH)

        # Should be same cached instance
        assert loader1 is loader2
        assert loader1.name == "ragbench"

    def test_get_metadata_returns_dict(self):
        """get_metadata should return loader metadata."""
        from evaluation_cc.datasets.registry import get_metadata
        from evaluation_cc.config import DatasetName

        metadata = get_metadata(DatasetName.RAGBENCH)

        assert "name" in metadata
        assert "description" in metadata
        assert metadata["name"] == "ragbench"


@pytest.mark.eval
class TestRAGBenchLoader:
    """Tests for RAGBench dataset loader."""

    def test_loader_properties(self):
        """RAGBench loader should have correct properties."""
        from evaluation_cc.datasets.ragbench import RAGBenchLoader

        loader = RAGBenchLoader()

        assert loader.name == "ragbench"
        assert "legal" in loader.domains or "finance" in loader.domains
        assert "retrieval" in loader.primary_aspects

    def test_load_small_sample(self):
        """Should load small sample from RAGBench."""
        from evaluation_cc.datasets.ragbench import RAGBenchLoader

        loader = RAGBenchLoader()
        dataset = loader.load(split="test", max_samples=3, seed=42)

        assert dataset.name == "ragbench"
        assert len(dataset.questions) <= 3
        assert all(q.question for q in dataset.questions)

    def test_sampling_is_reproducible(self):
        """Same seed should produce same samples."""
        from evaluation_cc.datasets.ragbench import RAGBenchLoader

        loader = RAGBenchLoader()

        dataset1 = loader.load(split="test", max_samples=5, seed=42)
        dataset2 = loader.load(split="test", max_samples=5, seed=42)

        # Questions should be identical with same seed
        assert len(dataset1) == len(dataset2)
        ids1 = {q.id for q in dataset1.questions}
        ids2 = {q.id for q in dataset2.questions}
        assert ids1 == ids2


@pytest.mark.eval
class TestQasperLoader:
    """Tests for Qasper dataset loader."""

    @pytest.mark.xfail(
        reason="Qasper dataset uses loading script which is no longer supported in datasets>=4.0"
    )
    def test_load_qasper_sample(self):
        """Should load sample from Qasper dataset."""
        from evaluation_cc.datasets.qasper import QasperLoader

        loader = QasperLoader()
        dataset = loader.load(split="test", max_samples=2, seed=42)

        assert dataset.name == "qasper"
        assert len(dataset.questions) <= 2


@pytest.mark.eval
class TestSquadV2Loader:
    """Tests for SQuAD v2 dataset loader."""

    def test_load_squad_sample(self):
        """Should load sample from SQuAD v2 dataset."""
        from evaluation_cc.datasets.squad_v2 import SquadV2Loader

        loader = SquadV2Loader()
        dataset = loader.load(split="validation", max_samples=2, seed=42)

        assert dataset.name == "squad_v2"
        assert len(dataset.questions) <= 2

    def test_squad_has_unanswerable_questions(self):
        """SQuAD v2 should include some unanswerable questions."""
        from evaluation_cc.datasets.squad_v2 import SquadV2Loader

        loader = SquadV2Loader()
        # Load larger sample to find unanswerable
        dataset = loader.load(split="validation", max_samples=50, seed=42)

        unanswerable = [q for q in dataset.questions if q.is_unanswerable]

        # SQuAD v2 has ~50% unanswerable questions
        assert len(unanswerable) > 0


@pytest.mark.eval
class TestHotpotQALoader:
    """Tests for HotpotQA dataset loader."""

    def test_load_hotpotqa_sample(self):
        """Should load sample from HotpotQA dataset."""
        from evaluation_cc.datasets.hotpotqa import HotpotQALoader

        loader = HotpotQALoader()
        dataset = loader.load(split="validation", max_samples=2, seed=42)

        assert dataset.name == "hotpotqa"
        assert len(dataset.questions) <= 2


@pytest.mark.eval
class TestMSMarcoLoader:
    """Tests for MS MARCO dataset loader."""

    def test_load_msmarco_sample(self):
        """Should load sample from MS MARCO dataset."""
        from evaluation_cc.datasets.msmarco import MSMarcoLoader

        loader = MSMarcoLoader()
        dataset = loader.load(split="validation", max_samples=2, seed=42)

        assert dataset.name == "msmarco"
        assert len(dataset.questions) <= 2


@pytest.mark.eval
class TestLoadDatasets:
    """Tests for loading multiple datasets."""

    def test_load_multiple_datasets(self):
        """Should load multiple datasets with combined samples."""
        from evaluation_cc import load_datasets, DatasetName

        datasets = load_datasets(
            [DatasetName.RAGBENCH, DatasetName.SQUAD_V2],
            max_samples=2,
            seed=42,
        )

        assert len(datasets) == 2
        assert datasets[0].name == "ragbench"
        assert datasets[1].name == "squad_v2"


# =============================================================================
# INTEGRATION TESTS (require running services)
# =============================================================================


# =============================================================================
# PERFORMANCE METRICS TESTS
# =============================================================================


class TestCostCalculation:
    """Tests for cost calculation based on model pricing."""

    def test_anthropic_model_cost(self):
        """Anthropic models should use correct pricing."""
        from evaluation_cc.config import get_model_cost

        # Claude Sonnet: $3/1M input, $15/1M output
        cost = get_model_cost(
            model="claude-sonnet-4-20250514",
            prompt_tokens=1000,
            completion_tokens=500,
        )

        # Expected: (1000 * 3 + 500 * 15) / 1_000_000 = 0.0105
        assert cost == pytest.approx(0.0105, abs=0.0001)

    def test_openai_model_cost(self):
        """OpenAI models should use correct pricing."""
        from evaluation_cc.config import get_model_cost

        # GPT-4o: $2.50/1M input, $10/1M output
        cost = get_model_cost(
            model="gpt-4o",
            prompt_tokens=2000,
            completion_tokens=1000,
        )

        # Expected: (2000 * 2.5 + 1000 * 10) / 1_000_000 = 0.015
        assert cost == pytest.approx(0.015, abs=0.0001)

    def test_ollama_model_free(self):
        """Ollama models should be free."""
        from evaluation_cc.config import get_model_cost

        cost = get_model_cost(
            model="ollama/gemma3:4b",
            prompt_tokens=10000,
            completion_tokens=5000,
        )

        assert cost == 0.0

    def test_unknown_model_defaults_free(self):
        """Unknown models should default to free."""
        from evaluation_cc.config import get_model_cost

        cost = get_model_cost(
            model="some-unknown-model",
            prompt_tokens=10000,
            completion_tokens=5000,
        )

        assert cost == 0.0

    def test_cost_per_query_metric(self):
        """CostPerQuery metric should compute correctly."""
        from evaluation_cc.metrics.performance import CostPerQuery
        from evaluation_cc.schemas import (
            EvalQuestion,
            EvalResponse,
            QueryMetrics,
            TokenUsage,
        )

        metric = CostPerQuery(model="gpt-4o-mini")

        question = EvalQuestion(
            id="q1",
            question="What is X?",
            expected_answer="Y",
        )

        # GPT-4o-mini: $0.15/1M input, $0.60/1M output
        response = EvalResponse(
            question_id="q1",
            answer="Y",
            metrics=QueryMetrics(
                latency_ms=100,
                token_usage=TokenUsage(
                    prompt_tokens=1000,
                    completion_tokens=500,
                    total_tokens=1500,
                ),
            ),
        )

        result = metric.compute(question, response)

        # Expected: (1000 * 0.15 + 500 * 0.60) / 1_000_000 = 0.00045
        assert result.name == "cost_per_query"
        assert result.value == pytest.approx(0.00045, abs=0.00001)


class TestParetoAnalysis:
    """Tests for Pareto frontier analysis."""

    def test_pareto_single_run(self):
        """Single run should be on the frontier."""
        from evaluation_cc.runner import compute_pareto_frontier
        from evaluation_cc.schemas import EvalRun, ConfigSnapshot, WeightedScore
        from datetime import datetime

        run = EvalRun(
            id="run1",
            name="test-run",
            created_at=datetime.now(),
            config=ConfigSnapshot(
                llm_model="test",
                llm_provider="test",
                embedding_model="test",
            ),
            weighted_score=WeightedScore(
                score=0.8,
                objectives={"accuracy": 0.9, "latency": 0.7},
            ),
        )

        points = compute_pareto_frontier([run])

        assert len(points) == 1
        assert not points[0].is_dominated

    def test_pareto_dominated_run(self):
        """A dominated run should be marked as such."""
        from evaluation_cc.runner import compute_pareto_frontier
        from evaluation_cc.schemas import EvalRun, ConfigSnapshot, WeightedScore
        from datetime import datetime

        # Run 2 dominates Run 1 (better in both objectives)
        run1 = EvalRun(
            id="run1",
            name="dominated-run",
            created_at=datetime.now(),
            config=ConfigSnapshot(
                llm_model="test",
                llm_provider="test",
                embedding_model="test",
            ),
            weighted_score=WeightedScore(
                score=0.6,
                objectives={"accuracy": 0.6, "latency": 0.5},
            ),
        )

        run2 = EvalRun(
            id="run2",
            name="dominant-run",
            created_at=datetime.now(),
            config=ConfigSnapshot(
                llm_model="test",
                llm_provider="test",
                embedding_model="test",
            ),
            weighted_score=WeightedScore(
                score=0.8,
                objectives={"accuracy": 0.9, "latency": 0.7},
            ),
        )

        points = compute_pareto_frontier([run1, run2])

        # Find each point
        point1 = next(p for p in points if p.run_id == "run1")
        point2 = next(p for p in points if p.run_id == "run2")

        assert point1.is_dominated
        assert not point2.is_dominated
        assert "run1" in point2.dominates

    def test_pareto_frontier_tradeoff(self):
        """Runs with tradeoffs should both be on frontier."""
        from evaluation_cc.runner import compute_pareto_frontier
        from evaluation_cc.schemas import EvalRun, ConfigSnapshot, WeightedScore
        from datetime import datetime

        # Run 1: Better accuracy, worse latency
        run1 = EvalRun(
            id="run1",
            name="high-accuracy",
            created_at=datetime.now(),
            config=ConfigSnapshot(
                llm_model="test",
                llm_provider="test",
                embedding_model="test",
            ),
            weighted_score=WeightedScore(
                score=0.8,
                objectives={"accuracy": 0.95, "latency": 0.3},
            ),
        )

        # Run 2: Better latency, worse accuracy
        run2 = EvalRun(
            id="run2",
            name="low-latency",
            created_at=datetime.now(),
            config=ConfigSnapshot(
                llm_model="test",
                llm_provider="test",
                embedding_model="test",
            ),
            weighted_score=WeightedScore(
                score=0.7,
                objectives={"accuracy": 0.7, "latency": 0.9},
            ),
        )

        points = compute_pareto_frontier([run1, run2])

        # Both should be on the frontier (neither dominates the other)
        point1 = next(p for p in points if p.run_id == "run1")
        point2 = next(p for p in points if p.run_id == "run2")

        assert not point1.is_dominated
        assert not point2.is_dominated

    def test_cli_pareto_from_dicts(self):
        """CLI helper should compute Pareto from dict data."""
        from evaluation_cc.cli import _compute_pareto_from_dicts

        runs = [
            {
                "id": "run1",
                "name": "dominated",
                "weighted_score": {
                    "score": 0.5,
                    "objectives": {"accuracy": 0.5, "latency": 0.5},
                },
            },
            {
                "id": "run2",
                "name": "dominant",
                "weighted_score": {
                    "score": 0.8,
                    "objectives": {"accuracy": 0.8, "latency": 0.8},
                },
            },
        ]

        points = _compute_pareto_from_dicts(runs)

        point1 = next(p for p in points if p["run_id"] == "run1")
        point2 = next(p for p in points if p["run_id"] == "run2")

        assert point1["is_dominated"]
        assert not point2["is_dominated"]


@pytest.mark.eval
class TestEvalIntegration:
    """Integration tests for evaluation framework.

    These tests require:
    - RAG server running at localhost:8001
    - ANTHROPIC_API_KEY set for LLM judge
    """

    def test_eval_module_imports(self):
        """Verify evaluation module can be imported."""
        from evaluation_cc import EvalConfig, run_evaluation

        assert EvalConfig is not None
        assert run_evaluation is not None

    def test_eval_config_creation(self):
        """Verify EvalConfig can be created with defaults."""
        from evaluation_cc import EvalConfig, DatasetName

        config = EvalConfig(
            datasets=[DatasetName.RAGBENCH],
            samples_per_dataset=5,
        )

        assert config.samples_per_dataset == 5
        assert DatasetName.RAGBENCH in config.datasets

    def test_eval_metrics_initialization(self):
        """Verify metrics can be initialized."""
        from evaluation_cc.metrics import METRIC_GROUPS

        assert "retrieval" in METRIC_GROUPS
        assert "generation" in METRIC_GROUPS
        assert "citation" in METRIC_GROUPS

    def test_list_available_datasets(self):
        """Verify available datasets can be listed."""
        from evaluation_cc import list_available_datasets

        datasets = list_available_datasets()

        assert "ragbench" in datasets or len(datasets) >= 1

    def test_eval_runner_creation(self):
        """Verify EvaluationRunner can be created."""
        from evaluation_cc import EvalConfig, EvaluationRunner, DatasetName

        config = EvalConfig(
            datasets=[DatasetName.RAGBENCH],
            samples_per_dataset=1,
        )
        config.judge.enabled = False  # Don't require API key

        runner = EvaluationRunner(config)

        assert runner is not None
        assert runner.config == config
