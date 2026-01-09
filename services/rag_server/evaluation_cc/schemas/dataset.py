"""Dataset schemas for evaluation questions and gold passages."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryType(str, Enum):
    """Types of queries in evaluation datasets."""

    FACTOID = "factoid"  # Simple fact-based questions
    MULTI_HOP = "multi_hop"  # Requires reasoning across multiple sources
    SUMMARY = "summary"  # Long-form summary/report tasks
    UNANSWERABLE = "unanswerable"  # Questions that cannot be answered from context
    COMPARISON = "comparison"  # Compare/contrast questions
    PROCEDURAL = "procedural"  # How-to questions


class Difficulty(str, Enum):
    """Difficulty levels for evaluation questions."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class GoldPassage:
    """A gold (ground truth) passage that should be retrieved for a question.

    Attributes:
        doc_id: Unique identifier for the source document
        chunk_id: Unique identifier for the specific chunk within the document
        text: The actual text content of the passage
        relevance_score: Optional relevance score (1.0 = fully relevant)
    """

    doc_id: str
    chunk_id: str
    text: str
    relevance_score: float = 1.0

    def __post_init__(self):
        if not self.doc_id:
            raise ValueError("doc_id cannot be empty")
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")


@dataclass
class EvalQuestion:
    """An evaluation question with expected answer and gold passages.

    Attributes:
        id: Unique identifier for this question
        question: The question text
        expected_answer: The ground truth answer (may be None for unanswerable)
        gold_passages: List of passages that contain the answer
        query_type: Type of query (factoid, multi_hop, etc.)
        difficulty: Difficulty level
        domain: Domain/category of the question (e.g., "legal", "technical")
        is_unanswerable: Whether this question is intentionally unanswerable
        metadata: Additional dataset-specific metadata
    """

    id: str
    question: str
    expected_answer: str | None
    gold_passages: list[GoldPassage] = field(default_factory=list)
    query_type: QueryType = QueryType.FACTOID
    difficulty: Difficulty = Difficulty.MEDIUM
    domain: str = "general"
    is_unanswerable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            raise ValueError("id cannot be empty")
        if not self.question:
            raise ValueError("question cannot be empty")
        if self.is_unanswerable and self.expected_answer:
            # Unanswerable questions shouldn't have expected answers
            pass  # Some datasets may include "unanswerable" as the answer


@dataclass
class EvalDataset:
    """A collection of evaluation questions from a specific source.

    Attributes:
        name: Dataset name (e.g., "ragbench", "squad_v2")
        version: Dataset version string
        questions: List of evaluation questions
        description: Human-readable description
        source_url: URL to the original dataset
        domains: List of domains covered by this dataset
        metadata: Additional dataset-level metadata
    """

    name: str
    version: str
    questions: list[EvalQuestion]
    description: str = ""
    source_url: str = ""
    domains: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    def filter_by_domain(self, domain: str) -> "EvalDataset":
        """Return a new dataset filtered to a specific domain."""
        filtered = [q for q in self.questions if q.domain == domain]
        return EvalDataset(
            name=f"{self.name}:{domain}",
            version=self.version,
            questions=filtered,
            description=f"{self.description} (filtered to {domain})",
            source_url=self.source_url,
            domains=[domain],
            metadata=self.metadata,
        )

    def filter_by_query_type(self, query_type: QueryType) -> "EvalDataset":
        """Return a new dataset filtered to a specific query type."""
        filtered = [q for q in self.questions if q.query_type == query_type]
        return EvalDataset(
            name=f"{self.name}:{query_type.value}",
            version=self.version,
            questions=filtered,
            description=f"{self.description} (filtered to {query_type.value})",
            source_url=self.source_url,
            domains=self.domains,
            metadata=self.metadata,
        )

    def sample(self, n: int, seed: int | None = None) -> "EvalDataset":
        """Return a random sample of questions."""
        import random

        if seed is not None:
            random.seed(seed)

        sampled = random.sample(self.questions, min(n, len(self.questions)))
        return EvalDataset(
            name=f"{self.name}:sample({n})",
            version=self.version,
            questions=sampled,
            description=f"{self.description} (sampled {len(sampled)} questions)",
            source_url=self.source_url,
            domains=self.domains,
            metadata=self.metadata,
        )
