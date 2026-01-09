"""Response schemas for capturing RAG system outputs."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """A chunk retrieved by the RAG system.

    Attributes:
        doc_id: Document identifier
        chunk_id: Chunk identifier within the document
        text: The retrieved text content
        score: Retrieval/reranking score
        rank: Position in the retrieval results (1-indexed)
        metadata: Additional chunk metadata
    """

    doc_id: str
    chunk_id: str
    text: str
    score: float | None = None
    rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    """A citation in the RAG system's answer.

    Attributes:
        source_index: Index in the sources list (1-indexed as shown to user)
        doc_id: Document identifier
        chunk_id: Chunk identifier
        chunk_index: Chunk index within document (if available)
        text_span: The cited text span (if extractable)
    """

    source_index: int
    doc_id: str | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    text_span: str | None = None


@dataclass
class TokenUsage:
    """Token usage for a query.

    Attributes:
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        total_tokens: Total tokens used
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class QueryMetrics:
    """Performance metrics for a single query.

    Attributes:
        latency_ms: Query latency in milliseconds
        token_usage: Token counts (if available)
    """

    latency_ms: float
    token_usage: TokenUsage | None = None


@dataclass
class EvalResponse:
    """The complete response from the RAG system for evaluation.

    Attributes:
        question_id: ID of the question being answered
        answer: The generated answer text
        retrieved_chunks: All chunks retrieved by the system
        citations: Citations extracted from the answer
        session_id: Session ID used for the query
        metrics: Performance metrics (latency, tokens)
        raw_response: The raw API response (for debugging)
    """

    question_id: str
    answer: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    session_id: str | None = None
    metrics: QueryMetrics | None = None
    raw_response: dict[str, Any] | None = None

    @property
    def has_answer(self) -> bool:
        """Check if a non-empty answer was provided."""
        return bool(self.answer and self.answer.strip())

    @property
    def cited_doc_ids(self) -> set[str]:
        """Get set of document IDs that were cited."""
        return {c.doc_id for c in self.citations if c.doc_id}

    @property
    def cited_chunk_ids(self) -> set[str]:
        """Get set of chunk IDs that were cited."""
        return {c.chunk_id for c in self.citations if c.chunk_id}

    @property
    def retrieved_doc_ids(self) -> set[str]:
        """Get set of document IDs that were retrieved."""
        return {c.doc_id for c in self.retrieved_chunks}

    @property
    def retrieved_chunk_ids(self) -> set[str]:
        """Get set of chunk IDs that were retrieved."""
        return {c.chunk_id for c in self.retrieved_chunks}
