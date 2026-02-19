from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    is_temporary: bool = False
    include_chunks: bool = False


class ContextPassage(BaseModel):
    text: str
    doc_id: str


class QueryWithContextRequest(BaseModel):
    query: str
    context_passages: list[ContextPassage]
    session_id: str | None = None


class TokenUsage(BaseModel):
    """Token usage statistics for a query."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class QueryMetrics(BaseModel):
    """Performance metrics for a query."""
    latency_ms: float
    token_usage: TokenUsage | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str
    citations: list[dict] | None = None
    metrics: QueryMetrics | None = None
