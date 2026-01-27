from pydantic import BaseModel


class ModelsInfoResponse(BaseModel):
    llm_model: str
    llm_provider: str
    llm_hosting: str
    embedding_model: str
    reranker_model: str | None
    reranker_enabled: bool
    cost_per_1m_input_tokens: float
    cost_per_1m_output_tokens: float


class ConfigResponse(BaseModel):
    max_upload_size_mb: int
