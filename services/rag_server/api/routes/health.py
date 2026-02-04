import os
from fastapi import APIRouter

from schemas.health import ModelsInfoResponse, ConfigResponse
from pipelines.inference import get_inference_config
from infrastructure.config.models_config import get_models_config

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.get("/models/info", response_model=ModelsInfoResponse)
async def get_models_info():
    """Get information about the models used in the RAG system"""
    # Model costs per 1M tokens (inline lookup)
    MODEL_COSTS = {
        "gpt-4o": {"input": 2.50, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
        "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
        "deepseek-chat": {"input": 0.27, "output": 1.10},
        "deepseek-reasoner": {"input": 0.55, "output": 2.19},
        "moonshot-v1-8k": {"input": 1.0, "output": 1.0},
        "moonshot-v1-32k": {"input": 2.0, "output": 2.0},
        "moonshot-v1-128k": {"input": 5.0, "output": 5.0},
    }

    models_config = get_models_config()
    llm_model = models_config.llm.model
    llm_provider = models_config.llm.provider

    # Determine hosting type
    llm_hosting = "local" if llm_provider == "ollama" else "cloud"

    # Get cost rates (0 for local models)
    cost_rates = MODEL_COSTS.get(llm_model, {"input": 0.0, "output": 0.0})

    embedding_model = os.getenv("EMBEDDING_MODEL", "unknown")

    inference_config = get_inference_config()
    reranker_enabled = inference_config['reranker_enabled']
    reranker_model = inference_config['reranker_model'] if reranker_enabled else None

    return ModelsInfoResponse(
        llm_model=llm_model,
        llm_provider=llm_provider,
        llm_hosting=llm_hosting,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        reranker_enabled=reranker_enabled,
        cost_per_1m_input_tokens=cost_rates["input"],
        cost_per_1m_output_tokens=cost_rates["output"]
    )


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get configuration settings for the RAG system"""
    max_upload_size = int(os.getenv("MAX_UPLOAD_SIZE", "80"))

    return ConfigResponse(
        max_upload_size_mb=max_upload_size
    )
