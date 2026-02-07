"""API key management endpoints."""

import logging
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException

from app.settings import get_api_key_for_provider, set_runtime_api_key
from infrastructure.llm.validation import validate_api_key
from schemas.api_keys import ApiKeySetRequest, ApiKeySetResponse, ApiKeyStatus

logger = logging.getLogger(__name__)

router = APIRouter()


def _load_config() -> dict[str, Any]:
    """Load config.yml to get model definitions."""
    config_paths = [
        Path("/app/config.yml"),  # Docker path
        Path(__file__).parent.parent.parent.parent.parent / "config.yml",  # Dev path
    ]

    for config_path in config_paths:
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)

    raise FileNotFoundError("config.yml not found")


def _get_providers_requiring_keys() -> set[str]:
    """Get set of providers that require API keys from config.yml."""
    config = _load_config()
    providers = set()

    # Check all model types (inference, embedding, eval)
    for model_type in ["inference", "embedding", "eval"]:
        if model_type in config.get("models", {}):
            for model_def in config["models"][model_type].values():
                if model_def.get("requires_api_key", False):
                    provider = model_def.get("provider")
                    if provider:
                        providers.add(provider.lower())

    return providers


def _mask_api_key(api_key: str) -> str:
    """Mask an API key for display, showing only first 7 and last 3 characters."""
    if len(api_key) <= 10:
        return "***" + api_key[-3:]
    return api_key[:7] + "***" + api_key[-3:]


@router.get("/api-keys", response_model=list[ApiKeyStatus])
async def list_api_key_status():
    """List all providers requiring API keys and their current status."""
    providers = _get_providers_requiring_keys()
    statuses = []

    for provider in sorted(providers):
        api_key = get_api_key_for_provider(provider)
        has_key = api_key is not None and api_key.strip() != ""
        masked_key = _mask_api_key(api_key) if has_key else None

        statuses.append(
            ApiKeyStatus(provider=provider, has_key=has_key, masked_key=masked_key)
        )

    return statuses


@router.post("/api-keys/{provider}", response_model=ApiKeySetResponse)
async def set_api_key(provider: str, request: ApiKeySetRequest):
    """Set and validate an API key for a provider."""
    provider_lower = provider.lower()

    # Check if provider is configured
    providers = _get_providers_requiring_keys()
    if provider_lower not in providers:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{provider}' is not configured or does not require an API key",
        )

    # Validate the API key
    logger.info(f"Validating API key for provider: {provider_lower}")
    valid, error_message = await validate_api_key(provider_lower, request.api_key)

    if not valid:
        logger.warning(f"API key validation failed for {provider_lower}: {error_message}")
        raise HTTPException(status_code=400, detail=error_message or "Invalid API key")

    # Store the key in runtime memory
    set_runtime_api_key(provider_lower, request.api_key)
    logger.info(f"API key set successfully for provider: {provider_lower}")

    return ApiKeySetResponse(
        provider=provider_lower,
        status="valid",
        masked_key=_mask_api_key(request.api_key),
    )
