"""User-facing settings endpoints (GET /settings, PATCH /settings)."""

import logging

from fastapi import APIRouter, HTTPException

from infrastructure.config.models_config import get_models_config, update_config_file
from schemas.settings import SettingsResponse, SettingsUpdate

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/settings", response_model=SettingsResponse)
async def get_settings():
    """Return current toggleable settings."""
    config = get_models_config()
    return SettingsResponse(
        contextual_retrieval_enabled=config.retrieval.enable_contextual_retrieval,
    )


@router.patch("/settings", response_model=SettingsResponse)
async def update_settings(body: SettingsUpdate):
    """Update settings. Writes to config.yml so both rag-server and worker pick up changes."""
    try:
        if body.contextual_retrieval_enabled is not None:
            update_config_file(
                "retrieval.enable_contextual_retrieval",
                body.contextual_retrieval_enabled,
            )
            logger.info(f"[SETTINGS] contextual_retrieval_enabled set to {body.contextual_retrieval_enabled}")
    except Exception as e:
        logger.error(f"[SETTINGS] Failed to update settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    config = get_models_config()
    return SettingsResponse(
        contextual_retrieval_enabled=config.retrieval.enable_contextual_retrieval,
    )
