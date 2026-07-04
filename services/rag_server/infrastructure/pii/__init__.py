"""PII masking infrastructure — reversible masking for the opt-in cloud generation tier."""

from .config import get_pii_config
from .postprocessor import (
    PIIMaskingPostprocessor,
    clear_session_token_mapping,
    get_session_token_mapping,
)
from .service import (
    MaskingResult,
    PIILeakageError,
    PIIMaskingService,
    TokenMapping,
    UnmaskingResult,
    get_pii_service,
    mask_text,
    reset_pii_service,
    unmask_text,
)

__all__ = [
    "get_pii_config",
    "PIIMaskingPostprocessor",
    "get_session_token_mapping",
    "clear_session_token_mapping",
    "PIIMaskingService",
    "MaskingResult",
    "UnmaskingResult",
    "TokenMapping",
    "PIILeakageError",
    "get_pii_service",
    "reset_pii_service",
    "mask_text",
    "unmask_text",
]
