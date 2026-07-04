"""PII audit logging for compliance tracking."""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from infrastructure.config.models_config import PiiAuditConfig

# Dedicated audit logger (separate from application logs)
audit_logger = logging.getLogger("pii.audit")


class PIIAuditLogger:
    """Structured audit logging for PII operations."""

    def __init__(self, config: PiiAuditConfig):
        self.config = config
        if not audit_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - PII_AUDIT - %(levelname)s - %(message)s"))
            audit_logger.addHandler(handler)
        audit_logger.setLevel(getattr(logging, config.log_level, logging.INFO))

    def _log(self, level: str, data: dict) -> None:
        getattr(audit_logger, level)(json.dumps(data))

    def log_mask_operation(self, context_id: Optional[str], entities_count: int, entity_types: list[str]) -> None:
        self._log(
            "info",
            {
                "operation": "MASK",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context_id": context_id,
                "entities_count": entities_count,
                "entity_types": sorted(set(entity_types)),
            },
        )

    def log_unmask_operation(
        self, context_id: Optional[str], tokens_found: int, tokens_replaced: int, validation_passed: bool
    ) -> None:
        self._log(
            "info",
            {
                "operation": "UNMASK",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context_id": context_id,
                "tokens_found": tokens_found,
                "tokens_replaced": tokens_replaced,
                "validation_passed": validation_passed,
            },
        )

    def log_pii_leak_detected(self, context_id: Optional[str], entities_count: int, entity_types: list[str]) -> None:
        self._log(
            "warning",
            {
                "operation": "PII_LEAK_DETECTED",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context_id": context_id,
                "entities_count": entities_count,
                "entity_types": sorted(set(entity_types)),
            },
        )
