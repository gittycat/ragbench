# PII Masking Implementation Plan
Created Jan 24 2026

## Overview

Implement reversible PII masking for cloud LLM providers using Microsoft Presidio. Data sent to cloud LLMs is masked with tokens (e.g., `[[[PERSON_0]]]`), and responses are unmasked back to original values.

**Purpose**: Protect sensitive data (names, emails, SSNs, etc.) when using cloud LLM providers while maintaining response quality.

**Selected Approach:**
- **Masking Tool**: Microsoft Presidio (open source, NER + regex, LlamaIndex integration)
- **Unmasking**: Token mapping table with validation
- **Additional Features**: Audit logging + output guardrails

## Architecture

### Design Decision: Service Layer Approach

A dedicated `infrastructure/pii/` module provides mask/unmask functions called explicitly at LLM interaction points. This is preferred over:
- **Pure middleware**: Would require significant FastAPI changes, wouldn't cover Celery workers
- **LlamaIndex PresidioPIINodePostprocessor**: Only works on nodes/retrieval, doesn't cover user queries, chat history, or LLM responses

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PII MASKING DATA FLOW                             │
└─────────────────────────────────────────────────────────────────────────────┘

OUTBOUND (to LLM):
User Query ──► PIIMaskingService.mask() ──► Masked Query
                    │                            │
                    ▼                            ▼
              TokenMapping stored          "[[[PERSON_0]]]'s salary?"
              (session-scoped)                   │
                                                 ▼
                                          ┌─────────────┐
                                          │  Cloud LLM  │
                                          └──────┬──────┘
                                                 │
INBOUND (from LLM):                              ▼
              LLM Response ◄────────── "[[[PERSON_0]]]'s salary is $85k"
                    │
                    ▼
              validate_tokens_preserved()
                    │
              ┌─────┴─────┐
              │           │
         Tokens OK   Tokens Altered
              │           │
              ▼           ▼
         unmask()    fuzzy_recovery() → unmask()
              │           │
              ▼           ▼
        "John's salary is $85k"
              │
              ▼
        scan_for_leaked_pii() (output guardrail)
              │
              ▼
        Return to User
```

### Data Flow Points (4 paths to protect)

| Path | File | Function | Description |
|------|------|----------|-------------|
| User queries | `pipelines/inference.py` | `query_rag()` | User query, chat history, retrieved context |
| Contextual retrieval | `pipelines/ingestion.py` | `add_contextual_prefix_to_chunk()` | Document chunks during ingestion |
| Session titles | `services/session_titles.py` | `generate_ai_title()` | First user message for title generation |
| Evaluation | `evals/judges/llm_judge.py` | `_evaluate()` | Test data sent to eval LLM |

## Configuration Schema

Add to `config/models.yml`:

```yaml
pii:
  enabled: false  # Master toggle

  # Entity types to detect and mask
  # Full list: PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD,
  # US_SSN, IBAN_CODE, IP_ADDRESS, LOCATION, DATE_TIME, NRP,
  # MEDICAL_LICENSE, US_BANK_NUMBER, US_DRIVER_LICENSE, US_PASSPORT
  entities:
    - PERSON
    - EMAIL_ADDRESS
    - PHONE_NUMBER
    - CREDIT_CARD
    - US_SSN
    - IBAN_CODE
    - IP_ADDRESS

  # Token format - distinctive to avoid LLM alteration
  token_format: "[[[{entity_type}_{index}]]]"  # e.g., [[[PERSON_0]]]

  # Score threshold for PII detection (0.0-1.0)
  score_threshold: 0.5

  # Language for Presidio analyzer
  language: en

  # Validation settings
  validation:
    enabled: true
    max_retries: 2        # Retry with stricter prompt if tokens altered
    alert_on_failure: true  # Log warning if unmasking fails

  # Output guardrails - scan LLM responses for leaked PII
  output_guardrails:
    enabled: true
    block_on_detection: false  # If true, return error instead of response

  # Audit logging
  audit:
    enabled: true
    log_level: INFO  # DEBUG logs full text, INFO logs summaries only
```

## Files to Create

### 1. `services/rag_server/infrastructure/pii/__init__.py`

```python
"""PII masking infrastructure module."""
from .service import PIIMaskingService, mask_text, unmask_text, TokenMapping, MaskingResult, UnmaskingResult
from .config import PIIConfig, get_pii_config

__all__ = [
    "PIIMaskingService",
    "PIIConfig",
    "get_pii_config",
    "mask_text",
    "unmask_text",
    "TokenMapping",
    "MaskingResult",
    "UnmaskingResult",
]
```

### 2. `services/rag_server/infrastructure/pii/config.py`

```python
"""PII masking configuration."""
from pydantic import BaseModel, Field


class PIIValidationConfig(BaseModel):
    """Configuration for token validation."""
    enabled: bool = True
    max_retries: int = 2
    alert_on_failure: bool = True


class PIIGuardrailsConfig(BaseModel):
    """Configuration for output guardrails."""
    enabled: bool = True
    block_on_detection: bool = False


class PIIAuditConfig(BaseModel):
    """Configuration for audit logging."""
    enabled: bool = True
    log_level: str = "INFO"


class PIIConfig(BaseModel):
    """PII masking configuration."""
    enabled: bool = False
    entities: list[str] = Field(default_factory=lambda: [
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
        "CREDIT_CARD", "US_SSN", "IBAN_CODE", "IP_ADDRESS"
    ])
    token_format: str = "[[[{entity_type}_{index}]]]"
    score_threshold: float = 0.5
    language: str = "en"
    validation: PIIValidationConfig = Field(default_factory=PIIValidationConfig)
    output_guardrails: PIIGuardrailsConfig = Field(default_factory=PIIGuardrailsConfig)
    audit: PIIAuditConfig = Field(default_factory=PIIAuditConfig)


# Convenience function to get config from ModelsConfig
def get_pii_config() -> PIIConfig:
    """Get PII config from the global models config."""
    from infrastructure.config.models_config import get_models_config
    return get_models_config().pii
```

### 3. `services/rag_server/infrastructure/pii/service.py`

```python
"""PII masking service using Microsoft Presidio."""
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine

from .config import PIIConfig, get_pii_config
from .audit import PIIAuditLogger

logger = logging.getLogger(__name__)


@dataclass
class TokenMapping:
    """Stores original->token mappings for reversible anonymization."""
    mappings: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # Structure: {entity_type: {original_value: token}}
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_reverse_mapping(self) -> Dict[str, str]:
        """Get token->original mapping for unmasking."""
        reverse = {}
        for entity_type, type_mappings in self.mappings.items():
            for original, token in type_mappings.items():
                reverse[token] = original
        return reverse

    def get_expected_tokens(self) -> set[str]:
        """Get set of all tokens that should appear in response."""
        tokens = set()
        for type_mappings in self.mappings.values():
            tokens.update(type_mappings.values())
        return tokens


@dataclass
class MaskingResult:
    """Result of masking operation."""
    masked_text: str
    token_mapping: TokenMapping
    entities_found: List[RecognizerResult]


@dataclass
class UnmaskingResult:
    """Result of unmasking operation."""
    unmasked_text: str
    tokens_found: int
    tokens_replaced: int
    tokens_missing: List[str]
    validation_passed: bool


class PIIMaskingService:
    """Service for masking and unmasking PII in text."""

    def __init__(self, config: Optional[PIIConfig] = None):
        self.config = config or get_pii_config()
        self._analyzer: Optional[AnalyzerEngine] = None
        self._anonymizer: Optional[AnonymizerEngine] = None
        self._audit_logger: Optional[PIIAuditLogger] = None

    @property
    def analyzer(self) -> AnalyzerEngine:
        """Lazy-load Presidio analyzer."""
        if self._analyzer is None:
            self._analyzer = AnalyzerEngine()
            logger.info("[PII] Initialized Presidio AnalyzerEngine")
        return self._analyzer

    @property
    def anonymizer(self) -> AnonymizerEngine:
        """Lazy-load Presidio anonymizer."""
        if self._anonymizer is None:
            self._anonymizer = AnonymizerEngine()
        return self._anonymizer

    @property
    def audit_logger(self) -> PIIAuditLogger:
        """Lazy-load audit logger."""
        if self._audit_logger is None:
            self._audit_logger = PIIAuditLogger(self.config.audit)
        return self._audit_logger

    def mask(
        self,
        text: str,
        existing_mapping: Optional[TokenMapping] = None,
        context_id: Optional[str] = None
    ) -> MaskingResult:
        """
        Mask PII in text with reversible tokens.

        Args:
            text: Text to mask
            existing_mapping: Reuse existing mapping for consistency across multiple texts
            context_id: Session/request ID for audit logging

        Returns:
            MaskingResult with masked text and token mapping
        """
        if not self.config.enabled:
            return MaskingResult(
                masked_text=text,
                token_mapping=existing_mapping or TokenMapping(),
                entities_found=[]
            )

        # Analyze text for PII
        results = self.analyzer.analyze(
            text=text,
            entities=self.config.entities,
            language=self.config.language,
            score_threshold=self.config.score_threshold
        )

        if not results:
            return MaskingResult(
                masked_text=text,
                token_mapping=existing_mapping or TokenMapping(),
                entities_found=[]
            )

        # Build or extend token mapping
        mapping = existing_mapping or TokenMapping()
        entity_counters: Dict[str, int] = {}

        # Initialize counters from existing mapping
        for entity_type, type_mappings in mapping.mappings.items():
            entity_counters[entity_type] = len(type_mappings)

        def get_or_create_token(entity_type: str, original_value: str) -> str:
            """Get existing token or create new one for an entity."""
            if entity_type not in mapping.mappings:
                mapping.mappings[entity_type] = {}

            # Return existing token if this value was already mapped
            if original_value in mapping.mappings[entity_type]:
                return mapping.mappings[entity_type][original_value]

            # Create new token
            index = entity_counters.get(entity_type, 0)
            token = self.config.token_format.format(
                entity_type=entity_type,
                index=index
            )
            mapping.mappings[entity_type][original_value] = token
            entity_counters[entity_type] = index + 1
            return token

        # Sort results by start position (reverse) for safe string replacement
        sorted_results = sorted(results, key=lambda x: x.start, reverse=True)

        masked_text = text
        for result in sorted_results:
            original = text[result.start:result.end]
            token = get_or_create_token(result.entity_type, original)
            masked_text = masked_text[:result.start] + token + masked_text[result.end:]

        # Audit log
        if self.config.audit.enabled:
            self.audit_logger.log_mask_operation(
                context_id=context_id,
                entities_count=len(results),
                entity_types=[r.entity_type for r in results]
            )

        return MaskingResult(
            masked_text=masked_text,
            token_mapping=mapping,
            entities_found=results
        )

    def unmask(
        self,
        text: str,
        token_mapping: TokenMapping,
        context_id: Optional[str] = None
    ) -> UnmaskingResult:
        """
        Unmask tokens back to original PII values.

        Args:
            text: Text with tokens to unmask
            token_mapping: Mapping from mask operation
            context_id: Session/request ID for audit logging

        Returns:
            UnmaskingResult with validation status
        """
        if not self.config.enabled:
            return UnmaskingResult(
                unmasked_text=text,
                tokens_found=0,
                tokens_replaced=0,
                tokens_missing=[],
                validation_passed=True
            )

        reverse_mapping = token_mapping.get_reverse_mapping()
        expected_tokens = token_mapping.get_expected_tokens()

        # Find tokens present in text
        tokens_in_text = set()
        for token in expected_tokens:
            if token in text:
                tokens_in_text.add(token)

        # Replace tokens with originals
        unmasked_text = text
        tokens_replaced = 0

        for token, original in reverse_mapping.items():
            if token in unmasked_text:
                unmasked_text = unmasked_text.replace(token, original)
                tokens_replaced += 1

        # Identify missing/altered tokens
        tokens_missing = [t for t in expected_tokens if t not in text]
        validation_passed = len(tokens_missing) == 0

        # Audit log
        if self.config.audit.enabled:
            self.audit_logger.log_unmask_operation(
                context_id=context_id,
                tokens_found=len(tokens_in_text),
                tokens_replaced=tokens_replaced,
                validation_passed=validation_passed
            )

        return UnmaskingResult(
            unmasked_text=unmasked_text,
            tokens_found=len(tokens_in_text),
            tokens_replaced=tokens_replaced,
            tokens_missing=tokens_missing,
            validation_passed=validation_passed
        )

    def validate_tokens_preserved(
        self,
        token_mapping: TokenMapping,
        response_text: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate that all tokens from input are preserved in output.

        Args:
            token_mapping: Original mapping from mask operation
            response_text: LLM response to validate

        Returns:
            Tuple of (validation_passed, list of altered/missing tokens)
        """
        expected_tokens = token_mapping.get_expected_tokens()
        altered_tokens = [t for t in expected_tokens if t not in response_text]
        return len(altered_tokens) == 0, altered_tokens

    def attempt_fuzzy_recovery(
        self,
        text: str,
        token_mapping: TokenMapping
    ) -> str:
        """
        Attempt to recover altered tokens using fuzzy matching.

        Handles common LLM alterations:
        - Bracket removal: [[[PERSON_0]]] -> PERSON_0
        - Case changes: [[[PERSON_0]]] -> [[[person_0]]]
        - Separator changes: [[[PERSON_0]]] -> [[[PERSON-0]]]

        Args:
            text: Response text with potentially altered tokens
            token_mapping: Original mapping

        Returns:
            Text with fuzzy-matched tokens replaced with originals
        """
        reverse_mapping = token_mapping.get_reverse_mapping()
        result = text

        for token, original in reverse_mapping.items():
            if token in result:
                continue  # Token is intact, skip

            # Generate common variants
            variants = [
                token.replace("[[[", "").replace("]]]", ""),  # Brackets removed
                token.replace("[[[", "[").replace("]]]", "]"),  # Single brackets
                token.replace("_", " "),  # Underscore to space
                token.replace("_", "-"),  # Underscore to hyphen
                token.lower(),  # Lowercase
                token.upper(),  # Uppercase
            ]

            for variant in variants:
                if variant in result:
                    result = result.replace(variant, original)
                    logger.debug(f"[PII] Fuzzy recovered: {variant} -> {original[:20]}...")
                    break

        return result

    def scan_for_leaked_pii(
        self,
        text: str,
        context_id: Optional[str] = None
    ) -> List[RecognizerResult]:
        """
        Scan text for any PII that may have leaked through.
        Used as output guardrail after unmasking.

        Args:
            text: Final response text to scan
            context_id: Session/request ID for audit logging

        Returns:
            List of detected PII entities (empty if clean)
        """
        if not self.config.output_guardrails.enabled:
            return []

        results = self.analyzer.analyze(
            text=text,
            entities=self.config.entities,
            language=self.config.language,
            score_threshold=self.config.score_threshold
        )

        if results and self.config.audit.enabled:
            self.audit_logger.log_pii_leak_detected(
                context_id=context_id,
                entities_count=len(results),
                entity_types=[r.entity_type for r in results]
            )

        return results


# ============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ============================================================================

_service: Optional[PIIMaskingService] = None


def get_pii_service() -> PIIMaskingService:
    """Get or create the global PII masking service."""
    global _service
    if _service is None:
        _service = PIIMaskingService()
    return _service


def reset_pii_service() -> None:
    """Reset the global PII service. Useful for testing."""
    global _service
    _service = None


def mask_text(
    text: str,
    existing_mapping: Optional[TokenMapping] = None,
    context_id: Optional[str] = None
) -> MaskingResult:
    """Convenience function to mask text using global service."""
    return get_pii_service().mask(text, existing_mapping, context_id)


def unmask_text(
    text: str,
    token_mapping: TokenMapping,
    context_id: Optional[str] = None
) -> UnmaskingResult:
    """Convenience function to unmask text using global service."""
    return get_pii_service().unmask(text, token_mapping, context_id)
```

### 4. `services/rag_server/infrastructure/pii/audit.py`

```python
"""PII audit logging for compliance tracking."""
import json
import logging
from datetime import datetime
from typing import List, Optional

from .config import PIIAuditConfig

# Dedicated audit logger (separate from application logs)
audit_logger = logging.getLogger("pii.audit")


class PIIAuditLogger:
    """Structured audit logging for PII operations."""

    def __init__(self, config: PIIAuditConfig):
        self.config = config
        self._configure_logger()

    def _configure_logger(self) -> None:
        """Configure the audit logger with appropriate handlers."""
        if not audit_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - PII_AUDIT - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            audit_logger.addHandler(handler)
        audit_logger.setLevel(getattr(logging, self.config.log_level))

    def _log(self, level: str, data: dict) -> None:
        """Log structured audit event."""
        log_method = getattr(audit_logger, level.lower())
        log_method(json.dumps(data))

    def log_mask_operation(
        self,
        context_id: Optional[str],
        entities_count: int,
        entity_types: List[str]
    ) -> None:
        """Log a masking operation."""
        self._log("info", {
            "operation": "MASK",
            "timestamp": datetime.utcnow().isoformat(),
            "context_id": context_id,
            "entities_count": entities_count,
            "entity_types": list(set(entity_types))
        })

    def log_unmask_operation(
        self,
        context_id: Optional[str],
        tokens_found: int,
        tokens_replaced: int,
        validation_passed: bool
    ) -> None:
        """Log an unmasking operation."""
        self._log("info", {
            "operation": "UNMASK",
            "timestamp": datetime.utcnow().isoformat(),
            "context_id": context_id,
            "tokens_found": tokens_found,
            "tokens_replaced": tokens_replaced,
            "validation_passed": validation_passed
        })

    def log_pii_leak_detected(
        self,
        context_id: Optional[str],
        entities_count: int,
        entity_types: List[str]
    ) -> None:
        """Log when PII is detected in output (guardrail violation)."""
        self._log("warning", {
            "operation": "PII_LEAK_DETECTED",
            "timestamp": datetime.utcnow().isoformat(),
            "context_id": context_id,
            "entities_count": entities_count,
            "entity_types": list(set(entity_types))
        })

    def log_token_validation_failure(
        self,
        context_id: Optional[str],
        altered_tokens_count: int
    ) -> None:
        """Log when LLM alters tokens."""
        self._log("warning", {
            "operation": "TOKEN_VALIDATION_FAILED",
            "timestamp": datetime.utcnow().isoformat(),
            "context_id": context_id,
            "altered_tokens_count": altered_tokens_count
        })
```

## Files to Modify

### 1. `services/rag_server/infrastructure/config/models_config.py`

Add import and field to `ModelsConfig`:

```python
# Add to imports at top
from infrastructure.pii.config import PIIConfig

# Add to ModelsConfig class (around line 107)
class ModelsConfig(BaseModel):
    """Root configuration for all models and retrieval settings."""

    llm: LLMConfig
    embedding: EmbeddingConfig
    eval: EvalConfig
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    pii: PIIConfig = Field(default_factory=PIIConfig)  # ADD THIS LINE
```

### 2. `services/rag_server/pipelines/inference.py`

Modify `query_rag()` function to apply PII masking:

```python
# Add imports at top
from infrastructure.pii.service import (
    mask_text, unmask_text, TokenMapping, get_pii_service
)
from infrastructure.pii.config import get_pii_config

# Modify query_rag() function - add masking before LLM call and unmasking after
def query_rag(
    query_text: str,
    session_id: str,
    index: VectorStoreIndex,
    is_temporary: bool = False,
) -> Dict:
    """Process RAG query with optional PII masking."""

    pii_config = get_pii_config()
    token_mapping = TokenMapping()

    # === PII MASKING (if enabled) ===
    if pii_config.enabled:
        # Mask user query
        mask_result = mask_text(query_text, context_id=session_id)
        masked_query = mask_result.masked_text
        token_mapping = mask_result.token_mapping

        if mask_result.entities_found:
            logger.info(f"[PII] Masked {len(mask_result.entities_found)} entities in query")
    else:
        masked_query = query_text

    # ... existing code for chat engine creation ...
    # Use masked_query instead of query_text when calling chat_engine

    response = chat_engine.chat(masked_query)  # Use masked query
    response_text = str(response)

    # === PII UNMASKING (if enabled) ===
    if pii_config.enabled and token_mapping.mappings:
        pii_service = get_pii_service()

        # Validate tokens preserved
        if pii_config.validation.enabled:
            valid, altered = pii_service.validate_tokens_preserved(token_mapping, response_text)
            if not valid:
                logger.warning(f"[PII] {len(altered)} tokens altered by LLM, attempting recovery")
                response_text = pii_service.attempt_fuzzy_recovery(response_text, token_mapping)

        # Unmask response
        unmask_result = unmask_text(response_text, token_mapping, context_id=session_id)
        final_answer = unmask_result.unmasked_text

        if not unmask_result.validation_passed:
            logger.warning(f"[PII] Unmasking incomplete: {len(unmask_result.tokens_missing)} tokens not found")

        # Output guardrail scan
        if pii_config.output_guardrails.enabled:
            leaked = pii_service.scan_for_leaked_pii(final_answer, context_id=session_id)
            if leaked:
                logger.warning(f"[PII] Output guardrail: {len(leaked)} PII entities in response")
                if pii_config.output_guardrails.block_on_detection:
                    raise PIILeakageError(f"PII detected in response: {len(leaked)} entities")
    else:
        final_answer = response_text

    return {
        'answer': final_answer,
        # ... rest of response
    }


class PIILeakageError(Exception):
    """Raised when PII is detected in LLM output and blocking is enabled."""
    pass
```

### 3. `services/rag_server/pipelines/ingestion.py`

Modify `add_contextual_prefix_to_chunk()` to mask chunks:

```python
# Add imports at top
from infrastructure.pii.service import mask_text, unmask_text
from infrastructure.pii.config import get_pii_config

# Modify add_contextual_prefix_to_chunk()
def add_contextual_prefix_to_chunk(
    node: TextNode,
    document_name: str,
    document_type: str
) -> TextNode:
    """Add contextual prefix to chunk with PII masking."""

    pii_config = get_pii_config()
    chunk_preview = node.get_content()[:400]

    # Mask chunk preview if PII masking enabled
    if pii_config.enabled:
        mask_result = mask_text(chunk_preview)
        masked_preview = mask_result.masked_text
        token_mapping = mask_result.token_mapping
    else:
        masked_preview = chunk_preview
        token_mapping = None

    prompt = f"""Document: {document_name} ({document_type})
Chunk content:
{masked_preview}

Provide 1-2 sentences of context for this chunk."""

    llm = get_llm_client()
    response = llm.complete(prompt)
    context = response.text.strip()

    # Unmask context if PII masking was applied
    if pii_config.enabled and token_mapping and token_mapping.mappings:
        unmask_result = unmask_text(context, token_mapping)
        context = unmask_result.unmasked_text

    # Prepend context to original (unmasked) text
    enhanced_text = f"{context}\n\n{node.text}"
    node.text = enhanced_text
    return node
```

### 4. `services/rag_server/services/session_titles.py`

Modify `generate_ai_title()` to mask user message:

```python
# Add imports at top
from infrastructure.pii.service import mask_text
from infrastructure.pii.config import get_pii_config

# Modify generate_ai_title()
def generate_ai_title(first_user_message: str, max_length: int = 40) -> str:
    """Generate AI title with PII masking."""

    pii_config = get_pii_config()
    message_for_llm = first_user_message[:500]

    # Mask message if PII masking enabled
    if pii_config.enabled:
        mask_result = mask_text(message_for_llm)
        message_for_llm = mask_result.masked_text

    prompt = f"""Generate a very short (3-6 words) title for this chat.
User message: {message_for_llm}
Title:"""

    llm = get_llm_client()
    response = llm.complete(prompt)
    title = response.text.strip()

    # No unmasking needed - title should not contain PII
    # Truncate if needed
    if len(title) > max_length:
        title = title[:max_length-3] + "..."

    return title
```

### 5. `config/models.yml.example`

Add PII configuration section:

```yaml
# ... existing config ...

# PII Masking Configuration (optional)
# Enable to anonymize data sent to cloud LLM providers
pii:
  enabled: false
  entities:
    - PERSON
    - EMAIL_ADDRESS
    - PHONE_NUMBER
    - CREDIT_CARD
    - US_SSN
    - IBAN_CODE
    - IP_ADDRESS
  token_format: "[[[{entity_type}_{index}]]]"
  score_threshold: 0.5
  language: en
  validation:
    enabled: true
    max_retries: 2
    alert_on_failure: true
  output_guardrails:
    enabled: true
    block_on_detection: false
  audit:
    enabled: true
    log_level: INFO
```

### 6. `services/rag_server/pyproject.toml`

Add Presidio dependencies:

```toml
dependencies = [
    # ... existing dependencies ...
    "presidio-analyzer>=2.2.0",
    "presidio-anonymizer>=2.2.0",
]
```

## Testing Strategy

### Unit Tests (`tests/test_pii_masking.py`)

```python
import pytest
from infrastructure.pii.service import PIIMaskingService, TokenMapping
from infrastructure.pii.config import PIIConfig


@pytest.fixture
def pii_service():
    config = PIIConfig(enabled=True)
    return PIIMaskingService(config)


def test_mask_basic_pii(pii_service):
    """Test masking of common PII types."""
    text = "Contact John Smith at john@example.com or 555-123-4567"
    result = pii_service.mask(text)

    assert "[[[PERSON_0]]]" in result.masked_text
    assert "[[[EMAIL_ADDRESS_0]]]" in result.masked_text
    assert "[[[PHONE_NUMBER_0]]]" in result.masked_text
    assert "John Smith" not in result.masked_text


def test_unmask_roundtrip(pii_service):
    """Test that mask/unmask preserves original text."""
    original = "Hello John Smith, your SSN is 123-45-6789"
    mask_result = pii_service.mask(original)
    unmask_result = pii_service.unmask(mask_result.masked_text, mask_result.token_mapping)

    assert unmask_result.unmasked_text == original
    assert unmask_result.validation_passed


def test_token_validation_detects_alterations(pii_service):
    """Test that validation catches modified tokens."""
    text = "Hello John Smith"
    mask_result = pii_service.mask(text)

    # Simulate LLM altering the token
    altered_response = mask_result.masked_text.replace("[[[", "[[")

    valid, altered = pii_service.validate_tokens_preserved(
        mask_result.token_mapping, altered_response
    )
    assert not valid
    assert len(altered) > 0


def test_fuzzy_recovery(pii_service):
    """Test fuzzy recovery of altered tokens."""
    text = "Hello John Smith"
    mask_result = pii_service.mask(text)

    # Simulate common LLM alterations
    altered = mask_result.masked_text.replace("[[[PERSON_0]]]", "PERSON_0")
    recovered = pii_service.attempt_fuzzy_recovery(altered, mask_result.token_mapping)

    assert "John Smith" in recovered


def test_output_guardrail(pii_service):
    """Test that guardrail detects leaked PII."""
    # Response that accidentally contains PII
    response = "The user John Smith can be reached at john@example.com"
    leaked = pii_service.scan_for_leaked_pii(response)

    assert len(leaked) > 0


def test_disabled_passthrough():
    """Test that disabled PII service is transparent."""
    config = PIIConfig(enabled=False)
    service = PIIMaskingService(config)

    text = "Hello John Smith at john@example.com"
    result = service.mask(text)

    assert result.masked_text == text
    assert len(result.entities_found) == 0


def test_consistent_token_mapping(pii_service):
    """Test that same entity gets same token when reusing mapping."""
    text1 = "Hello John Smith"
    text2 = "Goodbye John Smith"

    result1 = pii_service.mask(text1)
    result2 = pii_service.mask(text2, existing_mapping=result1.token_mapping)

    # Same person should have same token
    assert "[[[PERSON_0]]]" in result1.masked_text
    assert "[[[PERSON_0]]]" in result2.masked_text
```

### Integration Tests (`tests/integration/test_pii_integration.py`)

```python
import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.integration
def test_rag_query_with_pii():
    """Test complete RAG flow with PII in query."""
    # Setup: Enable PII masking
    # Query: "What is John Smith's email?"
    # Verify: Query sent to LLM has [[[PERSON_0]]] instead of "John Smith"
    # Verify: Response has original name restored
    pass


@pytest.mark.integration
def test_contextual_retrieval_masks_chunks():
    """Test that document chunks are masked during contextual retrieval."""
    # Setup: Enable PII masking + contextual retrieval
    # Upload: Document containing PII
    # Verify: Chunks sent to LLM for context are masked
    # Verify: Stored chunks have original PII (masking is only for LLM call)
    pass
```

## Verification Steps

1. **Unit tests pass**:
   ```bash
   cd services/rag_server
   .venv/bin/pytest tests/test_pii_masking.py -v
   ```

2. **Manual integration test**:
   ```bash
   # Enable PII in config/models.yml
   # Start services
   docker compose up -d

   # Query with PII
   curl -X POST http://localhost:8001/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is John Smith email address?"}'

   # Check logs for PII_AUDIT entries
   docker compose logs rag-server | grep PII_AUDIT
   ```

3. **Verify masking works**:
   - Set `LOG_LEVEL=DEBUG`
   - Query with PII
   - Check logs show masked query sent to LLM
   - Check response has original values restored

4. **Verify audit logging**:
   ```bash
   docker compose logs rag-server | grep PII_AUDIT | head -5
   # Should see MASK and UNMASK operations logged
   ```

## Implementation Order

### Phase 1: Core Infrastructure
1. Create `infrastructure/pii/` directory
2. Implement `config.py` with `PIIConfig`
3. Implement `audit.py` with `PIIAuditLogger`
4. Implement `service.py` with `PIIMaskingService`
5. Create `__init__.py` with exports
6. Add Presidio dependencies to `pyproject.toml`

### Phase 2: Configuration Integration
1. Add `PIIConfig` to `ModelsConfig` in `models_config.py`
2. Add PII section to `config/models.yml.example`
3. Write unit tests for PII service

### Phase 3: Pipeline Integration
1. Modify `pipelines/inference.py` to mask queries and unmask responses
2. Modify `pipelines/ingestion.py` to mask chunks during contextual retrieval
3. Modify `services/session_titles.py` to mask user messages
4. Write integration tests

### Phase 4: Documentation
1. Update `DEVELOPMENT.md` with PII Masking section
2. Update `CLAUDE.md` if needed
3. Test full flow end-to-end

## References

- [Microsoft Presidio Documentation](https://microsoft.github.io/presidio/)
- [LlamaIndex PII Masking](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/PII/)
- [Presidio Supported Entities](https://microsoft.github.io/presidio/supported_entities/)
- [Enterprise PII Best Practices](https://www.elastic.co/search-labs/blog/rag-security-masking-pii)
