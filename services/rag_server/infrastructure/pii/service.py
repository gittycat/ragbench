"""PII masking service using Microsoft Presidio.

Reversible token-based masking: PII spans are detected with Presidio's
AnalyzerEngine (spaCy NER + pattern recognizers), replaced with distinctive
bracket tokens via AnonymizerEngine's custom operator, and reversed on the
way back with a plain string replace. The token<->original mapping is kept
in a TokenMapping the caller owns (session-scoped for chat, see
infrastructure/pii/postprocessor.py) so the same PERSON always gets the same
token across multiple mask() calls that share a mapping.
"""

import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import tldextract
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from .audit import PIIAuditLogger
from .config import get_pii_config
from infrastructure.config.models_config import PiiConfig

logger = logging.getLogger(__name__)

# Presidio's EmailRecognizer validates TLDs via tldextract, which by default fetches
# the public suffix list from the network on first use and caches it under ~/.cache
# (root-owned in the Docker image — see Docker Volume Permissions notes). Both are
# wrong for a privacy/offline-first product: force the snapshot bundled with the
# tldextract package, cached under a directory guaranteed writable in every tier.
tldextract.tldextract.TLD_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=str(Path(tempfile.gettempdir()) / "tldextract_cache")
)


@dataclass
class TokenMapping:
    """Reversible original<->token mapping. Structure: {entity_type: {original: token}}."""

    mappings: dict[str, dict[str, str]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def get_reverse_mapping(self) -> dict[str, str]:
        reverse: dict[str, str] = {}
        for type_mappings in self.mappings.values():
            for original, token in type_mappings.items():
                reverse[token] = original
        return reverse

    def get_expected_tokens(self) -> set[str]:
        tokens: set[str] = set()
        for type_mappings in self.mappings.values():
            tokens.update(type_mappings.values())
        return tokens


@dataclass
class MaskingResult:
    masked_text: str
    token_mapping: TokenMapping
    entities_found: list[RecognizerResult]


@dataclass
class UnmaskingResult:
    unmasked_text: str
    tokens_found: int
    tokens_replaced: int
    tokens_missing: list[str]
    validation_passed: bool


class PIILeakageError(Exception):
    """Raised when PII is detected in LLM output and blocking is enabled."""


class PIIMaskingService:
    """Mask/unmask PII in text using a shared, reversible token mapping."""

    def __init__(self, config: Optional[PiiConfig] = None):
        self.config = config or get_pii_config()
        self._analyzer: Optional[AnalyzerEngine] = None
        self._anonymizer: Optional[AnonymizerEngine] = None
        self._audit_logger: Optional[PIIAuditLogger] = None

    @property
    def analyzer(self) -> AnalyzerEngine:
        """Lazy-load Presidio's analyzer (spaCy NER + pattern recognizers)."""
        if self._analyzer is None:
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            nlp_engine = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": self.config.language, "model_name": self.config.spacy_model}],
                }
            ).create_engine()
            self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            logger.info(f"[PII] Initialized Presidio AnalyzerEngine (model={self.config.spacy_model})")
        return self._analyzer

    @property
    def anonymizer(self) -> AnonymizerEngine:
        if self._anonymizer is None:
            self._anonymizer = AnonymizerEngine()
        return self._anonymizer

    @property
    def audit_logger(self) -> PIIAuditLogger:
        if self._audit_logger is None:
            self._audit_logger = PIIAuditLogger(self.config.audit)
        return self._audit_logger

    def mask(
        self,
        text: str,
        existing_mapping: Optional[TokenMapping] = None,
        context_id: Optional[str] = None,
    ) -> MaskingResult:
        """Mask PII in text with reversible tokens, reusing existing_mapping's tokens for repeat entities."""
        if not self.config.enabled:
            return MaskingResult(masked_text=text, token_mapping=existing_mapping or TokenMapping(), entities_found=[])

        results = self.analyzer.analyze(
            text=text,
            entities=self.config.entities,
            language=self.config.language,
            score_threshold=self.config.score_threshold,
        )

        mapping = existing_mapping or TokenMapping()
        if not results:
            return MaskingResult(masked_text=text, token_mapping=mapping, entities_found=[])

        entity_counters = {entity_type: len(values) for entity_type, values in mapping.mappings.items()}

        def get_or_create_token(entity_type: str, original_value: str) -> str:
            type_mappings = mapping.mappings.setdefault(entity_type, {})
            if original_value in type_mappings:
                return type_mappings[original_value]
            index = entity_counters.get(entity_type, 0)
            token = self.config.token_format.format(entity_type=entity_type, index=index)
            type_mappings[original_value] = token
            entity_counters[entity_type] = index + 1
            return token

        operators = {
            entity_type: OperatorConfig(
                "custom", {"lambda": lambda original, et=entity_type: get_or_create_token(et, original)}
            )
            for entity_type in {r.entity_type for r in results}
        }
        masked_text = self.anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text

        if self.config.audit.enabled:
            self.audit_logger.log_mask_operation(
                context_id=context_id,
                entities_count=len(results),
                entity_types=[r.entity_type for r in results],
            )

        return MaskingResult(masked_text=masked_text, token_mapping=mapping, entities_found=results)

    def unmask(
        self,
        text: str,
        token_mapping: TokenMapping,
        context_id: Optional[str] = None,
    ) -> UnmaskingResult:
        """Replace tokens with their original values."""
        if not self.config.enabled:
            return UnmaskingResult(
                unmasked_text=text, tokens_found=0, tokens_replaced=0, tokens_missing=[], validation_passed=True
            )

        expected_tokens = token_mapping.get_expected_tokens()
        tokens_in_text = {token for token in expected_tokens if token in text}

        unmasked_text = text
        tokens_replaced = 0
        for token, original in token_mapping.get_reverse_mapping().items():
            occurrences = unmasked_text.count(token)
            if occurrences:
                unmasked_text = unmasked_text.replace(token, original)
                tokens_replaced += occurrences

        tokens_missing = [t for t in expected_tokens if t not in text]
        validation_passed = not tokens_missing

        if self.config.audit.enabled:
            self.audit_logger.log_unmask_operation(
                context_id=context_id,
                tokens_found=len(tokens_in_text),
                tokens_replaced=tokens_replaced,
                validation_passed=validation_passed,
            )

        return UnmaskingResult(
            unmasked_text=unmasked_text,
            tokens_found=len(tokens_in_text),
            tokens_replaced=tokens_replaced,
            tokens_missing=tokens_missing,
            validation_passed=validation_passed,
        )

    def validate_tokens_preserved(self, token_mapping: TokenMapping, response_text: str) -> tuple[bool, list[str]]:
        """Check whether every token minted for this mapping survived into response_text verbatim."""
        altered = [t for t in token_mapping.get_expected_tokens() if t not in response_text]
        return not altered, altered

    def attempt_fuzzy_recovery(self, text: str, token_mapping: TokenMapping) -> str:
        """Recover common LLM alterations of a token (bracket/case/separator changes) before unmasking."""
        result = text
        for token, original in token_mapping.get_reverse_mapping().items():
            if token in result:
                continue
            variants = [
                token.replace("[[[", "").replace("]]]", ""),
                token.replace("[[[", "[").replace("]]]", "]"),
                token.replace("_", " "),
                token.replace("_", "-"),
                token.lower(),
                token.upper(),
            ]
            for variant in variants:
                if variant in result:
                    result = result.replace(variant, original)
                    logger.debug(f"[PII] Fuzzy recovered: {variant} -> {original[:20]}...")
                    break
        return result

    def scan_for_leaked_pii(self, text: str, context_id: Optional[str] = None) -> list[RecognizerResult]:
        """Output guardrail: scan final (unmasked) response text for verbatim PII leaks."""
        if not self.config.output_guardrails.enabled:
            return []

        results = self.analyzer.analyze(
            text=text,
            entities=self.config.entities,
            language=self.config.language,
            score_threshold=self.config.score_threshold,
        )

        if results and self.config.audit.enabled:
            self.audit_logger.log_pii_leak_detected(
                context_id=context_id, entities_count=len(results), entity_types=[r.entity_type for r in results]
            )
        return results


_service: Optional[PIIMaskingService] = None


def get_pii_service() -> PIIMaskingService:
    global _service
    if _service is None:
        _service = PIIMaskingService()
    return _service


def reset_pii_service() -> None:
    """Reset the global PII service. Useful for testing and config reloads."""
    global _service
    _service = None


def mask_text(
    text: str, existing_mapping: Optional[TokenMapping] = None, context_id: Optional[str] = None
) -> MaskingResult:
    return get_pii_service().mask(text, existing_mapping, context_id)


def unmask_text(text: str, token_mapping: TokenMapping, context_id: Optional[str] = None) -> UnmaskingResult:
    return get_pii_service().unmask(text, token_mapping, context_id)
