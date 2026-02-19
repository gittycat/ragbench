# PII Masking

Optional feature to anonymize sensitive data before sending to cloud LLM providers. Uses Microsoft Presidio for PII detection and reversible token-based masking.

**Status**: Planned (see [implementation plan](../PII_MASKING_IMPLEMENTATION_PLAN.md))

## How It Works

1. **Masking (outbound)**: PII detected via Microsoft Presidio (NER + regex), replaced with tokens like `[[[PERSON_0]]]`
2. **Token mapping**: Original values stored temporarily (session-scoped)
3. **Unmasking (inbound)**: Tokens in LLM response replaced with original values
4. **Validation**: Detects if LLM altered tokens, attempts fuzzy recovery
5. **Output guardrails**: Scans final response for accidentally leaked PII

## Configuration

Enable in `config.yml`:

```yaml
pii:
  enabled: true
  entities:
    - PERSON
    - EMAIL_ADDRESS
    - PHONE_NUMBER
    - CREDIT_CARD
    - US_SSN
  token_format: "[[[{entity_type}_{index}]]]"
  score_threshold: 0.5
  validation:
    enabled: true
    max_retries: 2
  output_guardrails:
    enabled: true
    block_on_detection: false
  audit:
    enabled: true
    log_level: INFO
```

## Supported Entity Types

`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, `US_SSN`, `IBAN_CODE`, `IP_ADDRESS`, `LOCATION`, `DATE_TIME`, `US_BANK_NUMBER`, `US_DRIVER_LICENSE`, `US_PASSPORT`, `MEDICAL_LICENSE`

## Audit Logging

When `audit.enabled: true`, all masking/unmasking operations are logged:

```json
{"operation": "MASK", "timestamp": "...", "context_id": "session_123", "entities_count": 3, "entity_types": ["PERSON", "EMAIL_ADDRESS"]}
{"operation": "UNMASK", "timestamp": "...", "context_id": "session_123", "tokens_found": 3, "tokens_replaced": 3, "validation_passed": true}
```

## Data Flow Points

| Path | Description |
|------|-------------|
| User queries | Query text, chat history, retrieved context sent to LLM |
| Contextual retrieval | Document chunks sent to LLM during ingestion |
| Session titles | First user message sent for title generation |
| Evaluation | Test data sent to evaluation LLM |

## Limitations

- **Token preservation**: LLMs may alter tokens (e.g., remove brackets). Validation detects this; fuzzy recovery attempts restoration
- **Performance**: Adds ~20-50ms per request for Presidio analysis
- **Not for embeddings**: Embeddings are generated from original text (stored locally in PostgreSQL)

## When to Enable

- Using cloud LLM providers (OpenAI, Anthropic, Google, etc.)
- Documents contain PII that shouldn't leave your infrastructure
- Compliance requirements (GDPR, HIPAA, etc.)

Not needed when using Ollama (local inference) as data never leaves your network.
