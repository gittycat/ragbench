"""End-to-end PII masking test against a real cloud LLM.

Gated behind --run-eval (needs ANTHROPIC_API_KEY) rather than --run-integration:
it doesn't touch ChromaDB/Postgres, only the mask -> real cloud LLM -> unmask
round trip that Task 2.3 exists to protect.
"""

import os

import pytest

from infrastructure.config.models_config import PiiConfig
from infrastructure.llm.config import LLMConfig, LLMProvider
from infrastructure.llm.factory import create_llm_client
from infrastructure.pii.service import PIIMaskingService

pytestmark = pytest.mark.eval


@pytest.fixture
def anthropic_llm():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return create_llm_client(
        LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-20250514", api_key=api_key)
    )


def test_masked_query_survives_real_cloud_llm_roundtrip(anthropic_llm):
    """Prove mask -> cloud LLM -> unmask is correct end-to-end against a real provider:
    the LLM never sees "John Smith", but the caller gets the real name back."""
    service = PIIMaskingService(PiiConfig(enabled=True))
    query = "My name is John Smith. Reply with only my name, nothing else."

    mask_result = service.mask(query)
    assert "John Smith" not in mask_result.masked_text
    assert "[[[PERSON_0]]]" in mask_result.masked_text

    response_text = anthropic_llm.complete(mask_result.masked_text).text

    valid, _ = service.validate_tokens_preserved(mask_result.token_mapping, response_text)
    if not valid:
        response_text = service.attempt_fuzzy_recovery(response_text, mask_result.token_mapping)

    unmask_result = service.unmask(response_text, mask_result.token_mapping)

    assert "John Smith" in unmask_result.unmasked_text
    assert "[[[" not in unmask_result.unmasked_text
