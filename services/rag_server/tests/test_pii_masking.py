import pytest

from infrastructure.config.models_config import PiiConfig
from infrastructure.pii.service import PIIMaskingService, TokenMapping
from infrastructure.pii.streaming import buffer_and_unmask_stream, buffer_and_unmask_stream_async


@pytest.fixture
def pii_service():
    return PIIMaskingService(PiiConfig(enabled=True))


def test_mask_basic_pii(pii_service):
    text = "Contact John Smith at john@example.com"
    result = pii_service.mask(text)

    assert "[[[PERSON_0]]]" in result.masked_text
    assert "[[[EMAIL_ADDRESS_0]]]" in result.masked_text
    assert "John Smith" not in result.masked_text
    assert "john@example.com" not in result.masked_text


def test_unmask_roundtrip(pii_service):
    original = "Hello John Smith, contact them at john@example.com"
    mask_result = pii_service.mask(original)
    unmask_result = pii_service.unmask(mask_result.masked_text, mask_result.token_mapping)

    assert unmask_result.unmasked_text == original
    assert unmask_result.validation_passed


def test_token_validation_detects_alterations(pii_service):
    mask_result = pii_service.mask("Hello John Smith")

    altered_response = mask_result.masked_text.replace("[[[", "[[")

    valid, altered = pii_service.validate_tokens_preserved(mask_result.token_mapping, altered_response)
    assert not valid
    assert len(altered) > 0


def test_fuzzy_recovery(pii_service):
    mask_result = pii_service.mask("Hello John Smith")

    altered = mask_result.masked_text.replace("[[[PERSON_0]]]", "PERSON_0")
    recovered = pii_service.attempt_fuzzy_recovery(altered, mask_result.token_mapping)

    assert "John Smith" in recovered


def test_output_guardrail_detects_leak(pii_service):
    response = "The user John Smith can be reached at john@example.com"
    leaked = pii_service.scan_for_leaked_pii(response)

    assert len(leaked) > 0


def test_output_guardrail_disabled_returns_empty():
    config = PiiConfig(enabled=True)
    config.output_guardrails.enabled = False
    service = PIIMaskingService(config)

    leaked = service.scan_for_leaked_pii("John Smith's email is john@example.com")
    assert leaked == []


def test_disabled_passthrough():
    service = PIIMaskingService(PiiConfig(enabled=False))

    text = "Hello John Smith at john@example.com"
    result = service.mask(text)

    assert result.masked_text == text
    assert result.entities_found == []


def test_consistent_token_mapping_across_calls(pii_service):
    result1 = pii_service.mask("Hello John Smith")
    result2 = pii_service.mask("Goodbye John Smith", existing_mapping=result1.token_mapping)

    assert "[[[PERSON_0]]]" in result1.masked_text
    assert "[[[PERSON_0]]]" in result2.masked_text
    # Second call reuses the same mapping object and doesn't mint a second PERSON token
    assert result1.token_mapping is result2.token_mapping
    assert len(result2.token_mapping.mappings["PERSON"]) == 1


def test_new_entity_appended_to_existing_mapping(pii_service):
    result1 = pii_service.mask("Hello John Smith")
    result2 = pii_service.mask("Hello Jane Doe", existing_mapping=result1.token_mapping)

    assert "[[[PERSON_0]]]" in result1.masked_text
    assert "[[[PERSON_1]]]" in result2.masked_text


def test_unmask_reports_missing_tokens(pii_service):
    mask_result = pii_service.mask("Hello John Smith")
    # Simulate the LLM dropping the token entirely
    unmask_result = pii_service.unmask("Hello there!", mask_result.token_mapping)

    assert not unmask_result.validation_passed
    assert "[[[PERSON_0]]]" in unmask_result.tokens_missing


class TestStreamingBuffer:
    def test_buffers_until_sentence_boundary(self, pii_service):
        mask_result = pii_service.mask("Hello John Smith")
        token_mapping = mask_result.token_mapping

        # Token split across two raw deltas, as an LLM stream would emit it
        tokens = iter(["The person is [[[PERSON", "_0]]]. Nice to meet them.\n"])
        chunks = list(buffer_and_unmask_stream(tokens, pii_service, token_mapping))

        full_text = "".join(chunks)
        assert "John Smith" in full_text
        assert "[[[" not in full_text

    def test_flushes_remainder_without_trailing_boundary(self, pii_service):
        mask_result = pii_service.mask("Hello John Smith")
        token_mapping = mask_result.token_mapping

        tokens = iter(["No sentence boundary here [[[PERSON_0]]]"])
        chunks = list(buffer_and_unmask_stream(tokens, pii_service, token_mapping))

        assert "".join(chunks) == "No sentence boundary here John Smith"

    @pytest.mark.asyncio
    async def test_async_variant_matches_sync(self, pii_service):
        mask_result = pii_service.mask("Hello John Smith")
        token_mapping = mask_result.token_mapping

        async def tokens():
            for t in ["The person is [[[PERSON", "_0]]]. Done.\n"]:
                yield t

        chunks = [c async for c in buffer_and_unmask_stream_async(tokens(), pii_service, token_mapping)]
        full_text = "".join(chunks)
        assert "John Smith" in full_text
        assert "[[[" not in full_text


def test_pii_disabled_by_default():
    assert PiiConfig().enabled is False
