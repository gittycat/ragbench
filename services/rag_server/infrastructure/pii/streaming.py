"""Sentence-buffered unmasking for streaming responses.

Token-by-token streaming can split a `[[[PERSON_0]]]` token across two SSE
events, so per-token unmasking is unsafe. Instead we buffer masked tokens
until a sentence boundary (". " or "\\n") is seen, then unmask and emit the
completed chunk. This trades per-token latency for correctness — only used
when pii.enabled is true.
"""

import logging
from typing import AsyncIterator, Iterator

from .service import PIIMaskingService, TokenMapping

logger = logging.getLogger(__name__)


def _find_boundary(buffer: str) -> int | None:
    """Return the end index of the earliest sentence boundary in buffer, or None."""
    candidates = []
    idx = buffer.find(". ")
    if idx != -1:
        candidates.append(idx + 2)
    idx = buffer.find("\n")
    if idx != -1:
        candidates.append(idx + 1)
    return min(candidates) if candidates else None


def _unmask_chunk(service: PIIMaskingService, chunk: str, token_mapping: TokenMapping, context_id: str | None) -> str:
    valid, _ = service.validate_tokens_preserved(token_mapping, chunk)
    if not valid:
        chunk = service.attempt_fuzzy_recovery(chunk, token_mapping)
    return service.unmask(chunk, token_mapping, context_id=context_id).unmasked_text


def buffer_and_unmask_stream(
    tokens: Iterator[str],
    service: PIIMaskingService,
    token_mapping: TokenMapping,
    context_id: str | None = None,
) -> Iterator[str]:
    """Wrap a masked-token stream, yielding unmasked text in sentence-sized flushes."""
    buffer = ""
    for token in tokens:
        buffer += token
        boundary = _find_boundary(buffer)
        while boundary is not None:
            chunk, buffer = buffer[:boundary], buffer[boundary:]
            yield _unmask_chunk(service, chunk, token_mapping, context_id)
            boundary = _find_boundary(buffer)
    if buffer:
        yield _unmask_chunk(service, buffer, token_mapping, context_id)


async def buffer_and_unmask_stream_async(
    tokens: AsyncIterator[str],
    service: PIIMaskingService,
    token_mapping: TokenMapping,
    context_id: str | None = None,
) -> AsyncIterator[str]:
    """Async variant of buffer_and_unmask_stream."""
    buffer = ""
    async for token in tokens:
        buffer += token
        boundary = _find_boundary(buffer)
        while boundary is not None:
            chunk, buffer = buffer[:boundary], buffer[boundary:]
            yield _unmask_chunk(service, chunk, token_mapping, context_id)
            boundary = _find_boundary(buffer)
    if buffer:
        yield _unmask_chunk(service, buffer, token_mapping, context_id)
