"""Text-based passage matching for retrieval and citation metrics."""

from evals.schemas.dataset import GoldPassage
from evals.schemas.response import RetrievedChunk


def _tokenize(text: str) -> set[str]:
    return set(text.lower().split())


def _token_overlap(text_a: str, text_b: str) -> float:
    """Jaccard similarity between two texts. Returns value in [0, 1]."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


def match_retrieved_to_gold(
    retrieved_chunks: list[RetrievedChunk],
    gold_passages: list[GoldPassage],
    threshold: float = 0.3,
) -> set[int]:
    """Return indices of retrieved chunks that match any gold passage.

    Tries exact chunk_id match first, falls back to Jaccard text overlap.
    Returns 0-based indices of matched retrieved chunks.
    """
    gold_chunk_ids = {p.chunk_id for p in gold_passages}
    matched_indices: set[int] = set()

    for i, chunk in enumerate(retrieved_chunks):
        if chunk.chunk_id in gold_chunk_ids:
            matched_indices.add(i)
            continue

        if not chunk.text:
            continue
        for gold in gold_passages:
            if gold.text and _token_overlap(chunk.text, gold.text) >= threshold:
                matched_indices.add(i)
                break

    return matched_indices
