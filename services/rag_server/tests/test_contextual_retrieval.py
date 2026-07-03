import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_index.core.schema import TextNode

from pipelines.ingestion import add_contextual_retrieval, add_contextual_prefix_to_chunk_async


def _make_nodes(n):
    return [TextNode(text=f"chunk {i}") for i in range(n)]


def test_add_contextual_retrieval_skipped_when_disabled():
    nodes = _make_nodes(3)

    with patch("pipelines.ingestion.get_ingestion_config", return_value={"contextual_retrieval_enabled": False}):
        result = add_contextual_retrieval(nodes, "/tmp/doc.txt")

    assert result is nodes


def test_add_contextual_retrieval_runs_concurrently_and_preserves_order():
    nodes = _make_nodes(20)

    mock_llm = MagicMock()

    async def fake_acomplete(prompt):
        response = MagicMock()
        # Encode which chunk this is via the prompt content isn't reliable;
        # instead vary latency to prove concurrency, and check order via node identity.
        await asyncio.sleep(0.01)
        response.text = "context"
        return response

    mock_llm.acomplete = fake_acomplete

    with patch("pipelines.ingestion.get_ingestion_config", return_value={"contextual_retrieval_enabled": True}), \
         patch("pipelines.ingestion.get_llm_client", return_value=mock_llm), \
         patch("pipelines.ingestion.get_models_config") as mock_get_config:
        mock_get_config.return_value.retrieval.contextual_concurrency = 5

        import time
        start = time.time()
        result = add_contextual_retrieval(nodes, "/tmp/doc.txt")
        elapsed = time.time() - start

    # 20 chunks at concurrency 5 with 0.01s latency each should take ~4 batches (~0.04s),
    # not 20 sequential calls (~0.2s).
    assert elapsed < 0.15
    assert len(result) == 20
    for i, node in enumerate(result):
        assert node.text.startswith("context")
        assert f"chunk {i}" in node.text


def test_add_contextual_prefix_to_chunk_async_falls_back_on_error():
    node = TextNode(text="original text")
    mock_llm = MagicMock()
    mock_llm.acomplete = AsyncMock(side_effect=Exception("LLM unavailable"))

    with patch("pipelines.ingestion.get_llm_client", return_value=mock_llm):
        result = asyncio.run(add_contextual_prefix_to_chunk_async(node, "doc.txt", ".txt"))

    assert result.text == "original text"
