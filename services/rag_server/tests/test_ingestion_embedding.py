import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_index.core.schema import TextNode

from pipelines.ingestion import embed_and_index_chunks, INGEST_BATCH_SIZE


def _make_nodes(n):
    return [TextNode(text=f"chunk {i}", id_=f"doc-chunk-{i}") for i in range(n)]


def test_embed_and_index_chunks_batches_embedding_calls():
    nodes = _make_nodes(70)  # 3 batches of 32/32/6
    index = MagicMock()

    mock_embed_model = MagicMock()
    mock_embed_model.get_text_embedding_batch.side_effect = lambda texts: [[0.1] * 8 for _ in texts]

    with patch("pipelines.ingestion.Settings") as mock_settings:
        mock_settings.embed_model = mock_embed_model
        embed_and_index_chunks(index, nodes)

    assert mock_embed_model.get_text_embedding_batch.call_count == 3
    assert index.insert_nodes.call_count == 3
    total_inserted = sum(len(call.args[0]) for call in index.insert_nodes.call_args_list)
    assert total_inserted == 70


def test_embed_and_index_chunks_sets_node_embeddings():
    nodes = _make_nodes(2)
    index = MagicMock()

    mock_embed_model = MagicMock()
    mock_embed_model.get_text_embedding_batch.return_value = [[0.1] * 8, [0.2] * 8]

    with patch("pipelines.ingestion.Settings") as mock_settings:
        mock_settings.embed_model = mock_embed_model
        embed_and_index_chunks(index, nodes)

    assert nodes[0].embedding == [0.1] * 8
    assert nodes[1].embedding == [0.2] * 8


def test_embed_and_index_chunks_calls_progress_callback_per_batch():
    nodes = _make_nodes(40)  # 2 batches
    index = MagicMock()
    progress_calls = []

    mock_embed_model = MagicMock()
    mock_embed_model.get_text_embedding_batch.side_effect = lambda texts: [[0.1] * 8 for _ in texts]

    with patch("pipelines.ingestion.Settings") as mock_settings:
        mock_settings.embed_model = mock_embed_model
        embed_and_index_chunks(index, nodes, progress_callback=lambda done, total: progress_calls.append((done, total)))

    assert progress_calls == [(32, 40), (40, 40)]


def test_embed_and_index_chunks_retries_on_connection_error():
    nodes = _make_nodes(1)
    index = MagicMock()

    mock_embed_model = MagicMock()
    mock_embed_model.get_text_embedding_batch.side_effect = [
        ConnectionError("connection refused"),
        [[0.1] * 8],
    ]

    with patch("pipelines.ingestion.Settings") as mock_settings, \
         patch("pipelines.ingestion.time.sleep"):
        mock_settings.embed_model = mock_embed_model
        embed_and_index_chunks(index, nodes)

    assert mock_embed_model.get_text_embedding_batch.call_count == 2
    assert index.insert_nodes.call_count == 1
