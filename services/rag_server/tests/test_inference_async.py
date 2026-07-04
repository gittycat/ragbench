import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.inference import (
    _AsyncSafeCondensePlusContextChatEngine,
    query_rag_async,
    query_rag_stream_async,
)


def test_async_safe_chat_engine_offloads_postprocessing_to_thread():
    engine = _AsyncSafeCondensePlusContextChatEngine.__new__(_AsyncSafeCondensePlusContextChatEngine)

    mock_retriever = MagicMock()
    mock_retriever.aretrieve = AsyncMock(return_value=["node1", "node2"])
    engine._retriever = mock_retriever

    call_thread_ids = []

    def fake_postprocess(nodes, query_bundle=None):
        import threading
        call_thread_ids.append(threading.get_ident())
        return [f"processed-{n}" for n in nodes]

    mock_postprocessor = MagicMock()
    mock_postprocessor.postprocess_nodes = MagicMock(side_effect=fake_postprocess)
    engine._node_postprocessors = [mock_postprocessor]

    import threading
    main_thread_id = threading.get_ident()

    result = asyncio.run(engine._aget_nodes("what is this about?"))

    assert result == ["processed-node1", "processed-node2"]
    assert call_thread_ids[0] != main_thread_id


def test_query_rag_async_happy_path():
    mock_chat_engine = MagicMock()
    mock_response = MagicMock()
    mock_response.source_nodes = []
    mock_response.__str__.return_value = "the answer"
    mock_chat_engine.achat = AsyncMock(return_value=mock_response)

    with patch("infrastructure.search.vector_store.get_vector_index", return_value=MagicMock()), \
         patch("pipelines.inference.get_inference_config", return_value={
             "reranker_enabled": True, "reranker_model": "x", "reranker_top_n": 5,
             "retrieval_top_k": 10, "hybrid_search_enabled": True, "rrf_k": 60,
         }), \
         patch("pipelines.inference.create_chat_engine", return_value=mock_chat_engine) as mock_create, \
         patch("pipelines.inference.get_token_counts", return_value={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}), \
         patch("pipelines.inference.reset_token_counter"):
        result = asyncio.run(
            query_rag_async("what is this?", session_id="sess-1", update_session_metadata=False)
        )

    assert result["answer"] == "the answer"
    assert result["session_id"] == "sess-1"
    mock_chat_engine.achat.assert_awaited_once_with("what is this?")
    # async_safe=True must be threaded through so reranking doesn't block the loop
    assert mock_create.call_args.kwargs["async_safe"] is True


def test_query_rag_stream_async_yields_sse_events():
    mock_chat_engine = MagicMock()
    mock_streaming_response = MagicMock()
    mock_streaming_response.source_nodes = []
    mock_streaming_response.__str__.return_value = "streamed answer"

    async def fake_gen():
        for tok in ["hel", "lo"]:
            yield tok

    mock_streaming_response.async_response_gen = fake_gen
    mock_chat_engine.astream_chat = AsyncMock(return_value=mock_streaming_response)

    async def collect():
        events = []
        async for chunk in query_rag_stream_async(
            "hi", session_id="sess-2", update_session_metadata=False
        ):
            events.append(chunk)
        return events

    with patch("infrastructure.search.vector_store.get_vector_index", return_value=MagicMock()), \
         patch("pipelines.inference.get_inference_config", return_value={
             "reranker_enabled": True, "reranker_model": "x", "reranker_top_n": 5,
             "retrieval_top_k": 10, "hybrid_search_enabled": True, "rrf_k": 60,
         }), \
         patch("pipelines.inference.create_chat_engine", return_value=mock_chat_engine):
        events = asyncio.run(collect())

    assert any('"token": "hel"' in e for e in events)
    assert any('"token": "lo"' in e for e in events)
    assert any(e.startswith("event: sources") for e in events)
    assert any(e.startswith("event: done") for e in events)
