import asyncio
import sys
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.tasks.worker import process_document_async


def test_ingest_document_runs_off_the_event_loop_thread(tmp_path):
    """ingest_document() must run in an executor thread, not block the event loop."""
    main_thread_id = threading.get_ident()
    observed_thread_id = {}

    def fake_ingest_document(**kwargs):
        observed_thread_id["id"] = threading.get_ident()
        progress_callback = kwargs.get("progress_callback")
        if progress_callback:
            progress_callback(1, 1)
        return {"chunks_data": [], "document_id": kwargs["document_id"]}

    test_file = tmp_path / "doc.txt"
    test_file.write_text("hello")

    async def fake_session_ctx():
        session = AsyncMock()
        return session

    class _FakeSessionCtx:
        async def __aenter__(self):
            return AsyncMock()

        async def __aexit__(self, *args):
            return False

    with patch("infrastructure.tasks.worker.get_vector_index", return_value=MagicMock()), \
         patch("infrastructure.tasks.worker.extract_file_metadata", return_value={}), \
         patch("infrastructure.tasks.worker.get_session", return_value=_FakeSessionCtx()), \
         patch("infrastructure.tasks.worker.db_docs") as mock_db_docs, \
         patch("infrastructure.tasks.worker.db_jobs") as mock_db_jobs, \
         patch("infrastructure.tasks.worker.ingest_document", side_effect=fake_ingest_document), \
         patch("infrastructure.tasks.worker._complete_task", new=AsyncMock()), \
         patch("infrastructure.tasks.worker._cleanup_temp_file"):
        mock_db_docs.delete_document = AsyncMock()
        mock_db_docs.create_document = AsyncMock()
        mock_db_docs.add_chunks = AsyncMock()
        mock_db_jobs.set_task_total_chunks = AsyncMock()
        mock_db_jobs.increment_task_chunk_progress = AsyncMock()

        result = asyncio.run(
            process_document_async(
                file_path=str(test_file),
                filename="doc.txt",
                batch_id="batch-1",
                task_id="11111111-1111-1111-1111-111111111111",
            )
        )

    assert observed_thread_id["id"] != main_thread_id
    assert result["document_id"] == "11111111-1111-1111-1111-111111111111"
