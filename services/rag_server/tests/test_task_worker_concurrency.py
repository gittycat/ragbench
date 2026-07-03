import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# task_worker.py calls initialize_settings() at import time (before conftest's
# autouse fixture runs, since imports happen at collection time). Provide fake
# secrets and make the ChromaDB dimension guard fail soft just for the import.
from pydantic import SecretStr
from app import settings as app_settings


class _FakeSettings:
    OPENAI_API_KEY = SecretStr("test-openai-key")
    ANTHROPIC_API_KEY = SecretStr("test-anthropic-key")
    GOOGLE_API_KEY = SecretStr("test-google-key")
    DEEPSEEK_API_KEY = SecretStr("test-deepseek-key")
    MOONSHOT_API_KEY = SecretStr("test-moonshot-key")
    RAG_SERVER_DB_USER = SecretStr("raguser")
    RAG_SERVER_DB_PASSWORD = SecretStr("ragpass")


app_settings.SETTINGS = _FakeSettings()

_chroma_patch = patch("infrastructure.search.vector_store.get_chroma_client", side_effect=ValueError("no chromadb in tests"))
_chroma_patch.start()
import infrastructure.tasks.task_worker  # noqa: E402  (trigger side-effecting import while patched)
_chroma_patch.stop()


def test_default_worker_concurrency_is_two(monkeypatch):
    from infrastructure.tasks import task_worker

    monkeypatch.delenv("WORKER_CONCURRENCY", raising=False)
    assert task_worker.get_worker_concurrency() == 2


def test_worker_concurrency_respects_env(monkeypatch):
    from infrastructure.tasks import task_worker

    monkeypatch.setenv("WORKER_CONCURRENCY", "5")
    assert task_worker.get_worker_concurrency() == 5


def test_worker_concurrency_capped_at_max(monkeypatch):
    from infrastructure.tasks import task_worker

    monkeypatch.setenv("WORKER_CONCURRENCY", "20")
    assert task_worker.get_worker_concurrency() == task_worker.MAX_WORKER_CONCURRENCY


def test_worker_concurrency_minimum_one(monkeypatch):
    from infrastructure.tasks import task_worker

    monkeypatch.setenv("WORKER_CONCURRENCY", "0")
    assert task_worker.get_worker_concurrency() == 1


def test_run_worker_spawns_concurrent_claim_loops(monkeypatch):
    from infrastructure.tasks import task_worker

    monkeypatch.setenv("WORKER_CONCURRENCY", "3")

    call_counts = {}

    async def fake_claim_loop(loop_id):
        call_counts[loop_id] = call_counts.get(loop_id, 0) + 1

    async def fake_check_stuck_tasks():
        await asyncio.sleep(3600)

    with patch.object(task_worker, "claim_loop", side_effect=fake_claim_loop), \
         patch.object(task_worker, "check_stuck_tasks", side_effect=fake_check_stuck_tasks):
        asyncio.run(task_worker.run_worker())

    assert set(call_counts.keys()) == {0, 1, 2}
