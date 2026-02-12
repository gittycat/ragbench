"""
Infrastructure smoke tests - verify services, database schema, and extensions.

Fast tests (~5s each). No uploads or mutations.

Run with: pytest tests/integration/test_infrastructure.py -v --run-integration
"""
import os
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.integration
class TestServiceConnectivity:
    """Verify all required services are reachable and responding."""

    def test_rag_server_health(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_rag_server_config(self, api_client):
        resp = api_client.get("/config")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["max_upload_size_mb"], int)
        assert data["max_upload_size_mb"] > 0

    def test_models_info(self, api_client):
        resp = api_client.get("/models/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("llm_model"), "llm_model should be non-empty"
        assert data.get("embedding_model"), "embedding_model should be non-empty"

    def test_postgres_extensions(self, integration_env, check_services):
        """Verify pg_textsearch extension is installed."""
        import asyncio
        from sqlalchemy import text
        from infrastructure.database.postgres import get_session

        async def _check():
            async with get_session() as session:
                result = await session.execute(
                    text("SELECT extname FROM pg_extension")
                )
                extensions = [row[0] for row in result.fetchall()]
                return extensions

        extensions = asyncio.run(_check())
        for ext in ["pg_textsearch"]:
            assert ext in extensions, (
                f"Extension '{ext}' not found. Installed: {extensions}"
            )

    def test_ollama_models_available(self, integration_env):
        """Verify required Ollama models are pulled."""
        import httpx

        ollama_url = integration_env.get("OLLAMA_URL", "http://localhost:11434")
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]

        for required in ["gemma3", "nomic-embed-text"]:
            assert any(required in m for m in models), (
                f"Required model '{required}' not found. Available: {models}"
            )


@pytest.mark.integration
class TestDatabaseSchema:
    """Verify database tables, queues, and indexes exist."""

    def test_required_tables_exist(self, integration_env, check_services):
        import asyncio
        from sqlalchemy import text
        from infrastructure.database.postgres import get_session

        required_tables = [
            "documents",
            "document_chunks",
            "chat_sessions",
            "chat_messages",
            "job_batches",
            "job_tasks",
        ]

        async def _check():
            async with get_session() as session:
                result = await session.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public'"
                    )
                )
                return [row[0] for row in result.fetchall()]

        tables = asyncio.run(_check())
        for table in required_tables:
            assert table in tables, (
                f"Table '{table}' not found. Existing: {tables}"
            )

    def test_claimable_tasks_index_exists(self, integration_env, check_services):
        """Verify partial index for SKIP LOCKED task claiming exists."""
        import asyncio
        from sqlalchemy import text
        from infrastructure.database.postgres import get_session

        async def _check():
            async with get_session() as session:
                result = await session.execute(
                    text(
                        "SELECT indexname FROM pg_indexes "
                        "WHERE tablename = 'job_tasks' "
                        "AND indexname = 'idx_tasks_claimable'"
                    )
                )
                return [row[0] for row in result.fetchall()]

        indexes = asyncio.run(_check())
        assert "idx_tasks_claimable" in indexes, (
            f"Partial index 'idx_tasks_claimable' not found on job_tasks"
        )

    def test_vector_index_exists(self, integration_env, check_services):
        """Verify HNSW index exists on document_chunks."""
        import asyncio
        from sqlalchemy import text
        from infrastructure.database.postgres import get_session

        async def _check():
            async with get_session() as session:
                result = await session.execute(
                    text(
                        "SELECT indexname, indexdef FROM pg_indexes "
                        "WHERE tablename = 'data_document_chunks'"
                    )
                )
                return result.fetchall()

        indexes = asyncio.run(_check())
        hnsw_indexes = [
            idx for idx in indexes if "hnsw" in (idx[1] or "").lower()
        ]
        assert hnsw_indexes, (
            f"No HNSW index found on document_chunks. Indexes: "
            f"{[(i[0], i[1][:80]) for i in indexes]}"
        )
