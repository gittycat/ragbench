"""
Integration test fixtures requiring docker services.

Run with: pytest tests/integration -v --run-integration
Requires: docker compose up -d (postgres, ollama)
"""
import pytest
import os
import tempfile
import uuid
import time
import httpx
from pathlib import Path
from sqlalchemy import text


@pytest.fixture(scope="module")
def test_collection_name():
    """Generate unique collection name for test isolation."""
    return f"test_documents_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="function")
def unique_doc_id():
    """Generate unique document ID for each test."""
    return str(uuid.uuid4())


@pytest.fixture(scope="session")
def check_services(integration_env):
    """
    Verify required services are running before tests.
    Fails hard if services are unavailable.
    """
    # Initialize settings from secrets if running in Docker
    if Path("/run/secrets").exists():
        from app import settings as app_settings
        # Force reload from secrets by resetting global
        app_settings.SETTINGS = None
        from app.settings import init_settings
        init_settings()

    # Check PostgreSQL
    try:
        import asyncio
        from infrastructure.database.postgres import get_session, close_db

        async def check_db():
            try:
                async with get_session() as session:
                    await session.execute(text("SELECT 1"))
            finally:
                # Close DB pool in the same loop that opened connections.
                await close_db()

        asyncio.run(check_db())
    except Exception as e:
        pytest.fail(f"PostgreSQL not available: {e}")

    # Check Ollama
    ollama_url = integration_env.get("OLLAMA_URL", "http://localhost:11434")
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except Exception as e:
        pytest.fail(f"Ollama not available at {ollama_url}: {e}")

    # Check required models are available
    models = [m.get("name", "") for m in resp.json().get("models", [])]
    for required in ["gemma3", "nomic-embed-text"]:
        if not any(required in m for m in models):
            pytest.fail(f"Required Ollama model '{required}' not found. Available: {models}")

    # Check RAG server
    rag_url = os.getenv("RAG_SERVER_URL", "http://localhost:8001")
    try:
        health = httpx.get(f"{rag_url}/health", timeout=5.0)
        health.raise_for_status()
    except Exception as e:
        pytest.fail(f"RAG server not available at {rag_url}: {e}")

    # Wait for any pre-existing tasks to drain so they don't block test uploads
    import asyncio as _asyncio
    from infrastructure.database.postgres import get_session as _get_session, close_db as _close_db

    async def _drain_queue():
        try:
            deadline = time.time() + 600  # 10 min max wait
            while time.time() < deadline:
                async with _get_session() as session:
                    result = await session.execute(text(
                        "SELECT count(*) FROM job_tasks "
                        "WHERE status IN ('pending', 'in_progress')"
                    ))
                    pending = result.scalar()
                if pending == 0:
                    break
                print(f"Waiting for {pending} task(s) in queue to drain...")
                await _asyncio.sleep(5)
            else:
                print("WARNING: Queue drain timed out after 600s, proceeding anyway")
        finally:
            await _close_db()

    _asyncio.run(_drain_queue())

    return True


@pytest.fixture
def sample_pdf(tmp_path):
    """
    Create a simple test PDF with known content.
    Uses fpdf2 to generate a valid PDF document.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    # Add content that's easy to verify in retrieval
    unique_marker = uuid.uuid4().hex[:8]
    content = """
    RAG System Test Document

    This document tests the document processing pipeline.
    It contains information about vector databases and embeddings.

    Key Concepts:
    1. PostgreSQL with pgvector stores embeddings.
    2. Ollama provides local LLM inference capabilities.
    3. pg_search provides BM25 full-text search.
    4. Hybrid search combines dense and sparse retrieval methods.

    The unique identifier for this test is: TESTID_XYZ789_""" + unique_marker + """

    This content should be retrievable via semantic search.
    """

    for line in content.strip().split("\n"):
        pdf.cell(0, 10, line.strip(), ln=True)

    pdf_path = tmp_path / "test_document.pdf"
    pdf.output(str(pdf_path))

    return pdf_path


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a simple test text file."""
    unique_marker = uuid.uuid4().hex[:8]
    content = """
    Test Document for RAG Pipeline

    This is a test document containing information about machine learning.
    The document discusses neural networks and deep learning concepts.

    Key Topics:
    - Embeddings are vector representations of text
    - Transformers use attention mechanisms
    - RAG combines retrieval with generation

    Unique test identifier: UNIQUE_TEXT_12345_""" + unique_marker + """
    """

    text_path = tmp_path / "test_document.txt"
    text_path.write_text(content)

    return text_path


@pytest.fixture
def corrupted_pdf(tmp_path):
    """Create an invalid/corrupted PDF file."""
    pdf_path = tmp_path / "corrupted.pdf"
    # Write invalid PDF content
    pdf_path.write_bytes(b"%PDF-1.4\nThis is not a valid PDF structure\n%%EOF")
    return pdf_path


@pytest.fixture
def large_text_file(tmp_path):
    """Create a multi-chunk text file to test chunking (generates ~5 chunks)."""
    content = (
        "This is paragraph number {paragraph}. "
        "It contains test content for chunking. "
    ) * 10

    paragraphs = [content.format(paragraph=i) for i in range(10)]
    full_content = "\n\n".join(paragraphs)

    text_path = tmp_path / "large_document.txt"
    text_path.write_text(full_content)

    return text_path


@pytest.fixture
def clean_test_database(integration_env, check_services):
    """
    Provide a clean database state for testing.
    Cleans up documents after test completes.
    """
    import asyncio
    from infrastructure.database.postgres import get_session, close_db
    from infrastructure.database import documents as db_docs

    # Track created documents for cleanup
    created_doc_ids = []

    yield {
        "doc_ids": created_doc_ids,
    }

    # Cleanup: delete test documents
    async def cleanup():
        try:
            async with get_session() as session:
                for doc_id in created_doc_ids:
                    try:
                        await db_docs.delete_document(session, uuid.UUID(doc_id))
                    except Exception:
                        pass
        finally:
            await close_db()

    asyncio.run(cleanup())


@pytest.fixture(scope="session")
def large_public_markdown(tmp_path_factory):
    """
    Download a large public markdown document for realistic testing.
    Uses Anthropic's Claude documentation as a comprehensive test document.
    """
    tmp_dir = tmp_path_factory.mktemp("downloads")
    doc_path = tmp_dir / "claude_docs.md"

    # Download Anthropic's Claude API documentation (large, well-structured markdown)
    url = "https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/README.md"

    try:
        response = httpx.get(url, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        doc_path.write_bytes(response.content)

        # Verify it's substantial (should be > 10KB)
        size_kb = doc_path.stat().st_size / 1024
        if size_kb < 10:
            pytest.skip(f"Downloaded document too small ({size_kb:.1f}KB), may not be valid")

        return doc_path
    except Exception as e:
        pytest.skip(f"Failed to download test document: {e}")


@pytest.fixture(scope="session")
def large_public_pdf(tmp_path_factory):
    """
    Download a large public PDF document for realistic testing.
    Uses a research paper from arXiv.
    """
    tmp_dir = tmp_path_factory.mktemp("downloads")
    pdf_path = tmp_dir / "research_paper.pdf"

    # Download "Attention Is All You Need" paper (Transformers paper, ~15 pages)
    url = "https://arxiv.org/pdf/1706.03762.pdf"

    try:
        response = httpx.get(url, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        pdf_path.write_bytes(response.content)

        # Verify it's a valid PDF and substantial
        size_kb = pdf_path.stat().st_size / 1024
        if size_kb < 50:
            pytest.skip(f"Downloaded PDF too small ({size_kb:.1f}KB), may not be valid")

        # Check PDF header
        with open(pdf_path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                pytest.skip("Downloaded file is not a valid PDF")

        return pdf_path
    except Exception as e:
        pytest.skip(f"Failed to download test PDF: {e}")


@pytest.fixture(scope="session")
def rag_server_url():
    """Get RAG server URL for API tests."""
    return os.getenv("RAG_SERVER_URL", "http://localhost:8001")


@pytest.fixture
def wait_for_task():
    """
    Factory fixture for waiting on async task completion.
    Returns a function that polls task status.
    """
    def _wait_for_task(rag_server_url: str, batch_id: str, timeout: int = 300):
        """
        Wait for batch processing to complete.

        Args:
            rag_server_url: Base URL of RAG server
            batch_id: Batch ID returned from upload
            timeout: Maximum seconds to wait

        Returns:
            dict: Final task status

        Raises:
            TimeoutError: If tasks don't complete in time
        """
        start = time.time()
        while time.time() - start < timeout:
            resp = httpx.get(f"{rag_server_url}/tasks/{batch_id}/status", timeout=10.0)
            if resp.status_code in (404, 500):
                # Batch/status may be transiently unavailable while workers commit progress.
                time.sleep(1)
                continue
            assert resp.status_code == 200, (
                f"Status request failed for batch {batch_id}: "
                f"{resp.status_code} {resp.text[:200]}"
            )

            status = resp.json()
            if "total" not in status:
                time.sleep(1)
                continue

            if status.get("completed", 0) == status.get("total", 0):
                return status

            # Check for errors
            tasks = status.get("tasks", {})
            for task_id, task_info in tasks.items():
                if task_info.get("status") == "error":
                    return status

            time.sleep(2)

        raise TimeoutError(f"Tasks did not complete within {timeout}s")

    return _wait_for_task


@pytest.fixture(scope="session")
def api_client(check_services, rag_server_url):
    """Session-scoped httpx client for API tests."""
    with httpx.Client(base_url=rag_server_url, timeout=60.0) as client:
        yield client


@pytest.fixture
def upload_and_wait(api_client, wait_for_task, rag_server_url):
    """
    Factory fixture: upload file(s) and wait for processing.
    Returns (doc_info_dict, batch_id) for the first uploaded file.
    """
    def _upload_and_wait(file_path: Path, filename: str | None = None, timeout: int = 300):
        fname = filename or file_path.name
        with open(file_path, "rb") as f:
            resp = api_client.post(
                "/upload",
                files={"files": (fname, f, "application/octet-stream")},
            )
        assert resp.status_code == 200, f"Upload failed: {resp.text}"
        batch_id = resp.json()["batch_id"]

        final_status = wait_for_task(rag_server_url, batch_id, timeout=timeout)
        tasks = final_status.get("tasks", {})
        errors = [t for t in tasks.values() if t.get("status") == "error"]
        assert not errors, f"Upload task(s) errored: {errors}"

        # Find the document in the list
        docs_resp = api_client.get("/documents")
        assert docs_resp.status_code == 200
        docs = docs_resp.json().get("documents", [])
        # Worker uses task_id as document_id for stable tracking.
        task_ids = list(tasks.keys())
        assert task_ids, f"No tasks found in final status for batch {batch_id}"
        expected_doc_id = task_ids[0]
        matched = [d for d in docs if d.get("id") == expected_doc_id]
        assert matched, (
            f"Uploaded file '{fname}' not found in /documents with expected id {expected_doc_id}"
        )

        return matched[0], batch_id

    return _upload_and_wait


@pytest.fixture
def document_cleanup(api_client):
    """Collects document IDs and deletes them on teardown."""
    doc_ids: list[str] = []
    yield doc_ids
    for doc_id in doc_ids:
        try:
            api_client.delete(f"/documents/{doc_id}")
        except Exception:
            pass


@pytest.fixture
def session_cleanup(api_client):
    """Collects session IDs and deletes them on teardown."""
    session_ids: list[str] = []
    yield session_ids
    for sid in session_ids:
        try:
            api_client.delete(f"/chat/sessions/{sid}")
        except Exception:
            pass


@pytest.fixture
def test_document(api_client, upload_and_wait, tmp_path):
    """
    Create, upload, and index a text file with a unique marker.
    Yields dict with doc_id, file_name, marker, chunks, batch_id.
    Deletes the document on teardown.
    """
    marker = f"MARKER_{uuid.uuid4().hex}"
    content = (
        f"Integration test document.\n\n"
        f"This file contains a unique marker for retrieval verification: {marker}\n\n"
        f"The marker has no semantic meaning and exists purely for round-trip testing.\n"
    )
    file_name = f"test_{uuid.uuid4().hex[:8]}.txt"
    file_path = tmp_path / file_name
    file_path.write_text(content)

    doc_info, batch_id = upload_and_wait(file_path, file_name)

    yield {
        "doc_id": doc_info["id"],
        "file_name": file_name,
        "marker": marker,
        "chunks": doc_info.get("chunks", 0),
        "batch_id": batch_id,
    }

    # Cleanup
    try:
        api_client.delete(f"/documents/{doc_info['id']}")
    except Exception:
        pass
