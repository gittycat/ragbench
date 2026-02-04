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
    Fails fast if services are unavailable.
    """
    # Check PostgreSQL
    try:
        import asyncio
        from infrastructure.database.postgres import init_db, close_db
        asyncio.run(init_db())
        asyncio.run(close_db())
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")

    # Check Ollama
    ollama_url = integration_env.get("OLLAMA_URL", "http://localhost:11434")
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except Exception as e:
        pytest.skip(f"Ollama not available at {ollama_url}: {e}")

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
    content = """
    RAG System Test Document

    This document tests the document processing pipeline.
    It contains information about vector databases and embeddings.

    Key Concepts:
    1. PostgreSQL with pgvector stores embeddings.
    2. Ollama provides local LLM inference capabilities.
    3. pg_search provides BM25 full-text search.
    4. Hybrid search combines dense and sparse retrieval methods.

    The unique identifier for this test is: TESTID_XYZ789

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
    content = """
    Test Document for RAG Pipeline

    This is a test document containing information about machine learning.
    The document discusses neural networks and deep learning concepts.

    Key Topics:
    - Embeddings are vector representations of text
    - Transformers use attention mechanisms
    - RAG combines retrieval with generation

    Unique test identifier: UNIQUE_TEXT_12345
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
    """Create a large text file to test chunking."""
    content = "This is paragraph number {}. It contains test content for chunking. " * 50

    paragraphs = [content.format(i) for i in range(100)]
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
    from infrastructure.database.postgres import get_session
    from infrastructure.database.repositories.documents import DocumentRepository

    # Track created documents for cleanup
    created_doc_ids = []

    yield {
        "doc_ids": created_doc_ids,
    }

    # Cleanup: delete test documents
    async def cleanup():
        async with get_session() as session:
            repo = DocumentRepository(session)
            for doc_id in created_doc_ids:
                try:
                    await repo.delete_document_with_chunks(uuid.UUID(doc_id))
                except Exception:
                    pass

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


@pytest.fixture
def rag_server_url():
    """Get RAG server URL for API tests."""
    return os.getenv("RAG_SERVER_URL", "http://localhost:8001")


@pytest.fixture
def wait_for_task():
    """
    Factory fixture for waiting on async task completion.
    Returns a function that polls task status.
    """
    def _wait_for_task(rag_server_url: str, batch_id: str, timeout: int = 120):
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
            status = resp.json()

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
