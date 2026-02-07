"""
Integration tests for document management endpoints.

Run with: pytest tests/integration/test_api_documents.py -v --run-integration
Requires: docker compose up -d (rag-server, pgmq-worker, postgres, ollama)
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.integration
def test_check_duplicates_after_upload(
    api_client,
    sample_text_file,
    upload_and_wait,
    document_cleanup,
):
    """
    Upload -> process -> duplicate check reports existing file.

    Validates /documents/check-duplicates against a real indexed document.
    """
    from pipelines.ingestion import compute_file_hash

    # Upload file via fixture
    doc_info, batch_id = upload_and_wait(sample_text_file)
    document_cleanup.append(doc_info["id"])

    # Check duplicates
    file_hash = compute_file_hash(sample_text_file)
    payload = {
        "files": [
            {
                "filename": sample_text_file.name,
                "size": sample_text_file.stat().st_size,
                "hash": file_hash,
            }
        ]
    }

    resp = api_client.post("/documents/check-duplicates", json=payload)

    assert resp.status_code == 200, f"Duplicate check failed: {resp.text}"
    results = resp.json()["results"]
    assert sample_text_file.name in results

    result = results[sample_text_file.name]
    assert result["exists"] is True
    assert result.get("document_id") == doc_info["id"]
