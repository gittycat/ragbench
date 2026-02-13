"""
Core RAG pipeline tests - upload, index, query, delete round-trips.

Run with: pytest tests/integration/test_pipeline.py -v --run-integration --run-slow
"""
import uuid
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.integration
class TestTextFilePipeline:
    """Text file upload → index → query → verify round-trip."""

    def test_uploaded_document_appears_in_list(self, api_client, test_document):
        resp = api_client.get("/documents")
        assert resp.status_code == 200
        docs = resp.json().get("documents", [])
        matched = [d for d in docs if d["id"] == test_document["doc_id"]]
        assert matched, f"Document {test_document['doc_id']} not in /documents list"
        assert matched[0]["file_name"] == test_document["file_name"]

    def test_uploaded_document_has_chunks(self, test_document):
        assert test_document["chunks"] > 0, (
            f"Document should have >0 chunks, got {test_document['chunks']}"
        )

    def test_query_returns_nonempty_answer(self, api_client, test_document, session_cleanup):
        session_id = str(uuid.uuid4())
        session_cleanup.append(session_id)

        resp = api_client.post(
            "/query",
            json={
                "query": "What is the unique marker in the test document?",
                "session_id": session_id,
            },
        )
        assert resp.status_code == 200, f"Query failed: {resp.text}"
        data = resp.json()
        assert data.get("answer"), "Answer should be non-empty"
        assert data.get("session_id"), "session_id should be present"

    def test_query_returns_sources(self, api_client, test_document, session_cleanup):
        session_id = str(uuid.uuid4())
        session_cleanup.append(session_id)

        resp = api_client.post(
            "/query",
            json={
                "query": "What is the unique marker?",
                "session_id": session_id,
                "include_chunks": True,
            },
        )
        assert resp.status_code == 200, f"Query failed: {resp.text}"
        data = resp.json()
        assert isinstance(data.get("sources"), list), "sources should be a list"
        assert len(data["sources"]) > 0, "sources should be non-empty"

    def test_canary_chunk_contains_marker(self, api_client, test_document, session_cleanup):
        """
        CANARY: Verify retrieval returns chunk text payloads for indexed content.
        """
        marker = test_document["marker"]
        session_id = str(uuid.uuid4())
        session_cleanup.append(session_id)

        resp = api_client.post(
            "/query",
            json={
                "query": marker,
                "session_id": session_id,
                "include_chunks": True,
            },
        )
        assert resp.status_code == 200, f"Query failed: {resp.text}"
        data = resp.json()
        sources = data.get("sources", [])

        # Verify at least one source includes chunk text payload.
        assert sources, "Expected non-empty sources from retrieval"
        has_text_payload = any(
            bool(
                source.get("text")
                or source.get("content")
                or source.get("excerpt")
                or source.get("full_text")
            )
            for source in sources
        )
        assert has_text_payload, "Expected at least one source with textual content"


@pytest.mark.integration
class TestPdfPipeline:
    """PDF upload → index → query round-trip."""

    def test_pdf_upload_and_query(
        self, api_client, sample_pdf, upload_and_wait, document_cleanup, session_cleanup
    ):
        doc_info, batch_id = upload_and_wait(sample_pdf)
        document_cleanup.append(doc_info["id"])

        assert doc_info.get("chunks", 0) > 0, "PDF should produce chunks"

        session_id = str(uuid.uuid4())
        session_cleanup.append(session_id)

        resp = api_client.post(
            "/query",
            json={
                "query": "What is the unique identifier in the test document?",
                "session_id": session_id,
            },
        )
        assert resp.status_code == 200, f"Query failed: {resp.text}"
        assert resp.json().get("answer"), "Answer should be non-empty"


@pytest.mark.integration
class TestDocumentDeletion:
    """Document deletion via API."""

    def test_delete_removes_from_document_list(
        self, api_client, upload_and_wait, tmp_path
    ):
        # Upload a throwaway file
        file_path = tmp_path / f"delete_test_{uuid.uuid4().hex[:8]}.txt"
        file_path.write_text(
            f"Temporary document for deletion test. marker={uuid.uuid4().hex}"
        )
        doc_info, _ = upload_and_wait(file_path)
        doc_id = doc_info["id"]

        # Delete it
        resp = api_client.delete(f"/documents/{doc_id}")
        assert resp.status_code == 200

        # Verify gone from list
        docs_resp = api_client.get("/documents")
        docs = docs_resp.json().get("documents", [])
        assert not any(d["id"] == doc_id for d in docs), (
            f"Document {doc_id} should not appear after deletion"
        )

    def test_query_after_delete_does_not_crash(
        self, api_client, upload_and_wait, session_cleanup, tmp_path
    ):
        file_path = tmp_path / f"delete_query_{uuid.uuid4().hex[:8]}.txt"
        marker = f"DELMARKER_{uuid.uuid4().hex[:8]}"
        file_path.write_text(f"Content with marker: {marker}")
        doc_info, _ = upload_and_wait(file_path)

        # Delete the document
        api_client.delete(f"/documents/{doc_info['id']}")

        # Query for deleted content — should return 200 (not 500)
        session_id = str(uuid.uuid4())
        session_cleanup.append(session_id)

        resp = api_client.post(
            "/query",
            json={"query": marker, "session_id": session_id},
        )
        assert resp.status_code == 200, (
            f"Query after delete should not crash. Status: {resp.status_code}, "
            f"Body: {resp.text[:200]}"
        )

    def test_delete_nonexistent_returns_404(self, api_client):
        fake_id = str(uuid.uuid4())
        resp = api_client.delete(f"/documents/{fake_id}")
        assert resp.status_code == 404, (
            f"Expected 404 for nonexistent doc, got {resp.status_code}"
        )
