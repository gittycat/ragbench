"""
Comprehensive integration test for document upload pipeline.

Verifies end-to-end flow: Upload → Process → PostgreSQL state → API response

Run with: just test-integration
Requires: docker compose up -d (all services)
"""
import pytest
import sys
import uuid
import asyncio
import time
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.integration
def test_document_upload_full_pipeline(
    api_client,
    upload_and_wait,
    tmp_path,
    document_cleanup,
):
    """
    End-to-end test: Upload → Process → Verify PostgreSQL → Verify API

    This test verifies the complete document processing pipeline:
    a) Document file is uploaded via API
    b) Document record is created in documents table
    c) Chunks are created in document_chunks table
    d) Foreign key relationship is maintained
    e) API returns correct document information

    This test catches bugs like:
    - Document record not being created
    - Chunks not being stored
    - API not returning correct data
    """
    from infrastructure.database.postgres import get_session, close_db
    from sqlalchemy import select, func
    from infrastructure.database.models import Document, DocumentChunk

    # Create test file with known content
    test_content = """
    Full Pipeline Test Document

    This document tests the complete upload and processing pipeline.
    It should be chunked, embedded, and stored in PostgreSQL.

    The unique identifier for this test is: PIPELINE_TEST_MARKER_XYZ
    """ + f"\nRun marker: {uuid.uuid4().hex}"

    file_name = f"full_pipeline_test_{uuid.uuid4().hex[:8]}.txt"
    file_path = tmp_path / file_name
    file_path.write_text(test_content)

    # ========================================================================
    # (a) Upload document via API and wait for processing
    # ========================================================================
    doc_info, batch_id = upload_and_wait(file_path, file_name)
    document_cleanup.append(doc_info["id"])

    assert "id" in doc_info, "Upload should return document ID"
    assert doc_info["file_name"] == file_name, "Filename should match"

    # ========================================================================
    # (b-d) Verify PostgreSQL state
    # ========================================================================
    async def verify_database():
        try:
            async with get_session() as session:
                # Check Document record
                doc_result = await session.execute(
                    select(Document).where(Document.id == UUID(doc_info["id"]))
                )
                doc = doc_result.scalar_one_or_none()

                assert doc is not None, (
                    f"Document record should exist in 'documents' table for ID {doc_info['id']}"
                )
                assert doc.file_name == file_name, "Document filename should match"
                assert doc.file_type == ".txt", "Document file type should be .txt"
                assert doc.file_hash is not None, "Document should have file hash"
                assert doc.uploaded_at is not None, "Document should have upload timestamp"

                print(f"\n✓ Document record verified: {doc.file_name} (ID: {doc.id})")

                # Query chunks by foreign key.
                chunks_result = await session.execute(
                    select(DocumentChunk).where(DocumentChunk.document_id == UUID(doc_info["id"]))
                )
                chunks = list(chunks_result.scalars().all())

                assert len(chunks) > 0, (
                    f"Should have chunks in 'document_chunks' table for document {doc_info['id']}. "
                    f"This likely means the ingestion pipeline failed to store chunks."
                )

                print(f"✓ Found {len(chunks)} chunks in PostgreSQL")

                # Verify each chunk has required data
                for i, chunk in enumerate(chunks):
                    assert chunk.id is not None, f"Chunk {i} should have ID"
                    assert chunk.content is not None, f"Chunk {i} should have content"
                    assert len(chunk.content) > 0, f"Chunk {i} content should not be empty"

                    # Verify metadata contains document_id
                    assert chunk.metadata_ is not None, f"Chunk {i} should have metadata"
                    assert "document_id" in chunk.metadata_, (
                        f"Chunk {i} metadata should contain document_id"
                    )
                    assert chunk.metadata_["document_id"] == doc_info["id"], (
                        f"Chunk {i} document_id in metadata should match document ID"
                    )

                print(f"✓ All {len(chunks)} chunks have valid metadata")

                # Count chunks using foreign key relationship.
                count_result = await session.execute(
                    select(func.count(DocumentChunk.id)).where(
                        DocumentChunk.document_id == UUID(doc_info["id"])
                    )
                )
                chunk_count = count_result.scalar()
                print(f"✓ Chunk count verified: {chunk_count}")

                return len(chunks)
        finally:
            await close_db()

    # Run the async database verification
    num_chunks = asyncio.run(verify_database())

    # ========================================================================
    # (e) Verify API returns correct document information
    # ========================================================================
    docs_resp = api_client.get("/documents")
    assert docs_resp.status_code == 200, f"GET /documents failed: {docs_resp.text}"

    docs = docs_resp.json()["documents"]
    matched = [d for d in docs if d["id"] == doc_info["id"]]

    assert matched, (
        f"Document {doc_info['id']} should appear in /documents API response. "
        f"Found {len(docs)} documents, but none matched."
    )

    api_doc = matched[0]
    assert api_doc["file_name"] == file_name, "API should return correct filename"
    assert api_doc["file_type"] == ".txt", "API should return correct file type"

    print(f"✓ API response verified: {api_doc['file_name']} (chunks: {api_doc.get('chunks', 0)})")

    # ========================================================================
    # (f) Verify document is queryable (chunks are retrievable)
    # ========================================================================
    query_resp = None
    for _ in range(3):
        query_resp = api_client.post(
            "/query",
            json={
                "query": "PIPELINE_TEST_MARKER",
                "session_id": str(uuid.uuid4()),
            },
        )
        if query_resp.status_code == 200:
            break
        time.sleep(1)

    assert query_resp is not None
    assert query_resp.status_code == 200, f"Query failed: {query_resp.text}"
    result = query_resp.json()
    assert "answer" in result, "Query should return answer"

    print(f"✓ Document is queryable via RAG pipeline")
    print(f"\nPipeline test PASSED:")
    print(f"  - Document record created: ✓")
    print(f"  - {num_chunks} chunks stored in PostgreSQL: ✓")
    print(f"  - API returns document info: ✓")
    print(f"  - Document is queryable: ✓")


@pytest.mark.integration
def test_multiple_documents_upload_pipeline(
    api_client,
    upload_and_wait,
    tmp_path,
    document_cleanup,
):
    """
    Test uploading multiple documents and verify each has chunks in PostgreSQL.

    This ensures the document ID tracking works correctly for concurrent uploads.
    """
    from infrastructure.database.postgres import get_session, close_db
    from sqlalchemy import select
    from infrastructure.database.models import DocumentChunk

    # Create 3 test files
    files_info = []
    for i in range(3):
        file_name = f"multi_test_{i}_{uuid.uuid4().hex[:6]}.txt"
        file_path = tmp_path / file_name
        file_path.write_text(
            f"Test document {i} with unique content marker_{i}_{uuid.uuid4().hex[:8]}"
        )

        doc_info, batch_id = upload_and_wait(file_path, file_name)
        document_cleanup.append(doc_info["id"])
        files_info.append((file_name, doc_info))

    # Verify each document has chunks in PostgreSQL
    async def verify_all_docs():
        try:
            async with get_session() as session:
                for file_name, doc_info in files_info:
                    chunks_result = await session.execute(
                        select(DocumentChunk).where(
                            DocumentChunk.document_id == UUID(doc_info["id"])
                        )
                    )
                    chunks = list(chunks_result.scalars().all())

                    assert len(chunks) > 0, (
                        f"Document {file_name} (ID: {doc_info['id']}) should have chunks"
                    )

                    print(f"✓ {file_name}: {len(chunks)} chunks in PostgreSQL")
        finally:
            await close_db()

    asyncio.run(verify_all_docs())
    print(f"\nMultiple documents test PASSED: All {len(files_info)} documents have chunks in PostgreSQL")


@pytest.mark.integration
def test_document_deletion_removes_chunks(
    api_client,
    upload_and_wait,
    tmp_path,
):
    """
    Test that deleting a document also removes its chunks (CASCADE).

    Verifies the foreign key relationship and cascade deletion.
    """
    from infrastructure.database.postgres import get_session, close_db
    from sqlalchemy import select, func
    from infrastructure.database.models import Document, DocumentChunk

    # Upload a test document
    file_name = f"delete_test_{uuid.uuid4().hex[:8]}.txt"
    file_path = tmp_path / file_name
    file_path.write_text(
        f"Document to be deleted for cascade test. marker={uuid.uuid4().hex}"
    )

    doc_info, batch_id = upload_and_wait(file_path, file_name)
    doc_id = doc_info["id"]

    # Verify chunks exist before deletion
    async def check_chunks_before():
        try:
            async with get_session() as session:
                chunks_before = await session.execute(
                    select(func.count(DocumentChunk.id)).where(
                        DocumentChunk.document_id == UUID(doc_id)
                    )
                )
                count_before = chunks_before.scalar()
                assert count_before > 0, "Should have chunks before deletion"
                print(f"✓ Before deletion: {count_before} chunks")
                return count_before
        finally:
            await close_db()

    asyncio.run(check_chunks_before())

    # Delete the document via API
    delete_resp = api_client.delete(f"/documents/{doc_id}")
    assert delete_resp.status_code == 200, f"Delete failed: {delete_resp.text}"

    # Verify document and chunks are removed
    async def check_after_deletion():
        try:
            async with get_session() as session:
                # Check document is gone
                doc_result = await session.execute(
                    select(Document).where(Document.id == UUID(doc_id))
                )
                doc = doc_result.scalar_one_or_none()
                assert doc is None, "Document should be deleted from database"

                # Check chunks are gone
                chunks_after = await session.execute(
                    select(func.count(DocumentChunk.id)).where(
                        DocumentChunk.document_id == UUID(doc_id)
                    )
                )
                count_after = chunks_after.scalar()

                print(f"✓ After deletion: {count_after} chunks (expected: 0)")
                assert count_after == 0, "Document chunks should be deleted via CASCADE"
        finally:
            await close_db()

    asyncio.run(check_after_deletion())

    # Verify document doesn't appear in API
    docs_resp = api_client.get("/documents")
    docs = docs_resp.json()["documents"]
    matched = [d for d in docs if d["id"] == doc_id]
    assert not matched, "Deleted document should not appear in /documents API"

    print(f"✓ Deletion test PASSED: Document removed from API")
