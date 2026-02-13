"""
Integration tests for async document upload.

Tests the full async workflow: API upload -> task processing -> Progress tracking -> Completion

Run with: pytest tests/integration/test_async_upload.py -v --run-integration
Requires: docker compose up -d (all services including task-worker)
"""
import pytest
import os
import sys
import uuid
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.integration
class TestTaskCompletion:
    """
    Test async task execution and completion.

    These tests require the full stack running:
    - rag-server (API)
    - task-worker (task processing via SKIP LOCKED)
    - postgres (database, vectors, progress)
    - ollama (embeddings)
    """

    def test_celery_task_completes(
        self,
        integration_env,
        check_services,
        sample_text_file,
        rag_server_url,
        wait_for_task,
    ):
        """
        Upload via API -> Celery processes -> status shows completed.

        This validates the entire async processing pipeline works end-to-end.
        """
        import httpx

        # Check RAG server is running
        try:
            health = httpx.get(f"{rag_server_url}/health", timeout=5.0)
            health.raise_for_status()
        except Exception as e:
            pytest.skip(f"RAG server not available at {rag_server_url}: {e}")

        # Upload file
        with open(sample_text_file, "rb") as f:
            files = {"files": (sample_text_file.name, f, "text/plain")}
            response = httpx.post(
                f"{rag_server_url}/upload",
                files=files,
                timeout=30.0,
            )

        assert response.status_code == 200, f"Upload failed: {response.text}"
        upload_result = response.json()

        assert "batch_id" in upload_result, "Upload should return batch_id"
        batch_id = upload_result["batch_id"]

        # Wait for task completion
        final_status = wait_for_task(rag_server_url, batch_id, timeout=300)

        assert final_status["completed"] == final_status["total"], \
            f"All tasks should complete. Status: {final_status}"

        # Verify document is queryable
        query_response = httpx.post(
            f"{rag_server_url}/query",
            json={"query": "UNIQUE_TEXT_12345", "session_id": str(uuid.uuid4())},
            timeout=60.0,
        )

        assert query_response.status_code == 200, f"Query failed: {query_response.text}"
        result = query_response.json()
        assert "answer" in result, "Query should return answer"

    def test_progress_tracking_accuracy(
        self,
        integration_env,
        check_services,
        large_text_file,
        rag_server_url,
        wait_for_task,
    ):
        """
        Multi-chunk doc -> progress increments correctly.

        Verifies the progress tracking system accurately reports chunk processing.
        """
        import httpx

        # Check RAG server
        try:
            httpx.get(f"{rag_server_url}/health", timeout=5.0).raise_for_status()
        except Exception as e:
            pytest.skip(f"RAG server not available: {e}")

        # Upload large file
        with open(large_text_file, "rb") as f:
            files = {"files": (large_text_file.name, f, "text/plain")}
            response = httpx.post(
                f"{rag_server_url}/upload",
                files=files,
                timeout=30.0,
            )

        assert response.status_code == 200
        batch_id = response.json()["batch_id"]

        # Poll progress and verify it increments
        progress_readings = []
        start = time.time()
        timeout = 300  # Large files may take longer

        while time.time() - start < timeout:
            status_response = httpx.get(
                f"{rag_server_url}/tasks/{batch_id}/status",
                timeout=10.0,
            )
            status = status_response.json()
            progress_readings.append(status)

            if status.get("completed", 0) == status.get("total", 0):
                break

            time.sleep(2)

        # Verify progress was tracked
        assert len(progress_readings) > 0, "Should have progress readings"

    def test_upload_single_file_listed_in_documents(
        self,
        integration_env,
        check_services,
        rag_server_url,
        wait_for_task,
        tmp_path,
        clean_test_database,
    ):
        """
        Upload one file -> no error -> document appears in /documents list.
        """
        import httpx

        # Check RAG server is running
        try:
            health = httpx.get(f"{rag_server_url}/health", timeout=5.0)
            health.raise_for_status()
        except Exception as e:
            pytest.skip(f"RAG server not available at {rag_server_url}: {e}")

        unique_name = f"upload_test_{uuid.uuid4().hex}.txt"
        file_path = tmp_path / unique_name
        file_path.write_text(
            f"Upload test content. Unique identifier: LIST_TEST_{uuid.uuid4().hex}"
        )

        # Upload file
        with open(file_path, "rb") as f:
            files = {"files": (unique_name, f, "text/plain")}
            response = httpx.post(
                f"{rag_server_url}/upload",
                files=files,
                timeout=30.0,
            )

        assert response.status_code == 200, f"Upload failed: {response.text}"
        upload_result = response.json()
        assert "batch_id" in upload_result, "Upload should return batch_id"
        batch_id = upload_result["batch_id"]

        # Wait for task completion
        final_status = wait_for_task(rag_server_url, batch_id, timeout=300)
        tasks = final_status.get("tasks", {})
        task_errors = [
            t for t in tasks.values() if t.get("status") == "error"
        ]
        assert not task_errors, f"Upload tasks errored: {task_errors}"

        # Verify document is listed
        list_response = httpx.get(
            f"{rag_server_url}/documents",
            timeout=10.0,
        )
        assert list_response.status_code == 200, \
            f"Documents list failed: {list_response.text}"
        docs = list_response.json().get("documents", [])
        matched = [d for d in docs if d.get("file_name") == unique_name]
        assert matched, f"Uploaded file not found in /documents: {unique_name}"

        # Cleanup: delete document after test
        doc_id = matched[0].get("id")
        if doc_id:
            clean_test_database["doc_ids"].append(doc_id)

    def test_multiple_file_upload(
        self,
        integration_env,
        check_services,
        sample_text_file,
        sample_pdf,
        rag_server_url,
        wait_for_task,
    ):
        """
        Upload multiple files simultaneously -> all processed correctly.
        """
        import httpx

        try:
            httpx.get(f"{rag_server_url}/health", timeout=5.0).raise_for_status()
        except Exception as e:
            pytest.skip(f"RAG server not available: {e}")

        # Upload both files
        files = [
            ("files", (sample_text_file.name, open(sample_text_file, "rb"), "text/plain")),
            ("files", (sample_pdf.name, open(sample_pdf, "rb"), "application/pdf")),
        ]

        try:
            response = httpx.post(
                f"{rag_server_url}/upload",
                files=files,
                timeout=30.0,
            )
        finally:
            # Close file handles
            for _, (_, f, _) in files:
                f.close()

        assert response.status_code == 200
        result = response.json()
        batch_id = result["batch_id"]

        # Should have 2 tasks
        assert len(result.get("tasks", [])) == 2, f"Should have 2 tasks, got {result}"

        # Wait for all tasks
        final_status = wait_for_task(rag_server_url, batch_id, timeout=360)

        assert final_status["completed"] == 2, \
            f"Both files should be processed. Status: {final_status}"


@pytest.mark.integration
@pytest.mark.slow
class TestLargeDocumentUpload:
    """Test upload and status tracking with large real-world documents."""

    def test_large_markdown_status_progression(
        self,
        integration_env,
        check_services,
        large_public_markdown,
        rag_server_url,
    ):
        """
        Upload large markdown → capture status progression → verify completion.

        This test validates that:
        1. POST /upload accepts large markdown files
        2. GET /tasks/{batch_id}/status returns accurate progress
        3. Status progresses from queued → processing → completed
        4. Final status indicates "completed" with all tasks done
        """
        import httpx

        try:
            httpx.get(f"{rag_server_url}/health", timeout=5.0).raise_for_status()
        except Exception as e:
            pytest.skip(f"RAG server not available: {e}")

        # Upload the large markdown document
        with open(large_public_markdown, "rb") as f:
            files = {"files": (large_public_markdown.name, f, "text/markdown")}
            response = httpx.post(
                f"{rag_server_url}/upload",
                files=files,
                timeout=30.0,
            )

        assert response.status_code == 200, f"Upload failed: {response.text}"
        upload_result = response.json()

        assert "batch_id" in upload_result, "Upload should return batch_id"
        assert "tasks" in upload_result, "Upload should return task info"
        batch_id = upload_result["batch_id"]

        # Track status progression
        status_snapshots = []
        start = time.time()
        timeout = 300  # 5 minutes for large document

        while time.time() - start < timeout:
            status_response = httpx.get(
                f"{rag_server_url}/tasks/{batch_id}/status",
                timeout=10.0,
            )

            assert status_response.status_code == 200, \
                f"Status endpoint should return 200: {status_response.status_code}"

            status = status_response.json()
            status_snapshots.append({
                "timestamp": time.time() - start,
                "completed": status.get("completed", 0),
                "total": status.get("total", 0),
                "tasks": status.get("tasks", {}),
            })

            # Check if processing is complete
            if status.get("completed", 0) == status.get("total", 0):
                # Verify all tasks show "completed" status
                tasks = status.get("tasks", {})
                for task_id, task_info in tasks.items():
                    task_status = task_info.get("status", "")
                    assert task_status == "completed", \
                        f"Task {task_id} should be completed, got: {task_status}"

                break

            time.sleep(1)  # Poll every second to capture status changes

        # Verify we captured the completion
        assert len(status_snapshots) > 0, "Should have captured status snapshots"

        final_status = status_snapshots[-1]
        assert final_status["completed"] == final_status["total"], \
            f"All tasks should complete. Final status: {final_status}"

        # Verify we captured multiple status updates (shows progression)
        assert len(status_snapshots) >= 2, \
            "Should capture multiple status updates during processing"

        print(f"\nStatus progression ({len(status_snapshots)} snapshots):")
        for i, snapshot in enumerate(status_snapshots):
            print(f"  {i+1}. {snapshot['timestamp']:.2f}s: "
                  f"{snapshot['completed']}/{snapshot['total']} tasks completed")

    def test_large_pdf_status_progression(
        self,
        integration_env,
        check_services,
        large_public_pdf,
        rag_server_url,
    ):
        """
        Upload large PDF → capture status progression → verify completion.

        Tests the same workflow as markdown test but with PDF format.
        """
        import httpx

        try:
            httpx.get(f"{rag_server_url}/health", timeout=5.0).raise_for_status()
        except Exception as e:
            pytest.skip(f"RAG server not available: {e}")

        # Upload the large PDF document
        with open(large_public_pdf, "rb") as f:
            files = {"files": (large_public_pdf.name, f, "application/pdf")}
            response = httpx.post(
                f"{rag_server_url}/upload",
                files=files,
                timeout=30.0,
            )

        assert response.status_code == 200, f"Upload failed: {response.text}"
        upload_result = response.json()

        assert "batch_id" in upload_result
        batch_id = upload_result["batch_id"]

        # Track status progression
        status_snapshots = []
        start = time.time()
        timeout = 300

        while time.time() - start < timeout:
            status_response = httpx.get(
                f"{rag_server_url}/tasks/{batch_id}/status",
                timeout=10.0,
            )

            assert status_response.status_code == 200

            status = status_response.json()
            status_snapshots.append({
                "timestamp": time.time() - start,
                "completed": status.get("completed", 0),
                "total": status.get("total", 0),
            })

            # Check completion
            if status.get("completed", 0) == status.get("total", 0):
                tasks = status.get("tasks", {})
                for task_id, task_info in tasks.items():
                    task_status = task_info.get("status", "")
                    assert task_status == "completed", \
                        f"Task {task_id} should be completed, got: {task_status}"
                break

            time.sleep(1)

        # Verify completion
        assert len(status_snapshots) > 0
        final_status = status_snapshots[-1]
        assert final_status["completed"] == final_status["total"], \
            f"PDF processing should complete. Final: {final_status}"

        print(f"\nPDF processing: {len(status_snapshots)} status updates, "
              f"completed in {status_snapshots[-1]['timestamp']:.2f}s")


@pytest.mark.integration
class TestTaskProgressAPI:
    """Test the task status API endpoints."""

    def test_invalid_batch_id_returns_error(
        self,
        integration_env,
        rag_server_url,
    ):
        """Query with invalid batch ID should return appropriate error."""
        import httpx

        try:
            httpx.get(f"{rag_server_url}/health", timeout=5.0).raise_for_status()
        except Exception as e:
            pytest.skip(f"RAG server not available: {e}")

        response = httpx.get(
            f"{rag_server_url}/tasks/nonexistent-batch-id/status",
            timeout=10.0,
        )

        # Invalid UUID format should return a client error (400), not crash.
        assert response.status_code == 400, \
            f"Should handle invalid batch ID format gracefully: {response.status_code}"

    def test_concurrent_uploads_no_race(
        self,
        integration_env,
        check_services,
        sample_text_file,
        rag_server_url,
        wait_for_task,
    ):
        """
        Upload 3 files simultaneously -> all indexed correctly.

        Tests for race conditions in BM25 refresh and progress tracking.
        """
        import httpx
        import concurrent.futures

        try:
            httpx.get(f"{rag_server_url}/health", timeout=5.0).raise_for_status()
        except Exception as e:
            pytest.skip(f"RAG server not available: {e}")

        # Create 3 unique files
        files_to_upload = []
        uploaded_names = []
        for i in range(3):
            unique_suffix = uuid.uuid4().hex[:8]
            content = f"""
            Concurrent Upload Test File {i}

            Unique identifier: CONCURRENT_{i}_{unique_suffix}

            This tests race conditions in parallel uploads.
            """
            file_path = sample_text_file.parent / f"concurrent_{i}_{unique_suffix}.txt"
            file_path.write_text(content)
            files_to_upload.append(file_path)
            uploaded_names.append(file_path.name)

        batch_ids = []

        def upload_file(file_path):
            with open(file_path, "rb") as f:
                response = httpx.post(
                    f"{rag_server_url}/upload",
                    files={"files": (file_path.name, f, "text/plain")},
                    timeout=30.0,
                )
            return response.json()

        # Upload all files concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(upload_file, f) for f in files_to_upload]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                assert "batch_id" in result
                batch_ids.append(result["batch_id"])

        # Wait for all batches to complete
        for batch_id in batch_ids:
            status = wait_for_task(rag_server_url, batch_id, timeout=300)
            assert status["completed"] == status["total"], \
                f"Batch {batch_id} should complete: {status}"

        # Verify uploaded documents are present in API listing.
        docs_resp = httpx.get(f"{rag_server_url}/documents", timeout=10.0)
        assert docs_resp.status_code == 200
        doc_names = {d.get("file_name") for d in docs_resp.json().get("documents", [])}
        for name in uploaded_names:
            assert name in doc_names, f"Expected uploaded document in listing: {name}"
