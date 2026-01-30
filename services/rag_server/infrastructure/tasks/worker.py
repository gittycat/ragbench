"""
RQ job for asynchronous document processing.

Converted from Celery task to plain function.
Retry logic is handled by RQ's Retry class at enqueue time.
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*validate_default.*")

import logging
import shutil
import time
import uuid
from pathlib import Path

from rq import get_current_job

from pipelines.ingestion import ingest_document
from infrastructure.database.chroma import get_or_create_collection
from infrastructure.tasks.progress import (
    update_task_progress,
    set_task_total_chunks,
    increment_task_chunk_progress
)

# Persistent document storage path
DOCUMENT_STORAGE_PATH = Path("/data/documents")

logger = logging.getLogger(__name__)


def process_document_task(file_path: str, filename: str, batch_id: str) -> dict:
    """
    Process a document and index it in ChromaDB.

    Args:
        file_path: Path to temporary file in /tmp/shared
        filename: Original filename
        batch_id: Batch ID for progress tracking

    Returns:
        dict with document_id and chunks count

    Note:
        Retry configuration is set at enqueue time via Retry(max=3, interval=[5, 15, 60])
    """
    job = get_current_job()
    task_id = job.id if job else str(uuid.uuid4())
    task_start = time.time()

    logger.info(f"[TASK {task_id}] ========== Starting document processing: {filename} ==========")

    try:
        # Generate document ID
        doc_id = str(uuid.uuid4())
        logger.info(f"[TASK {task_id}] Generated document ID: {doc_id}")

        # Update progress: processing
        update_task_progress(batch_id, task_id, "processing", {
            "filename": filename,
            "message": "Processing document..."
        })

        # Get ChromaDB index
        index = get_or_create_collection()

        # Create progress callback for embedding tracking
        def embedding_progress(current: int, total: int):
            if current == 1:
                set_task_total_chunks(batch_id, task_id, total)
            increment_task_chunk_progress(batch_id, task_id)
            update_task_progress(batch_id, task_id, "processing", {
                "filename": filename,
                "message": f"Embedding chunk {current}/{total}..."
            })

        # Run ingestion pipeline
        logger.info(f"[TASK {task_id}] Running ingestion pipeline...")
        result = ingest_document(
            file_path=file_path,
            index=index,
            document_id=doc_id,
            filename=filename,
            progress_callback=embedding_progress
        )

        # Store original document for download functionality
        logger.info(f"[TASK {task_id}] Storing original document for downloads...")
        try:
            doc_storage_dir = DOCUMENT_STORAGE_PATH / doc_id
            doc_storage_dir.mkdir(parents=True, exist_ok=True)
            dest_path = doc_storage_dir / filename
            shutil.copy2(file_path, dest_path)
            logger.info(f"[TASK {task_id}] Document stored at {dest_path}")
        except Exception as e:
            logger.warning(f"[TASK {task_id}] Failed to store document for downloads: {e}")

        # Update progress: completed
        update_task_progress(batch_id, task_id, "completed", {
            "filename": filename,
            "document_id": result['document_id'],
            "chunks": result['chunks'],
            "message": "Successfully indexed"
        })

        task_duration = time.time() - task_start
        logger.info(f"[TASK {task_id}] ========== Task completed in {task_duration:.2f}s ==========")

        # Clean up temp file on success
        _cleanup_temp_file(file_path, task_id)

        return result

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"[TASK {task_id}] Error processing {filename}: {str(e)}")
        logger.error(f"[TASK {task_id}] Traceback:\n{error_trace}")

        # Create user-friendly error message
        user_friendly_error = str(e).replace(file_path, filename)

        # Check if this is the final retry attempt
        job = get_current_job()
        retries_left = job.retries_left if job and hasattr(job, 'retries_left') else 0
        is_final_attempt = retries_left == 0

        if is_final_attempt:
            # Update error status on final failure
            update_task_progress(batch_id, task_id, "error", {
                "filename": filename,
                "error": user_friendly_error,
                "message": f"Error: {user_friendly_error}"
            })
            # Clean up temp file on final failure
            _cleanup_temp_file(file_path, task_id)
        else:
            logger.info(f"[TASK {task_id}] Will retry ({retries_left} retries left)")

        raise


def _cleanup_temp_file(file_path: str, task_id: str):
    """Clean up temporary file after processing."""
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger.info(f"[TASK {task_id}] Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"[TASK {task_id}] Could not delete temp file {file_path}: {e}")
