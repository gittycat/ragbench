"""
Document processing job for pgmq worker.

Handles async document processing: chunking, embedding, indexing.
Progress tracking via PostgreSQL job_batches/job_tasks tables.
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*validate_default.*")

import asyncio
import logging
import shutil
import time
from pathlib import Path
from uuid import UUID

from pipelines.ingestion import ingest_document
from infrastructure.search.vector_store import get_vector_index
from infrastructure.database.postgres import get_session
from infrastructure.database.repositories.jobs import JobRepository

# Persistent document storage path
DOCUMENT_STORAGE_PATH = Path("/data/documents")

logger = logging.getLogger(__name__)


async def process_document_async(file_path: str, filename: str, batch_id: str, task_id: str) -> dict:
    """
    Process a document and index it in PostgreSQL (pgvector).

    Args:
        file_path: Path to temporary file in /tmp/shared
        filename: Original filename
        batch_id: Batch ID for progress tracking
        task_id: Task ID for this specific file

    Returns:
        dict with document_id and chunks count
    """
    task_start = time.time()

    logger.info(f"[TASK {task_id}] ========== Starting document processing: {filename} ==========")

    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(
                f"Temporary upload file not found: {file_path}. "
                "The shared upload volume may have been reset or the file was already cleaned up. "
                "Please re-upload the document."
            )

        # Generate document ID (use task_id for consistency)
        doc_id = task_id
        logger.info(f"[TASK {task_id}] Using document ID: {doc_id}")

        # Update progress: processing
        await _update_task_status(task_id, "processing")

        # Get vector index
        index = get_vector_index()

        # Create progress callback for embedding tracking
        def embedding_progress(current: int, total: int):
            # Run async operations in the current event loop
            # Note: This is called from sync context (LlamaIndex), so we need asyncio.run
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, create a task
                if current == 1:
                    asyncio.create_task(_set_task_total_chunks(task_id, total))
                asyncio.create_task(_increment_task_chunk_progress(task_id))
            except RuntimeError:
                # No running loop - create one (shouldn't happen in async function)
                if current == 1:
                    asyncio.run(_set_task_total_chunks(task_id, total))
                asyncio.run(_increment_task_chunk_progress(task_id))

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
        await _complete_task(task_id)

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

        # Update error status
        await _fail_task(task_id, user_friendly_error)

        # Clean up temp file on failure
        _cleanup_temp_file(file_path, task_id)

        raise


def process_document(file_path: str, filename: str, batch_id: str, task_id: str) -> dict:
    """
    Sync wrapper for process_document_async. Used by pgmq worker.

    Note: Uses asyncio.run() which may produce event loop warnings when mixing
    with sync pgmq library, but these are generally harmless.
    """
    return asyncio.run(process_document_async(file_path, filename, batch_id, task_id))


async def _update_task_status(task_id: str, status: str) -> None:
    """Update task status in PostgreSQL."""
    async with get_session() as session:
        repo = JobRepository(session)
        await repo.update_task_status(UUID(task_id), status)


async def _set_task_total_chunks(task_id: str, total: int) -> None:
    """Set total chunks for task."""
    async with get_session() as session:
        repo = JobRepository(session)
        await repo.set_task_total_chunks(UUID(task_id), total)


async def _increment_task_chunk_progress(task_id: str) -> None:
    """Increment chunk progress for task."""
    async with get_session() as session:
        repo = JobRepository(session)
        await repo.increment_task_chunk_progress(UUID(task_id))


async def _complete_task(task_id: str) -> None:
    """Mark task as completed."""
    async with get_session() as session:
        repo = JobRepository(session)
        await repo.complete_task(UUID(task_id))


async def _fail_task(task_id: str, error_message: str) -> None:
    """Mark task as failed."""
    async with get_session() as session:
        repo = JobRepository(session)
        await repo.fail_task(UUID(task_id), error_message)


def _cleanup_temp_file(file_path: str, task_id: str):
    """Clean up temporary file after processing."""
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger.info(f"[TASK {task_id}] Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"[TASK {task_id}] Could not delete temp file {file_path}: {e}")


# Backward compatibility alias
process_document_task = process_document
