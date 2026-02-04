import uuid
import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File

from fastapi.responses import FileResponse

from schemas.document import (
    DocumentListResponse,
    DeleteResponse,
    FileCheckRequest,
    FileCheckResult,
    FileCheckResponse,
    BatchUploadResponse,
    TaskInfo,
    BatchProgressResponse,
)
from pipelines.ingestion import SUPPORTED_EXTENSIONS, compute_file_hash
from infrastructure.database.postgres import get_session
from infrastructure.database.repositories.documents import DocumentRepository
from infrastructure.database.repositories.jobs import JobRepository
from infrastructure.tasks.pgmq_queue import enqueue_document_task

logger = logging.getLogger(__name__)
router = APIRouter()

# Persistent document storage path
DOCUMENT_STORAGE_PATH = Path("/data/documents")


@router.get("/documents", response_model=DocumentListResponse)
async def get_documents(
    sort_by: str = "uploaded_at",
    sort_order: str = "desc"
):
    """
    Get all documents with sorting support.

    Args:
        sort_by: Field to sort by ('name', 'chunks', 'uploaded_at'). Default: 'uploaded_at'
        sort_order: Sort order ('asc' or 'desc'). Default: 'desc' (newest first)

    Returns:
        DocumentListResponse with sorted documents list
    """
    try:
        # Validate sort parameters
        valid_sort_fields = ['file_name', 'chunks', 'uploaded_at']
        # Map 'name' to 'file_name' for backward compatibility
        if sort_by == 'name':
            sort_by = 'file_name'
        if sort_by not in valid_sort_fields:
            sort_by = 'uploaded_at'
        if sort_order not in ['asc', 'desc']:
            sort_order = 'desc'

        async with get_session() as session:
            repo = DocumentRepository(session)
            documents = await repo.list_documents(sort_by=sort_by, sort_order=sort_order)

        # Convert to response format
        formatted_docs = []
        for doc in documents:
            formatted_docs.append({
                "id": doc["document_id"],
                "file_name": doc["file_name"],
                "file_type": doc["file_type"],
                "chunks": doc["chunks"],
                "uploaded_at": doc["uploaded_at"],
                "file_size_bytes": doc["file_size_bytes"],
            })

        return DocumentListResponse(documents=formatted_docs)
    except Exception as e:
        logger.error(f"[DOCUMENTS] Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/check-duplicates", response_model=FileCheckResponse)
async def check_duplicate_documents(request: FileCheckRequest):
    """
    Check if documents with given hashes already exist in the system.
    Returns information about which files are duplicates and should be skipped.

    Args:
        request: List of files with {filename, size, hash}

    Returns:
        Dictionary mapping filenames to their duplicate status
    """
    try:
        logger.info(f"[CHECK_DUPLICATES] Checking {len(request.files)} files for duplicates")

        # Get hashes to check
        hashes_to_check = [f.hash for f in request.files if f.hash]

        async with get_session() as session:
            repo = DocumentRepository(session)
            existing = await repo.check_duplicates(hashes_to_check)

        # Build results
        formatted_results = {}
        for file_info in request.files:
            filename = file_info.filename
            file_hash = file_info.hash

            if file_hash and file_hash in existing:
                formatted_results[filename] = FileCheckResult(
                    filename=filename,
                    exists=True,
                    document_id=existing[file_hash],
                    existing_filename=filename,
                    reason="File with same content already exists"
                )
            else:
                formatted_results[filename] = FileCheckResult(
                    filename=filename,
                    exists=False,
                    document_id=None,
                    existing_filename=None,
                    reason=None
                )

        duplicates_count = sum(1 for r in formatted_results.values() if r.exists)
        logger.info(f"[CHECK_DUPLICATES] Found {duplicates_count} duplicate(s) out of {len(request.files)} files")

        return FileCheckResponse(results=formatted_results)

    except Exception as e:
        logger.error(f"[CHECK_DUPLICATES] Error checking duplicates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=BatchUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    logger.info(f"[UPLOAD] Upload endpoint called with {len(files)} files")
    for f in files:
        logger.info(f"[UPLOAD] File: {f.filename} (Content-Type: {f.content_type})")

    batch_id = str(uuid.uuid4())
    task_infos = []
    errors = []

    for file in files:
        try:
            file_ext = Path(file.filename).suffix.lower()
            logger.info(f"[UPLOAD] Processing {file.filename} with extension: {file_ext}")

            if file_ext not in SUPPORTED_EXTENSIONS:
                error_msg = f"{file.filename}: Unsupported file type {file_ext}"
                logger.warning(f"[UPLOAD] {error_msg}")
                errors.append(error_msg)
                continue

            logger.info(f"[UPLOAD] Saving {file.filename} to temporary file")
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, dir="/tmp/shared") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            logger.info(f"[UPLOAD] Saved to: {tmp_path}")

            # Generate task ID
            task_id = str(uuid.uuid4())

            # Enqueue via pgmq
            enqueue_document_task(
                file_path=tmp_path,
                filename=file.filename,
                batch_id=batch_id,
                task_id=task_id,
            )

            task_infos.append(TaskInfo(task_id=task_id, filename=file.filename))
            logger.info(f"[UPLOAD] Queued task {task_id} for {file.filename}")

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"[UPLOAD] Error queueing {file.filename}: {str(e)}")
            logger.error(f"[UPLOAD] Traceback:\n{error_trace}")
            errors.append(f"{file.filename}: {str(e)}")

    if not task_infos and errors:
        error_msg = "; ".join(errors)
        logger.error(f"[UPLOAD] Upload failed: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)

    # Create batch and tasks in PostgreSQL
    async with get_session() as session:
        repo = JobRepository(session)
        await repo.create_batch(uuid.UUID(batch_id), len(task_infos))
        for ti in task_infos:
            await repo.create_task(uuid.UUID(ti.task_id), uuid.UUID(batch_id), ti.filename)

    logger.info(f"[UPLOAD] Created batch {batch_id} with {len(task_infos)} tasks")
    return BatchUploadResponse(
        status="queued",
        batch_id=batch_id,
        tasks=task_infos
    )


@router.get("/tasks/{batch_id}/status", response_model=BatchProgressResponse)
async def get_batch_status(batch_id: str):
    try:
        async with get_session() as session:
            repo = JobRepository(session)
            progress = await repo.get_batch_progress(uuid.UUID(batch_id))

        if not progress:
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

        return BatchProgressResponse(
            batch_id=progress["batch_id"],
            total=progress["total"],
            completed=progress["completed"],
            tasks=progress["tasks"]
        )
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid batch ID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document_by_id(document_id: str):
    try:
        async with get_session() as session:
            repo = DocumentRepository(session)
            deleted = await repo.delete_document_with_chunks(uuid.UUID(document_id))

        if not deleted:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

        # Clean up stored document file if it exists
        doc_storage_dir = DOCUMENT_STORAGE_PATH / document_id
        if doc_storage_dir.exists():
            try:
                shutil.rmtree(doc_storage_dir)
                logger.info(f"[DELETE] Removed stored document files for {document_id}")
            except Exception as e:
                logger.warning(f"[DELETE] Failed to remove stored files: {e}")

        return DeleteResponse(
            status="success",
            message=f"Document {document_id} deleted successfully"
        )
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/download")
async def download_document(document_id: str):
    """
    Download the original document file.

    Returns the original file if available, or 404 if not found.
    """
    try:
        doc_storage_dir = DOCUMENT_STORAGE_PATH / document_id

        if not doc_storage_dir.exists():
            # Try to get document metadata from PostgreSQL for error message
            async with get_session() as session:
                repo = DocumentRepository(session)
                doc_info = await repo.get_document_info(uuid.UUID(document_id))

            if doc_info:
                raise HTTPException(
                    status_code=404,
                    detail=f"Original file for '{doc_info.get('file_name', document_id)}' is no longer available"
                )
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

        # Find the file in the storage directory
        files = list(doc_storage_dir.iterdir())
        if not files:
            raise HTTPException(status_code=404, detail="Document file not found in storage")

        file_path = files[0]  # Should be only one file per document
        filename = file_path.name

        logger.info(f"[DOWNLOAD] Serving document {document_id}: {filename}")

        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/octet-stream"
        )

    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    except Exception as e:
        logger.error(f"[DOWNLOAD] Error downloading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
