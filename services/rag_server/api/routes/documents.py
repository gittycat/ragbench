import uuid
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
from pipelines.ingestion import SUPPORTED_EXTENSIONS
from infrastructure.database.chroma import get_or_create_collection, list_documents, delete_document, check_documents_exist
from infrastructure.tasks.progress import create_batch, get_batch_progress
from rq import Retry
from infrastructure.tasks.rq_queue import get_documents_queue
from infrastructure.tasks.worker import process_document_task

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
        valid_sort_fields = ['name', 'chunks', 'uploaded_at']
        if sort_by not in valid_sort_fields:
            sort_by = 'uploaded_at'
        if sort_order not in ['asc', 'desc']:
            sort_order = 'desc'

        index = get_or_create_collection()
        documents = list_documents(index, sort_by=sort_by, sort_order=sort_order)
        return DocumentListResponse(documents=documents)
    except Exception as e:
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

        index = get_or_create_collection()
        file_checks = [{"filename": f.filename, "size": f.size, "hash": f.hash} for f in request.files]
        results = check_documents_exist(index, file_checks)

        # Convert to response model
        formatted_results = {}
        for filename, info in results.items():
            formatted_results[filename] = FileCheckResult(
                filename=filename,
                exists=info["exists"],
                document_id=info.get("document_id"),
                existing_filename=info.get("existing_filename"),
                reason=info.get("reason")
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

    queue = get_documents_queue()

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

            job = queue.enqueue(
                process_document_task,
                file_path=tmp_path,
                filename=file.filename,
                batch_id=batch_id,
                retry=Retry(max=3, interval=[5, 15, 60]),
                job_timeout=3600,  # 1 hour timeout
            )
            task_infos.append(TaskInfo(task_id=job.id, filename=file.filename))
            logger.info(f"[UPLOAD] Queued task {job.id} for {file.filename}")

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

    task_ids = [ti.task_id for ti in task_infos]
    filenames = [ti.filename for ti in task_infos]
    create_batch(batch_id, task_ids, filenames)

    logger.info(f"[UPLOAD] Created batch {batch_id} with {len(task_infos)} tasks")
    return BatchUploadResponse(
        status="queued",
        batch_id=batch_id,
        tasks=task_infos
    )


@router.get("/tasks/{batch_id}/status", response_model=BatchProgressResponse)
async def get_batch_status(batch_id: str):
    try:
        progress = get_batch_progress(batch_id)
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document_by_id(document_id: str):
    try:
        index = get_or_create_collection()
        delete_document(index, document_id)

        # Refresh BM25 retriever after deleting documents (for hybrid search)
        try:
            from pipelines.inference import refresh_bm25_retriever
            refresh_bm25_retriever(index)
            logger.info(f"[DELETE] BM25 retriever refreshed after deleting document {document_id}")
        except Exception as e:
            logger.warning(f"[DELETE] Failed to refresh BM25 retriever: {e}")
            # Non-critical, continue

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
            # Try to get document metadata from ChromaDB for error message
            index = get_or_create_collection()
            documents = list_documents(index)
            doc_info = next((d for d in documents if d.get('id') == document_id), None)

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
    except Exception as e:
        logger.error(f"[DOWNLOAD] Error downloading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
