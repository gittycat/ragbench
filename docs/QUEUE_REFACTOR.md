Plan: Migrate from Celery to RQ (Redis Queue)

 Overview

 Replace Celery with RQ for async document processing. RQ is simpler, lighter, and sufficient for this use case.

 Context

 - Current: Celery worker processes uploaded documents asynchronously
 - Problem: Celery is heavyweight for this single-task use case
 - Solution: RQ provides same functionality with simpler API

 User Decisions

 1. Progress tracking: Keep current custom Redis batch tracking (progress.py unchanged)
 2. Periodic task: Remove unused auto_archive_sessions_task (Celery Beat was never running)
 3. Retries: Keep 3 retries with exponential-like intervals [5, 15, 60] seconds

 ---
 Current Celery Implementation (Reference)

 Task Definition (services/rag_server/infrastructure/tasks/worker.py)

 @celery_app.task(
     bind=True,
     name="infrastructure.tasks.worker.process_document_task",
     autoretry_for=(Exception,),
     retry_kwargs={'max_retries': 3, 'countdown': 5},
     retry_backoff=True,
     retry_backoff_max=60,
     retry_jitter=True
 )
 def process_document_task(self, file_path: str, filename: str, batch_id: str):
     task_id = self.request.id
     # ... processing logic
     # Uses self.request.retries and self.max_retries for retry tracking

 API Integration (services/rag_server/api/routes/documents.py)

 from infrastructure.tasks.worker import process_document_task

 task = process_document_task.apply_async(
     args=[tmp_path, file.filename, batch_id]
 )
 task_infos.append(TaskInfo(task_id=task.id, filename=file.filename))

 Docker Worker (docker-compose.yml)

 celery-worker:
   command: [".venv/bin/celery", "--quiet", "-A", "infrastructure.tasks.celery_app",
             "worker", "-Q", "celery,documents", "--concurrency=1",
             "--without-mingle", "--without-gossip"]

 ---
 Implementation Steps

 Step 1: Update Dependencies

 File: services/rag_server/pyproject.toml

 Find and replace in the dependencies array:
 - "celery>=5.5.0",
 + "rq>=2.0.0",

 Note: redis>=6.4.0 already exists (RQ compatible).

 ---
 Step 2: Create RQ Queue Module

 File: services/rag_server/infrastructure/tasks/rq_queue.py (NEW)

 """
 RQ Queue configuration.

 Replaces celery_app.py with simpler RQ setup.
 """
 from redis import Redis
 from rq import Queue
 from core.config import get_required_env
 from core.logging import configure_logging
 import logging

 configure_logging()
 logger = logging.getLogger(__name__)

 redis_url = get_required_env("REDIS_URL")


 def get_redis_connection() -> Redis:
     """Get Redis connection for RQ."""
     return Redis.from_url(redis_url)


 def get_documents_queue() -> Queue:
     """Get the documents processing queue."""
     return Queue('documents', connection=get_redis_connection())


 logger.info(f"[RQ] Queue configured with Redis: {redis_url}")

 ---
 Step 3: Create RQ Worker Entry Point

 File: services/rag_server/infrastructure/tasks/rq_worker.py (NEW)

 This replaces Celery's @worker_process_init.connect signal for initialization.

 """
 RQ Worker entry point with initialization.

 Initializes LlamaIndex settings before starting the RQ worker.
 Equivalent to Celery's worker_process_init signal.

 Usage: python -m infrastructure.tasks.rq_worker
 """
 import logging
 from rq import Worker, Queue
 from core.config import initialize_settings
 from core.logging import configure_logging
 from infrastructure.tasks.rq_queue import get_redis_connection

 configure_logging()
 logger = logging.getLogger(__name__)


 def main():
     """Initialize settings and start RQ worker."""
     # Initialize LlamaIndex settings (embedding model, LLM, etc.)
     logger.info("[RQ_WORKER] Initializing settings for worker process")
     initialize_settings()
     logger.info("[RQ_WORKER] Worker process initialization complete")

     # Get Redis connection
     conn = get_redis_connection()

     # Create queue to listen on
     queues = [Queue('documents', connection=conn)]

     # Start worker
     logger.info("[RQ_WORKER] Starting RQ worker on 'documents' queue")
     worker = Worker(queues, connection=conn)
     worker.work()


 if __name__ == '__main__':
     main()

 ---
 Step 4: Convert Worker Task

 File: services/rag_server/infrastructure/tasks/worker.py

 Replace entire file with:

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

 Key changes from Celery version:
 1. Removed @celery_app.task decorator
 2. Removed bind=True and self parameter
 3. self.request.id → get_current_job().id
 4. self.request.retries < self.max_retries → job.retries_left > 0
 5. Retry config removed (now at enqueue site)
 6. Removed auto_archive_sessions_task function entirely

 ---
 Step 5: Update API Route

 File: services/rag_server/api/routes/documents.py

 5a. Update imports (near top of file, around line 22-23)

 Replace:
 from infrastructure.tasks.worker import process_document_task

 With:
 from rq import Retry
 from infrastructure.tasks.rq_queue import get_documents_queue
 from infrastructure.tasks.worker import process_document_task

 5b. Update the enqueue call in upload_documents function (around line 130-133)

 Replace:
 task = process_document_task.apply_async(
     args=[tmp_path, file.filename, batch_id]
 )
 task_infos.append(TaskInfo(task_id=task.id, filename=file.filename))

 With:
 queue = get_documents_queue()
 job = queue.enqueue(
     process_document_task,
     file_path=tmp_path,
     filename=file.filename,
     batch_id=batch_id,
     retry=Retry(max=3, interval=[5, 15, 60]),
     job_timeout=3600,  # 1 hour timeout
 )
 task_infos.append(TaskInfo(task_id=job.id, filename=file.filename))

 Note: Move queue = get_documents_queue() outside the loop (before for file in files:) for efficiency.

 ---
 Step 6: Update Module Exports

 File: services/rag_server/infrastructure/tasks/__init__.py

 Replace contents with:
 """Task infrastructure for async document processing."""
 from infrastructure.tasks.rq_queue import get_documents_queue, get_redis_connection
 from infrastructure.tasks.worker import process_document_task

 __all__ = ['get_documents_queue', 'get_redis_connection', 'process_document_task']

 ---
 Step 7: Update Docker Compose

 File: docker-compose.yml

 7a. Rename and update worker service (around line 87-118)

 Replace the entire celery-worker service block:

   # Celery Worker - Async document processing (documents queue)
   celery-worker:
     build:
       context: .
       dockerfile: ./services/rag_server/Dockerfile
     command: [".venv/bin/celery", "--quiet", "-A", "infrastructure.tasks.celery_app", "worker", "-Q", "celery,documents",
 "--concurrency=1", "--without-mingle", "--without-gossip"]

 With:

   # RQ Worker - Async document processing (documents queue)
   rq-worker:
     build:
       context: .
       dockerfile: ./services/rag_server/Dockerfile
     command: [".venv/bin/python", "-m", "infrastructure.tasks.rq_worker"]

 Keep all other properties (user, restart, environment, extra_hosts, depends_on, networks, volumes) unchanged.

 7b. Update Redis comment (around line 77-78)

 Replace:
   # Redis - Message broker and result backend for Celery

 With:
   # Redis - Message broker for RQ, chat memory, progress tracking

 ---
 Step 8: Delete Celery App File

 File to delete: services/rag_server/infrastructure/tasks/celery_app.py

 ---
 Files Summary
 ┌────────────────────────────────────────────────────────┬─────────┬────────────────────────────────┐
 │                          File                          │ Action  │            Purpose             │
 ├────────────────────────────────────────────────────────┼─────────┼────────────────────────────────┤
 │ services/rag_server/pyproject.toml                     │ EDIT    │ Replace celery with rq         │
 ├────────────────────────────────────────────────────────┼─────────┼────────────────────────────────┤
 │ services/rag_server/infrastructure/tasks/rq_queue.py   │ CREATE  │ Queue + Redis connection       │
 ├────────────────────────────────────────────────────────┼─────────┼────────────────────────────────┤
 │ services/rag_server/infrastructure/tasks/rq_worker.py  │ CREATE  │ Worker entry point with init   │
 ├────────────────────────────────────────────────────────┼─────────┼────────────────────────────────┤
 │ services/rag_server/infrastructure/tasks/worker.py     │ REPLACE │ Convert task to plain function │
 ├────────────────────────────────────────────────────────┼─────────┼────────────────────────────────┤
 │ services/rag_server/infrastructure/tasks/__init__.py   │ EDIT    │ Update exports                 │
 ├────────────────────────────────────────────────────────┼─────────┼────────────────────────────────┤
 │ services/rag_server/api/routes/documents.py            │ EDIT    │ Use RQ enqueue                 │
 ├────────────────────────────────────────────────────────┼─────────┼────────────────────────────────┤
 │ docker-compose.yml                                     │ EDIT    │ Update worker service          │
 ├────────────────────────────────────────────────────────┼─────────┼────────────────────────────────┤
 │ services/rag_server/infrastructure/tasks/celery_app.py │ DELETE  │ No longer needed               │
 └────────────────────────────────────────────────────────┴─────────┴────────────────────────────────┘
 Unchanged files:
 - services/rag_server/infrastructure/tasks/progress.py - Custom Redis progress tracking works as-is
 - services/rag_server/tests/integration/test_async_upload.py - Tests use HTTP API, not Celery internals

 ---
 Verification

 # 1. Rebuild images (dependency change requires rebuild)
 docker compose build

 # 2. Start services
 docker compose up -d

 # 3. Check worker logs for initialization
 docker compose logs -f rq-worker

 # 4. Run integration tests
 just test-integration

 # 5. Manual test
 curl -X POST http://localhost:8001/upload \
   -F "files=@/path/to/test.pdf"

 # Then poll status:
 curl http://localhost:8001/tasks/{batch_id}/status

 Expected worker startup logs:
 [RQ_WORKER] Initializing settings for worker process
 [RQ_WORKER] Worker process initialization complete
 [RQ_WORKER] Starting RQ worker on 'documents' queue

 ---
 Rollback

 If issues arise, revert to Celery by:
 1. Restore celery_app.py from git
 2. Revert pyproject.toml changes
 3. Revert worker.py changes
 4. Revert documents.py changes
 5. Revert docker-compose.yml changes
 6. Delete new rq_queue.py and rq_worker.py files
 7. Rebuild: docker compose build
 