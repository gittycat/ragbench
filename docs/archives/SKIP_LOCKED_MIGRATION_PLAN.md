# Refactor: pgmq → SKIP LOCKED Work Queue

## Context

The work queue currently uses **pgmq** (a PostgreSQL extension + Python library) to queue document processing tasks. This adds complexity:
- A separate PostgreSQL extension installed via PGXN (requires build-essential, postgresql-server-dev-17)
- A sync Python library (`pgmq`) with its own psycopg2 connection (separate from the app's SQLAlchemy async pool)
- Schema grants for the pgmq schema
- A workaround for pgmq v1.9.0 returning 7 columns instead of 5
- Dual data path: pgmq messages AND job_tasks table track the same work

**Key insight**: The `job_tasks` table already tracks every task's status, progress, and metadata. With `SELECT FOR UPDATE SKIP LOCKED`, it can serve as both the queue AND the progress tracker — eliminating pgmq entirely.

**Outcome**: Simpler architecture, fewer dependencies, fully async worker using the shared SQLAlchemy pool, smaller Docker image.

---

## Implementation Order

Execute steps in this exact order. Each step lists the file path and the exact changes.

---

### Step 1: Schema — `services/postgres/init.sql`

**Remove these lines:**
- Line 6: `CREATE EXTENSION IF NOT EXISTS pgmq;`
- Line 83: `SELECT pgmq.create('documents');`
- Line 4 comment: change `pgmq for message queue` to just remove the pgmq reference

**Replace the `job_tasks` table definition (lines 70-78) with:**

```sql
-- Job tasks (also serves as the work queue via SKIP LOCKED)
CREATE TABLE IF NOT EXISTS job_tasks (
    id UUID PRIMARY KEY,
    batch_id UUID NOT NULL REFERENCES job_batches(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    status VARCHAR(30) DEFAULT 'pending',
    attempt INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    completed_chunks INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_tasks_batch ON job_tasks(batch_id);
-- Partial index: only pending tasks for fast SKIP LOCKED claims
CREATE INDEX IF NOT EXISTS idx_tasks_claimable ON job_tasks(created_at) WHERE status = 'pending';
```

Remove the old `idx_tasks_batch` index line (line 80) since it's now in the block above.

**Remove the final line:**
```sql
-- Remove this:
SELECT pgmq.create('documents');
```

---

### Step 2: Models — `services/rag_server/infrastructure/database/models.py`

**Replace the `JobTask` class (lines 169-191) with:**

```python
class JobTask(Base):
    """Individual tasks within a job batch. Also serves as the work queue via SKIP LOCKED."""
    __tablename__ = "job_tasks"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    batch_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("job_batches.id", ondelete="CASCADE"),
        nullable=False,
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="pending")
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    completed_chunks: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    batch: Mapped["JobBatch"] = relationship(back_populates="tasks")

    __table_args__ = (
        Index("idx_tasks_batch", "batch_id"),
        Index(
            "idx_tasks_claimable",
            "created_at",
            postgresql_where=text("status = 'pending'"),
        ),
    )

    def __repr__(self) -> str:
        return f"<JobTask {self.filename} ({self.status})>"
```

**IMPORTANT**: Add `text` to the existing sqlalchemy import on line 16. It's already imported there — verify it includes `text`. The current import is:
```python
from sqlalchemy import (
    BigInteger, Boolean, DateTime, ForeignKey, Index, Integer, String, Text, func,
)
```
Add `text` to this list (lowercase `text`, not `Text`).

---

### Step 3: Database functions — `services/rag_server/infrastructure/database/jobs.py`

**Add `text` import** — add to the existing imports at the top:
```python
from sqlalchemy import select, text, update
```

**Replace `create_task` function (lines 28-44) with:**

```python
async def create_task(
    session: AsyncSession,
    task_id: UUID,
    batch_id: UUID,
    filename: str,
    file_path: str,
) -> JobTask:
    task = JobTask(
        id=task_id,
        batch_id=batch_id,
        filename=filename,
        file_path=file_path,
        status="pending",
        total_chunks=0,
        completed_chunks=0,
    )
    session.add(task)
    await session.flush()
    return task
```

**Add these two new functions after `get_task` (after line 48):**

```python
async def claim_next_task(session: AsyncSession) -> dict | None:
    """Atomically claim the next pending task using SKIP LOCKED.

    Returns a dict with task data, or None if no tasks available.
    Uses a CTE to SELECT + UPDATE in a single round-trip.
    """
    result = await session.execute(
        text("""
            WITH next_task AS (
                SELECT id FROM job_tasks
                WHERE status = 'pending'
                ORDER BY created_at
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE job_tasks
            SET status = 'processing',
                started_at = NOW(),
                attempt = attempt + 1
            FROM next_task
            WHERE job_tasks.id = next_task.id
            RETURNING job_tasks.id, job_tasks.batch_id, job_tasks.filename,
                      job_tasks.file_path, job_tasks.attempt
        """)
    )
    row = result.fetchone()
    if row is None:
        return None

    return {
        "id": str(row.id),
        "batch_id": str(row.batch_id),
        "filename": row.filename,
        "file_path": row.file_path,
        "attempt": row.attempt,
    }


async def reset_task_for_retry(session: AsyncSession, task_id: UUID) -> None:
    """Reset a failed task back to pending for retry.

    The attempt counter is NOT reset — it was already incremented by claim_next_task.
    """
    await session.execute(
        update(JobTask)
        .where(JobTask.id == task_id)
        .values(status="pending", started_at=None, error_message=None)
    )
    await session.flush()


async def reset_stuck_tasks(
    session: AsyncSession,
    timeout_seconds: int = 3600,
    max_attempts: int = 3,
) -> int:
    """Reset tasks stuck in 'processing' back to pending, or mark as error if exhausted.

    Called periodically by the worker to recover from crashed workers.
    Returns the number of tasks reset to pending.
    """
    # Mark exhausted tasks as error
    await session.execute(
        text("""
            UPDATE job_tasks
            SET status = 'error',
                error_message = 'Task exceeded maximum retry attempts (stuck worker)'
            WHERE status = 'processing'
              AND started_at < NOW() - make_interval(secs => :timeout)
              AND attempt >= :max_attempts
        """),
        {"timeout": timeout_seconds, "max_attempts": max_attempts},
    )

    # Reset retryable tasks to pending
    result = await session.execute(
        text("""
            UPDATE job_tasks
            SET status = 'pending', started_at = NULL
            WHERE status = 'processing'
              AND started_at < NOW() - make_interval(secs => :timeout)
              AND attempt < :max_attempts
        """),
        {"timeout": timeout_seconds, "max_attempts": max_attempts},
    )
    await session.flush()
    return result.rowcount
```

---

### Step 4: New worker — CREATE `services/rag_server/infrastructure/tasks/task_worker.py`

Create this file with the following content:

```python
"""Task worker for async document processing using PostgreSQL SKIP LOCKED."""

import asyncio
import logging
import signal
import sys
from uuid import UUID

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*validate_default.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Initialize secrets and LlamaIndex settings
from app.settings import init_settings
from core.config import initialize_settings
init_settings()
initialize_settings()

from infrastructure.database.postgres import get_session
from infrastructure.database import jobs as db_jobs
from infrastructure.tasks.worker import process_document_async

# Worker configuration
POLL_INTERVAL = 1.0        # Seconds between queue checks when idle
MAX_ATTEMPTS = 3           # Max claim attempts before permanent failure
STUCK_TIMEOUT = 3600       # Seconds before a processing task is considered stuck
STUCK_CHECK_INTERVAL = 60  # Seconds between stuck-task checks
RETRY_DELAYS = [5, 15, 60] # Seconds to wait before re-queuing for retry

_shutdown = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    _shutdown = True


async def claim_and_process() -> bool:
    """Claim one pending task and process it.

    Returns True if a task was claimed and processed (success or failure).
    Returns False if no tasks were available.
    """
    async with get_session() as session:
        task = await db_jobs.claim_next_task(session)

    if task is None:
        return False

    task_id = task["id"]
    filename = task["filename"]
    attempt = task["attempt"]

    logger.info(f"[WORKER] Claimed task {task_id}: {filename} (attempt {attempt}/{MAX_ATTEMPTS})")

    try:
        await process_document_async(
            file_path=task["file_path"],
            filename=filename,
            batch_id=task["batch_id"],
            task_id=task_id,
        )
        logger.info(f"[WORKER] Task {task_id} completed successfully")

    except FileNotFoundError as e:
        # Permanent failure — file missing, no point retrying
        logger.error(f"[WORKER] Permanent failure for {filename}: {e}")
        async with get_session() as session:
            await db_jobs.fail_task(session, UUID(task_id), str(e))

    except Exception as e:
        logger.error(f"[WORKER] Task {task_id} failed (attempt {attempt}): {e}")

        if attempt >= MAX_ATTEMPTS:
            # Exhausted all retries
            logger.error(f"[WORKER] Task {task_id} exhausted all {MAX_ATTEMPTS} attempts")
            async with get_session() as session:
                await db_jobs.fail_task(
                    session, UUID(task_id),
                    f"Failed after {MAX_ATTEMPTS} attempts: {e}",
                )
        else:
            # Re-queue for retry after delay
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            logger.info(f"[WORKER] Re-queuing task {task_id} for retry (delay {delay}s)")
            await asyncio.sleep(delay)
            async with get_session() as session:
                await db_jobs.reset_task_for_retry(session, UUID(task_id))

    return True


async def check_stuck_tasks():
    """Background coroutine that periodically resets stuck tasks."""
    while not _shutdown:
        try:
            async with get_session() as session:
                reset_count = await db_jobs.reset_stuck_tasks(
                    session,
                    timeout_seconds=STUCK_TIMEOUT,
                    max_attempts=MAX_ATTEMPTS,
                )
                if reset_count > 0:
                    logger.warning(f"[WORKER] Reset {reset_count} stuck task(s) to pending")
        except Exception as e:
            logger.error(f"[WORKER] Error checking stuck tasks: {e}")

        await asyncio.sleep(STUCK_CHECK_INTERVAL)


async def run_worker():
    """Main worker loop — polls for tasks via SKIP LOCKED."""
    logger.info("=" * 60)
    logger.info("Task Worker starting (SKIP LOCKED)...")
    logger.info(f"Poll interval: {POLL_INTERVAL}s")
    logger.info(f"Max attempts: {MAX_ATTEMPTS}")
    logger.info(f"Stuck timeout: {STUCK_TIMEOUT}s")
    logger.info("=" * 60)

    # Start stuck-task checker as background coroutine
    stuck_checker = asyncio.create_task(check_stuck_tasks())

    try:
        while not _shutdown:
            try:
                processed = await claim_and_process()
                if not processed:
                    # No tasks available, wait before polling again
                    await asyncio.sleep(POLL_INTERVAL)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"[WORKER] Worker loop error: {e}")
                await asyncio.sleep(POLL_INTERVAL)
    finally:
        stuck_checker.cancel()
        try:
            await stuck_checker
        except asyncio.CancelledError:
            pass

    logger.info("Task Worker stopped")


def main():
    """Entry point for the task worker."""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
```

---

### Step 5: Upload route — `services/rag_server/api/routes/documents.py`

**Remove this import (line 27):**
```python
from infrastructure.tasks.pgmq_queue import enqueue_document_task
```

**Replace the `upload_documents` function (lines 138-205) with:**

```python
@router.post("/upload", response_model=BatchUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    logger.info(f"[UPLOAD] Upload endpoint called with {len(files)} files")
    for f in files:
        logger.info(f"[UPLOAD] File: {f.filename} (Content-Type: {f.content_type})")

    batch_id = str(uuid.uuid4())
    task_infos = []
    errors = []
    saved_files = []  # List of (task_id, filename, tmp_path) for DB insertion

    for file in files:
        try:
            file_ext = Path(file.filename).suffix.lower()
            logger.info(f"[UPLOAD] Processing {file.filename} with extension: {file_ext}")

            if file_ext not in SUPPORTED_EXTENSIONS:
                error_msg = f"{file.filename}: Unsupported file type {file_ext}"
                logger.warning(f"[UPLOAD] {error_msg}")
                errors.append(error_msg)
                continue

            shared_dir = Path(get_shared_upload_dir())
            shared_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[UPLOAD] Saving {file.filename} to temporary file in {shared_dir}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, dir=str(shared_dir)) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            logger.info(f"[UPLOAD] Saved to: {tmp_path}")

            task_id = str(uuid.uuid4())
            saved_files.append((task_id, file.filename, tmp_path))
            task_infos.append(TaskInfo(task_id=task_id, filename=file.filename))
            logger.info(f"[UPLOAD] Prepared task {task_id} for {file.filename}")

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"[UPLOAD] Error saving {file.filename}: {str(e)}")
            logger.error(f"[UPLOAD] Traceback:\n{error_trace}")
            errors.append(f"{file.filename}: {str(e)}")

    if not task_infos and errors:
        error_msg = "; ".join(errors)
        logger.error(f"[UPLOAD] Upload failed: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)

    # Create batch and tasks in a single transaction.
    # Tasks are immediately claimable by the worker via SKIP LOCKED.
    async with get_session() as session:
        await db_jobs.create_batch(session, uuid.UUID(batch_id), len(task_infos))
        for task_id, filename, tmp_path in saved_files:
            await db_jobs.create_task(
                session, uuid.UUID(task_id), uuid.UUID(batch_id), filename,
                file_path=tmp_path,
            )

    logger.info(f"[UPLOAD] Created batch {batch_id} with {len(task_infos)} tasks")
    return BatchUploadResponse(
        status="queued",
        batch_id=batch_id,
        tasks=task_infos
    )
```

---

### Step 6: Worker cleanup — `services/rag_server/infrastructure/tasks/worker.py`

**Update docstring (lines 1-6):** Change `pgmq worker` to `task worker`:
```python
"""
Document processing job for task worker.

Handles async document processing: chunking, embedding, indexing.
Progress tracking via PostgreSQL job_batches/job_tasks tables.
"""
```

**Remove the `_update_task_status(task_id, "processing")` call (line 60).** The `claim_next_task()` function already sets status to 'processing'. Remove this line:
```python
        await _update_task_status(task_id, "processing")
```

**Modify the except block (lines 132-147).** Remove the `_fail_task` call — the caller (task_worker.py) handles failure status. The except block should ONLY clean up the temp file and re-raise:

Replace lines 132-147 with:
```python
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"[TASK {task_id}] Error processing {filename}: {str(e)}")
        logger.error(f"[TASK {task_id}] Traceback:\n{error_trace}")

        # Clean up temp file on failure
        _cleanup_temp_file(file_path, task_id)

        raise
```

**Delete `process_document` sync wrapper (lines 150-157):**
```python
# DELETE THIS ENTIRE FUNCTION:
def process_document(file_path: str, filename: str, batch_id: str, task_id: str) -> dict:
    ...
```

**Delete backward compat alias (line 197):**
```python
# DELETE THIS LINE:
process_document_task = process_document
```

**Also delete the `_update_task_status` helper (lines 160-162)** since it's no longer called:
```python
# DELETE:
async def _update_task_status(task_id: str, status: str) -> None:
    async with get_session() as session:
        await db_jobs.update_task_status(session, UUID(task_id), status)
```

Keep all other helper functions: `_set_task_total_chunks`, `_increment_task_chunk_progress`, `_complete_task`, `_fail_task`, `_cleanup_temp_file`.

---

### Step 7: Module exports — `services/rag_server/infrastructure/tasks/__init__.py`

**Replace entire file with:**

```python
"""Task processing for async document ingestion via SKIP LOCKED."""

from infrastructure.tasks.worker import process_document_async

__all__ = ["process_document_async"]
```

---

### Step 8: Remove close_all_connections — `services/rag_server/infrastructure/database/postgres.py`

**Delete the `close_all_connections()` function (lines 127-171).** This existed only for pgmq-worker's sync/async boundary. The new task_worker is fully async and uses the standard SQLAlchemy pool.

---

### Step 9: Delete old files

Delete these two files entirely:
- `services/rag_server/infrastructure/tasks/pgmq_queue.py`
- `services/rag_server/infrastructure/tasks/pgmq_worker.py`

---

### Step 10: Postgres Dockerfile — `services/postgres/Dockerfile`

**Remove the pgmq PGXN install block (lines 17-26).** Delete these lines:

```dockerfile
# Install pgmq via PGXN (PostgreSQL Extension Network)
# pgmq is pure PL/pgSQL, no compilation needed but PGXN client requires build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pgxnclient \
        build-essential \
        postgresql-server-dev-17 && \
    pgxn install pgmq && \
    apt-get purge -y --auto-remove pgxnclient build-essential postgresql-server-dev-17 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
```

**Update the comment on line 1** to remove pgmq reference:
```dockerfile
# PostgreSQL 17 with pg_textsearch (BM25)
```

---

### Step 11: Grants — `services/postgres/02-grants.sh`

**Remove all pgmq schema grants.** Delete these lines from inside the `DO $$ ... END $$` block:

```sql
  EXECUTE format('GRANT USAGE ON SCHEMA pgmq TO %I', rag_user);
  EXECUTE format(
    'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA pgmq TO %I',
    rag_user
  );
  EXECUTE format(
    'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA pgmq TO %I',
    rag_user
  );
  EXECUTE format(
    'GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA pgmq TO %I',
    rag_user
  );
  EXECUTE format(
    'ALTER DEFAULT PRIVILEGES IN SCHEMA pgmq GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO %I',
    rag_user
  );
  EXECUTE format(
    'ALTER DEFAULT PRIVILEGES IN SCHEMA pgmq GRANT USAGE, SELECT ON SEQUENCES TO %I',
    rag_user
  );
  EXECUTE format(
    'ALTER DEFAULT PRIVILEGES IN SCHEMA pgmq GRANT EXECUTE ON FUNCTIONS TO %I',
    rag_user
  );
```

Keep all public schema grants as-is.

---

### Step 12: Docker Compose files

#### `docker-compose.yml`

**Rename the service (lines 107-146):**
- Change `pgmq-worker:` to `task-worker:`
- Change the comment from `# PGMQ Worker - Async document processing via PostgreSQL queue` to `# Task Worker - Async document processing via SKIP LOCKED`
- Change command from `[".venv/bin/python", "-m", "infrastructure.tasks.pgmq_worker"]` to `[".venv/bin/python", "-m", "infrastructure.tasks.task_worker"]`
- Update postgres comment (line 67): remove `+ pgmq` → just `# PostgreSQL 17 with pg_textsearch (BM25)`

#### `docker-compose.bench.yml`

- Rename `pgmq-worker-bench:` (line 77) to `task-worker-bench:`
- Update comment (line 76) from `# PGMQ Worker for benchmark` to `# Task Worker for benchmark`
- Change command (line 81) from `infrastructure.tasks.pgmq_worker` to `infrastructure.tasks.task_worker`

#### `docker-compose.cloud.yml`

- Rename `celery-worker:` (line 34) to `task-worker:`
- Update comment if present

---

### Step 13: Dependencies — `services/rag_server/pyproject.toml`

**Remove this line (line 36):**
```toml
    "pgmq>=1.0.3",
```

---

### Step 14: Tests

#### `tests/integration/test_infrastructure.py`

**In `test_postgres_extensions` (line 54):** Remove `"pgmq"` from the extensions list:
```python
        for ext in ["vector", "pg_search"]:
```

**Replace `test_pgmq_queue_exists` method (lines 108-123) with:**

```python
    def test_claimable_tasks_index_exists(self, integration_env, check_services):
        """Verify partial index for SKIP LOCKED task claiming exists."""
        import asyncio
        from sqlalchemy import text
        from infrastructure.database.postgres import get_session

        async def _check():
            async with get_session() as session:
                result = await session.execute(
                    text(
                        "SELECT indexname FROM pg_indexes "
                        "WHERE tablename = 'job_tasks' "
                        "AND indexname = 'idx_tasks_claimable'"
                    )
                )
                return [row[0] for row in result.fetchall()]

        indexes = asyncio.run(_check())
        assert "idx_tasks_claimable" in indexes, (
            f"Partial index 'idx_tasks_claimable' not found on job_tasks"
        )
```

#### `tests/integration/test_async_upload.py`

- Line 2: Change `"""Integration tests for async document upload via pgmq."""` to `"""Integration tests for async document upload."""`
- Line 4: Change reference from `pgmq task` to `task processing`
- Line 7: Change `pgmq-worker` to `task-worker`
- Line 21: Rename class `TestPgmqTaskCompletion` to `TestTaskCompletion`
- Lines 23-30: Update docstring to remove pgmq references, mention `task-worker`

No functional test changes needed — tests use API endpoints, not pgmq directly.

---

### Step 15: Documentation updates

#### `CLAUDE.md`

Update these sections:
- **Project Overview**: Remove "pgmq" mention
- **Architecture**: `pgmq-worker` → `task-worker`, describe as "Async document processing worker via SKIP LOCKED"
- **Document Processing / Upload**: Remove pgmq reference, describe SKIP LOCKED flow
- **Key Patterns / Async Processing**: Change from "pgmq + PostgreSQL job_batches/job_tasks" to "SKIP LOCKED on job_tasks table"
- **Key Files**: Remove `pgmq_queue.py`, add `task_worker.py`
- **Common Issues**: Remove pgmq-worker references, update to task-worker
- **Database Schema**: Remove pgmq extension, add new job_tasks columns

#### Other docs — search-replace in these files:
- `DEVELOPMENT.md` — `pgmq-worker` → `task-worker`
- `README.md` — `pgmq-worker` → `task-worker`, remove pgmq mentions
- `ERRORS_FIXED.md` — remove or mark pgmq entries as historical
- `docs/ROADMAP.md` — update pgmq references
- `config.yml` — line 190 comment: `pgmq-worker` → `task-worker`

---

## Files Summary

| # | File | Action |
|---|------|--------|
| 1 | `services/postgres/init.sql` | MODIFY — remove pgmq extension, add columns + partial index to job_tasks |
| 2 | `services/rag_server/infrastructure/database/models.py` | MODIFY — add file_path, attempt, created_at, started_at + partial index to JobTask |
| 3 | `services/rag_server/infrastructure/database/jobs.py` | MODIFY — add claim_next_task, reset_task_for_retry, reset_stuck_tasks; update create_task |
| 4 | `services/rag_server/infrastructure/tasks/task_worker.py` | CREATE — new fully async worker using SKIP LOCKED |
| 5 | `services/rag_server/api/routes/documents.py` | MODIFY — remove pgmq import/enqueue, pass file_path to create_task |
| 6 | `services/rag_server/infrastructure/tasks/worker.py` | MODIFY — remove _fail_task from except, remove sync wrapper, remove _update_task_status call |
| 7 | `services/rag_server/infrastructure/tasks/__init__.py` | REWRITE — remove pgmq exports |
| 8 | `services/rag_server/infrastructure/database/postgres.py` | MODIFY — delete close_all_connections function |
| 9 | `services/rag_server/infrastructure/tasks/pgmq_queue.py` | DELETE |
| 10 | `services/rag_server/infrastructure/tasks/pgmq_worker.py` | DELETE |
| 11 | `services/postgres/Dockerfile` | MODIFY — remove pgmq PGXN install block |
| 12 | `services/postgres/02-grants.sh` | MODIFY — remove all pgmq schema grants |
| 13 | `docker-compose.yml` | MODIFY — rename pgmq-worker → task-worker, update command |
| 14 | `docker-compose.bench.yml` | MODIFY — rename pgmq-worker-bench → task-worker-bench, update command |
| 15 | `docker-compose.cloud.yml` | MODIFY — rename celery-worker → task-worker |
| 16 | `services/rag_server/pyproject.toml` | MODIFY — remove pgmq dependency |
| 17 | `tests/integration/test_infrastructure.py` | MODIFY — remove pgmq tests, add index test |
| 18 | `tests/integration/test_async_upload.py` | MODIFY — rename class, update comments |
| 19 | `CLAUDE.md` | MODIFY — update architecture docs |
| 20 | Other docs | MODIFY — search-replace pgmq references |

---

## Verification

```bash
# 1. Clean rebuild (required — schema changed)
docker compose down -v && docker compose up -d --build

# 2. Verify job_tasks schema
docker compose exec postgres psql -U $(cat secrets/POSTGRES_SUPERUSER) -d ragbench -c "\d job_tasks"
# Expect: file_path, attempt, created_at, started_at columns present

# 3. Verify no pgmq extension
docker compose exec postgres psql -U $(cat secrets/POSTGRES_SUPERUSER) -d ragbench \
  -c "SELECT extname FROM pg_extension;"
# Expect: pg_textsearch only, NO pgmq

# 4. Verify partial index exists
docker compose exec postgres psql -U $(cat secrets/POSTGRES_SUPERUSER) -d ragbench \
  -c "\di idx_tasks_claimable"

# 5. Upload a document and watch worker logs
docker compose logs -f task-worker
# In another terminal: upload a test file via the API or web UI

# 6. Check progress
curl http://localhost:8001/tasks/{batch_id}/status

# 7. Run tests
just test-unit
just test-integration

# 8. Verify no pgmq references remain in code
grep -r "pgmq" services/ --include="*.py" --include="*.sql" --include="*.sh"
# Expect: no matches
```
