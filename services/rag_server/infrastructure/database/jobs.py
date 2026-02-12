"""Job and batch database operations for progress tracking."""

from typing import Any
from uuid import UUID

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import JobBatch, JobTask


async def create_batch(
    session: AsyncSession,
    batch_id: UUID,
    total_tasks: int,
) -> JobBatch:
    batch = JobBatch(
        id=batch_id,
        total_tasks=total_tasks,
        completed_tasks=0,
        status="pending",
    )
    session.add(batch)
    await session.flush()
    return batch


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


async def get_task(session: AsyncSession, task_id: UUID) -> JobTask | None:
    return await session.get(JobTask, task_id)


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


async def update_task_status(
    session: AsyncSession,
    task_id: UUID,
    status: str,
    error_message: str | None = None,
) -> None:
    values = {"status": status}
    if error_message is not None:
        values["error_message"] = error_message

    await session.execute(
        update(JobTask).where(JobTask.id == task_id).values(**values)
    )
    await session.flush()


async def set_task_total_chunks(session: AsyncSession, task_id: UUID, total_chunks: int) -> None:
    await session.execute(
        update(JobTask)
        .where(JobTask.id == task_id)
        .values(total_chunks=total_chunks)
    )
    await session.flush()


async def increment_task_chunk_progress(session: AsyncSession, task_id: UUID) -> None:
    await session.execute(
        update(JobTask)
        .where(JobTask.id == task_id)
        .values(completed_chunks=JobTask.completed_chunks + 1)
    )
    await session.flush()


async def _finish_task(
    session: AsyncSession,
    task_id: UUID,
    task_status: str,
    batch_status: str,
    error_message: str | None = None,
) -> None:
    task = await session.get(JobTask, task_id)
    if not task:
        return

    await update_task_status(session, task_id, task_status, error_message)

    await session.execute(
        update(JobBatch)
        .where(JobBatch.id == task.batch_id)
        .values(completed_tasks=JobBatch.completed_tasks + 1)
    )
    await session.flush()

    batch = await session.get(JobBatch, task.batch_id)
    if batch and batch.completed_tasks >= batch.total_tasks:
        await session.execute(
            update(JobBatch)
            .where(JobBatch.id == task.batch_id)
            .values(status=batch_status)
        )
        await session.flush()


async def complete_task(session: AsyncSession, task_id: UUID) -> None:
    """Mark task as completed and update batch progress."""
    await _finish_task(session, task_id, "completed", "completed")


async def fail_task(session: AsyncSession, task_id: UUID, error_message: str) -> None:
    """Mark task as failed and update batch status."""
    await _finish_task(session, task_id, "error", "completed_with_errors", error_message)


async def get_batch_progress(session: AsyncSession, batch_id: UUID) -> dict[str, Any] | None:
    batch = await session.get(JobBatch, batch_id)
    if not batch:
        return None

    result = await session.execute(
        select(JobTask).where(JobTask.batch_id == batch_id)
    )
    tasks = list(result.scalars().all())

    total_chunks = sum(t.total_chunks for t in tasks)
    completed_chunks = sum(t.completed_chunks for t in tasks)

    return {
        "batch_id": str(batch.id),
        "total": batch.total_tasks,
        "completed": batch.completed_tasks,
        "status": batch.status,
        "total_chunks": total_chunks,
        "completed_chunks": completed_chunks,
        "tasks": {
            str(t.id): {
                "filename": t.filename,
                "status": t.status,
                "total_chunks": t.total_chunks,
                "completed_chunks": t.completed_chunks,
                "error": t.error_message,
            }
            for t in tasks
        },
    }
