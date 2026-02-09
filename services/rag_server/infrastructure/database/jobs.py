"""Job and batch database operations for progress tracking."""

from typing import Any
from uuid import UUID

from sqlalchemy import select, update
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
) -> JobTask:
    task = JobTask(
        id=task_id,
        batch_id=batch_id,
        filename=filename,
        status="pending",
        total_chunks=0,
        completed_chunks=0,
    )
    session.add(task)
    await session.flush()
    return task


async def get_task(session: AsyncSession, task_id: UUID) -> JobTask | None:
    return await session.get(JobTask, task_id)


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
