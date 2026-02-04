"""Job and batch repository for progress tracking."""

import logging
from typing import Any
from uuid import UUID

from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import JobBatch, JobTask
from infrastructure.database.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class JobRepository(BaseRepository[JobBatch]):
    """Repository for job batch and task operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, JobBatch)

    async def create_batch(
        self,
        batch_id: UUID,
        total_tasks: int,
    ) -> JobBatch:
        """Create a new job batch."""
        batch = JobBatch(
            id=batch_id,
            total_tasks=total_tasks,
            completed_tasks=0,
            status="pending",
        )
        return await self.add(batch)

    async def create_task(
        self,
        task_id: UUID,
        batch_id: UUID,
        filename: str,
    ) -> JobTask:
        """Create a new task within a batch."""
        task = JobTask(
            id=task_id,
            batch_id=batch_id,
            filename=filename,
            status="pending",
            total_chunks=0,
            completed_chunks=0,
        )
        self.session.add(task)
        await self.session.flush()
        return task

    async def get_task(self, task_id: UUID) -> JobTask | None:
        """Get a task by ID."""
        return await self.session.get(JobTask, task_id)

    async def update_task_status(
        self,
        task_id: UUID,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Update task status."""
        values = {"status": status}
        if error_message is not None:
            values["error_message"] = error_message

        await self.session.execute(
            update(JobTask).where(JobTask.id == task_id).values(**values)
        )
        await self.session.flush()

    async def set_task_total_chunks(self, task_id: UUID, total_chunks: int) -> None:
        """Set total chunks for a task."""
        await self.session.execute(
            update(JobTask)
            .where(JobTask.id == task_id)
            .values(total_chunks=total_chunks)
        )
        await self.session.flush()

    async def increment_task_chunk_progress(self, task_id: UUID) -> None:
        """Increment completed chunks for a task."""
        await self.session.execute(
            update(JobTask)
            .where(JobTask.id == task_id)
            .values(completed_chunks=JobTask.completed_chunks + 1)
        )
        await self.session.flush()

    async def complete_task(self, task_id: UUID) -> None:
        """Mark task as completed and update batch progress."""
        task = await self.get_task(task_id)
        if not task:
            return

        # Update task status
        await self.update_task_status(task_id, "completed")

        # Increment batch completed count
        await self.session.execute(
            update(JobBatch)
            .where(JobBatch.id == task.batch_id)
            .values(completed_tasks=JobBatch.completed_tasks + 1)
        )
        await self.session.flush()

        # Check if batch is complete
        batch = await self.get_by_id(task.batch_id)
        if batch and batch.completed_tasks >= batch.total_tasks:
            await self.session.execute(
                update(JobBatch)
                .where(JobBatch.id == task.batch_id)
                .values(status="completed")
            )
            await self.session.flush()

    async def fail_task(self, task_id: UUID, error_message: str) -> None:
        """Mark task as failed and update batch status."""
        task = await self.get_task(task_id)
        if not task:
            return

        # Update task status
        await self.update_task_status(task_id, "error", error_message)

        # Increment batch completed count (error is still "complete" for progress)
        await self.session.execute(
            update(JobBatch)
            .where(JobBatch.id == task.batch_id)
            .values(completed_tasks=JobBatch.completed_tasks + 1)
        )
        await self.session.flush()

        # Check if batch is complete (with errors)
        batch = await self.get_by_id(task.batch_id)
        if batch and batch.completed_tasks >= batch.total_tasks:
            await self.session.execute(
                update(JobBatch)
                .where(JobBatch.id == task.batch_id)
                .values(status="completed_with_errors")
            )
            await self.session.flush()

    async def get_batch_progress(self, batch_id: UUID) -> dict[str, Any] | None:
        """Get batch progress including all tasks."""
        batch = await self.get_by_id(batch_id)
        if not batch:
            return None

        # Get all tasks for this batch
        result = await self.session.execute(
            select(JobTask).where(JobTask.batch_id == batch_id)
        )
        tasks = list(result.scalars().all())

        # Calculate totals
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
