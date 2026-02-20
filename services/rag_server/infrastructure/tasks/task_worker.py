"""Task worker for async document processing using PostgreSQL SKIP LOCKED."""

import asyncio
import logging
import signal
import sys
from uuid import UUID

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*validate_default.*")

import os
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "WARNING").upper()),
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
from infrastructure.tasks.worker import process_document_async, _cleanup_temp_file

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
            # Exhausted all retries — clean up temp file
            logger.error(f"[WORKER] Task {task_id} exhausted all {MAX_ATTEMPTS} attempts")
            _cleanup_temp_file(task["file_path"], task_id)
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
