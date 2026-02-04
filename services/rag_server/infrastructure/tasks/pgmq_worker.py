"""PGMQ worker for async document processing."""

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

# Suppress noisy warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*validate_default.*")

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Initialize LlamaIndex settings
from core.config import initialize_settings
initialize_settings()

from infrastructure.tasks.pgmq_queue import read_message, delete_message, archive_message
from infrastructure.tasks.worker import process_document

# Worker configuration
POLL_INTERVAL = 1  # Seconds between queue checks when idle
VISIBILITY_TIMEOUT = 3600  # 1 hour - long enough for large documents
MAX_RETRIES = 3
RETRY_DELAYS = [5, 15, 60]  # Seconds between retries

# Shutdown flag
_shutdown = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    _shutdown = True


def run_worker():
    """Main worker loop - polls pgmq for messages."""
    global _shutdown

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("=" * 60)
    logger.info("PGMQ Worker starting...")
    logger.info(f"Poll interval: {POLL_INTERVAL}s")
    logger.info(f"Visibility timeout: {VISIBILITY_TIMEOUT}s")
    logger.info(f"Max retries: {MAX_RETRIES}")
    logger.info("=" * 60)

    while not _shutdown:
        try:
            # Read message from queue
            message = read_message(visibility_timeout=VISIBILITY_TIMEOUT)

            if message is None:
                # Queue is empty, wait before polling again
                time.sleep(POLL_INTERVAL)
                continue

            # Process the message
            msg_id = message.msg_id
            data = message.message

            logger.info(f"[WORKER] Processing message {msg_id}: {data.get('filename', 'unknown')}")

            try:
                # Extract task data
                file_path = data["file_path"]
                filename = data["filename"]
                batch_id = data["batch_id"]
                task_id = data["task_id"]

                # Process with retry logic
                success = process_with_retry(
                    file_path=file_path,
                    filename=filename,
                    batch_id=batch_id,
                    task_id=task_id,
                    max_retries=MAX_RETRIES,
                    retry_delays=RETRY_DELAYS,
                )

                if success:
                    # Delete message on success
                    delete_message(msg_id)
                    logger.info(f"[WORKER] Message {msg_id} completed and deleted")
                else:
                    # Archive failed message for debugging
                    archive_message(msg_id)
                    logger.error(f"[WORKER] Message {msg_id} failed after all retries, archived")

            except Exception as e:
                logger.error(f"[WORKER] Error processing message {msg_id}: {e}")
                # Archive failed message
                archive_message(msg_id)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
            break
        except Exception as e:
            logger.error(f"[WORKER] Worker loop error: {e}")
            time.sleep(POLL_INTERVAL)

    logger.info("PGMQ Worker stopped")


def process_with_retry(
    file_path: str,
    filename: str,
    batch_id: str,
    task_id: str,
    max_retries: int,
    retry_delays: list[int],
) -> bool:
    """
    Process document with retry logic.

    Returns True on success, False on final failure.
    """
    for attempt in range(max_retries + 1):
        try:
            process_document(
                file_path=file_path,
                filename=filename,
                batch_id=batch_id,
                task_id=task_id,
            )
            return True

        except Exception as e:
            if attempt < max_retries:
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                logger.warning(
                    f"[WORKER] Attempt {attempt + 1}/{max_retries + 1} failed: {e}"
                )
                logger.info(f"[WORKER] Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(
                    f"[WORKER] All {max_retries + 1} attempts failed for {filename}: {e}"
                )
                return False

    return False


if __name__ == "__main__":
    run_worker()
