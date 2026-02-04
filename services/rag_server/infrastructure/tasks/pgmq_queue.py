"""PGMQ queue wrapper for PostgreSQL-based message queue."""

import json
import logging
import os
from typing import Any
from uuid import UUID

from tembo_pgmq_python import PGMQueue, Message

logger = logging.getLogger(__name__)

_queue: PGMQueue | None = None

QUEUE_NAME = "documents"


def _parse_database_url() -> dict:
    """Parse DATABASE_URL into connection parameters."""
    from urllib.parse import urlparse

    url = os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL environment variable not set")

    # Handle asyncpg:// prefix - convert to standard postgresql
    if "+asyncpg" in url:
        url = url.replace("+asyncpg", "")

    parsed = urlparse(url)
    return {
        "host": parsed.hostname or "localhost",
        "port": str(parsed.port or 5432),
        "username": parsed.username or "raguser",
        "password": parsed.password or "ragpass",
        "database": parsed.path.lstrip("/"),
    }


def get_queue() -> PGMQueue:
    """Get or create pgmq connection (lazy initialization)."""
    global _queue
    if _queue is None:
        params = _parse_database_url()
        _queue = PGMQueue(
            host=params["host"],
            port=params["port"],
            username=params["username"],
            password=params["password"],
            database=params["database"],
        )
        logger.info(f"[PGMQ] Connected to PostgreSQL queue at {params['host']}:{params['port']}")
    return _queue


def enqueue_document_task(
    file_path: str,
    filename: str,
    batch_id: str,
    task_id: str,
) -> int:
    """
    Enqueue a document processing task.

    Args:
        file_path: Path to temporary file
        filename: Original filename
        batch_id: Batch ID for progress tracking
        task_id: Task ID for this specific file

    Returns:
        Message ID from pgmq
    """
    queue = get_queue()

    message = {
        "file_path": file_path,
        "filename": filename,
        "batch_id": batch_id,
        "task_id": task_id,
    }

    msg_id = queue.send(QUEUE_NAME, message)
    logger.info(f"[PGMQ] Enqueued task {task_id} for {filename} (msg_id={msg_id})")
    return msg_id


def read_message(visibility_timeout: int = 60) -> Message | None:
    """
    Read a message from the queue.

    Args:
        visibility_timeout: Seconds before message becomes visible again if not deleted

    Returns:
        Message object or None if queue is empty
    """
    queue = get_queue()
    return queue.read(QUEUE_NAME, vt=visibility_timeout)


def delete_message(msg_id: int) -> bool:
    """Delete a message after successful processing."""
    queue = get_queue()
    return queue.delete(QUEUE_NAME, msg_id)


def archive_message(msg_id: int) -> bool:
    """Archive a message (keeps in archive table for debugging)."""
    queue = get_queue()
    return queue.archive(QUEUE_NAME, msg_id)


def get_queue_metrics() -> dict[str, Any]:
    """Get queue metrics (depth, oldest message age, etc.)."""
    queue = get_queue()
    metrics = queue.metrics(QUEUE_NAME)
    return {
        "queue_length": metrics.queue_length,
        "newest_msg_age_sec": metrics.newest_msg_age_sec,
        "oldest_msg_age_sec": metrics.oldest_msg_age_sec,
        "total_messages": metrics.total_messages,
    }
