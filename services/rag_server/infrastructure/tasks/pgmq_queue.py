"""PGMQ queue wrapper for PostgreSQL-based message queue."""

import json
import logging
from typing import Any

from pgmq import PGMQueue, Message

from app.settings import get_database_params

logger = logging.getLogger(__name__)

_queue: PGMQueue | None = None

QUEUE_NAME = "documents"


def get_queue() -> PGMQueue:
    """Get or create pgmq connection (lazy initialization)."""
    global _queue
    if _queue is None:
        params = get_database_params()
        _queue = PGMQueue(
            host=params["host"],
            port=params["port"],
            username=params["user"],
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

    Note: Patched to handle pgmq extension v1.9.0 which returns 7 columns
    (msg_id, read_ct, enqueued_at, last_read_at, vt, message, headers)
    instead of the 5 columns the Python library expects.
    """
    queue = get_queue()

    # Call the underlying SQL function directly to handle correct column mapping
    query = "select * from pgmq.read(queue_name=>%s::text, vt=>%s::integer, qty=>%s::integer);"
    rows = queue._execute_query_with_result(query, [QUEUE_NAME, visibility_timeout, 1])

    if not rows:
        return None

    # Handle 7-column structure: msg_id, read_ct, enqueued_at, last_read_at, vt, message, headers
    row = rows[0]
    return Message(
        msg_id=row[0],
        read_ct=row[1],
        enqueued_at=row[2],
        vt=row[4],  # Skip last_read_at at index 3
        message=row[5]  # Message is at index 5, not 4
    )


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
