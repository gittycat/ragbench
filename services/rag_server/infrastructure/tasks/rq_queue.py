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
