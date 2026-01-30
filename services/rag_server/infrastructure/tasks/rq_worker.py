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
