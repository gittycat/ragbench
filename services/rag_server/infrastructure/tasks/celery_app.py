from celery import Celery
from celery.signals import worker_process_init
from core.config import get_required_env
from core.logging import configure_logging
import logging

configure_logging()
logger = logging.getLogger(__name__)

redis_url = get_required_env("REDIS_URL")

celery_app = Celery(
    "rag_tasks",
    broker=redis_url,
    backend=redis_url,
    include=["infrastructure.tasks.worker"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_without_mingle=True,
    worker_without_gossip=True,
    worker_log_format='[%(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(levelname)s/%(processName)s] [%(task_name)s] %(message)s',
)

# Task routing for queue isolation
celery_app.conf.task_routes = {
    'infrastructure.tasks.worker.process_document_task': {'queue': 'documents'},
}

# Periodic task schedule (Celery Beat)
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    'auto-archive-inactive-sessions': {
        'task': 'auto_archive_sessions',
        'schedule': crontab(hour=0, minute=0),  # Daily at midnight UTC
    }
}

@worker_process_init.connect
def init_worker(**kwargs):
    from core.config import initialize_settings
    logger.info("[CELERY] Initializing settings for worker process")
    initialize_settings()
    logger.info("[CELERY] Worker process initialization complete")

logger.info(f"[CELERY] Celery app configured with Redis: {redis_url}")
