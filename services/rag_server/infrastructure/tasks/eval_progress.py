"""Redis-based progress tracking for evaluation runs."""

import json
import logging
from datetime import datetime
from typing import Any

import redis

from core.config import get_required_env

logger = logging.getLogger(__name__)

# 24 hours TTL for eval progress
EVAL_PROGRESS_TTL = 86400


def get_redis_client():
    """Get Redis client from environment configuration."""
    redis_url = get_required_env("REDIS_URL")
    return redis.from_url(redis_url, decode_responses=True)


def _get_key(run_id: str) -> str:
    """Get Redis key for an eval run."""
    return f"eval:run:{run_id}:progress"


def create_eval_run(
    run_id: str,
    name: str,
    groups: list[str],
    datasets: list[str],
    total_questions: int,
    config: dict[str, Any],
) -> None:
    """Create a new eval run entry in Redis.

    Args:
        run_id: Unique run identifier
        name: Human-readable run name
        groups: Metric groups to evaluate
        datasets: Datasets being used
        total_questions: Total number of questions to evaluate
        config: Run configuration snapshot
    """
    client = get_redis_client()
    key = _get_key(run_id)

    data = {
        "run_id": run_id,
        "name": name,
        "status": "pending",
        "phase": "initializing",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "groups": groups,
        "datasets": datasets,
        "total_questions": total_questions,
        "completed_questions": 0,
        "current_dataset": None,
        "current_question_id": None,
        "metrics_computed": [],
        "metrics_pending": [],
        "errors": [],
        "config": config,
        "results": None,
        "last_updated": datetime.now().isoformat(),
    }

    client.set(key, json.dumps(data), ex=EVAL_PROGRESS_TTL)
    logger.info(f"[EVAL_PROGRESS] Created eval run {run_id}")


def update_eval_progress(
    run_id: str,
    status: str | None = None,
    phase: str | None = None,
    completed_questions: int | None = None,
    current_dataset: str | None = None,
    current_question_id: str | None = None,
    metrics_computed: list[str] | None = None,
    metrics_pending: list[str] | None = None,
    error: str | None = None,
) -> None:
    """Update eval run progress.

    Args:
        run_id: Unique run identifier
        status: New status (pending, running, completed, failed, cancelled)
        phase: Current phase (loading, querying, computing_metrics, completed)
        completed_questions: Number of questions completed
        current_dataset: Dataset currently being processed
        current_question_id: Question currently being evaluated
        metrics_computed: List of metrics that have been computed
        metrics_pending: List of metrics still pending
        error: Error message to append to errors list
    """
    client = get_redis_client()
    key = _get_key(run_id)

    data_json = client.get(key)
    if not data_json:
        logger.warning(f"[EVAL_PROGRESS] Run {run_id} not found in Redis")
        return

    data = json.loads(data_json)

    if status is not None:
        data["status"] = status
    if phase is not None:
        data["phase"] = phase
    if completed_questions is not None:
        data["completed_questions"] = completed_questions
    if current_dataset is not None:
        data["current_dataset"] = current_dataset
    if current_question_id is not None:
        data["current_question_id"] = current_question_id
    if metrics_computed is not None:
        data["metrics_computed"] = metrics_computed
    if metrics_pending is not None:
        data["metrics_pending"] = metrics_pending
    if error:
        data["errors"].append({
            "time": datetime.now().isoformat(),
            "error": error,
        })

    data["last_updated"] = datetime.now().isoformat()
    client.set(key, json.dumps(data), ex=EVAL_PROGRESS_TTL)


def increment_eval_progress(run_id: str) -> None:
    """Increment the completed questions counter by 1.

    Args:
        run_id: Unique run identifier
    """
    client = get_redis_client()
    key = _get_key(run_id)

    data_json = client.get(key)
    if not data_json:
        logger.warning(f"[EVAL_PROGRESS] Run {run_id} not found in Redis")
        return

    data = json.loads(data_json)
    data["completed_questions"] += 1
    data["last_updated"] = datetime.now().isoformat()
    client.set(key, json.dumps(data), ex=EVAL_PROGRESS_TTL)


def complete_eval_run(run_id: str, results: dict[str, Any]) -> None:
    """Mark eval run as completed with results.

    Args:
        run_id: Unique run identifier
        results: Evaluation results dictionary
    """
    client = get_redis_client()
    key = _get_key(run_id)

    data_json = client.get(key)
    if not data_json:
        logger.warning(f"[EVAL_PROGRESS] Run {run_id} not found in Redis")
        return

    data = json.loads(data_json)
    data["status"] = "completed"
    data["phase"] = "completed"
    data["completed_at"] = datetime.now().isoformat()
    data["results"] = results
    data["last_updated"] = datetime.now().isoformat()

    client.set(key, json.dumps(data), ex=EVAL_PROGRESS_TTL)
    logger.info(f"[EVAL_PROGRESS] Completed eval run {run_id}")


def fail_eval_run(run_id: str, error: str) -> None:
    """Mark eval run as failed.

    Args:
        run_id: Unique run identifier
        error: Error message describing the failure
    """
    client = get_redis_client()
    key = _get_key(run_id)

    data_json = client.get(key)
    if not data_json:
        logger.warning(f"[EVAL_PROGRESS] Run {run_id} not found in Redis")
        return

    data = json.loads(data_json)
    data["status"] = "failed"
    data["phase"] = "failed"
    data["completed_at"] = datetime.now().isoformat()
    data["errors"].append({
        "time": datetime.now().isoformat(),
        "error": error,
    })
    data["last_updated"] = datetime.now().isoformat()

    client.set(key, json.dumps(data), ex=EVAL_PROGRESS_TTL)
    logger.error(f"[EVAL_PROGRESS] Failed eval run {run_id}: {error}")


def cancel_eval_run(run_id: str) -> bool:
    """Mark eval run as cancelled.

    Args:
        run_id: Unique run identifier

    Returns:
        True if run was found and cancelled, False otherwise
    """
    client = get_redis_client()
    key = _get_key(run_id)

    data_json = client.get(key)
    if not data_json:
        return False

    data = json.loads(data_json)
    data["status"] = "cancelled"
    data["phase"] = "cancelled"
    data["completed_at"] = datetime.now().isoformat()
    data["last_updated"] = datetime.now().isoformat()

    client.set(key, json.dumps(data), ex=EVAL_PROGRESS_TTL)
    logger.info(f"[EVAL_PROGRESS] Cancelled eval run {run_id}")
    return True


def get_eval_progress(run_id: str) -> dict[str, Any] | None:
    """Get current progress for an eval run.

    Args:
        run_id: Unique run identifier

    Returns:
        Progress data dictionary or None if not found
    """
    client = get_redis_client()
    key = _get_key(run_id)

    data_json = client.get(key)
    if not data_json:
        return None

    return json.loads(data_json)


def delete_eval_progress(run_id: str) -> bool:
    """Delete eval run progress from Redis.

    Args:
        run_id: Unique run identifier

    Returns:
        True if deleted, False if not found
    """
    client = get_redis_client()
    key = _get_key(run_id)
    deleted = client.delete(key)
    if deleted:
        logger.info(f"[EVAL_PROGRESS] Deleted eval run {run_id}")
    return deleted > 0


def list_eval_runs(status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """List eval runs from Redis.

    Args:
        status: Optional status filter
        limit: Maximum number of runs to return

    Returns:
        List of eval run progress dictionaries
    """
    client = get_redis_client()
    runs = []

    # Scan for all eval run keys
    for key in client.scan_iter(match="eval:run:*:progress", count=100):
        if len(runs) >= limit:
            break

        data_json = client.get(key)
        if data_json:
            data = json.loads(data_json)
            if status is None or data.get("status") == status:
                runs.append(data)

    # Sort by created_at descending
    runs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return runs[:limit]
