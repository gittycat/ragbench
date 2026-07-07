"""All eval API endpoints."""

import os

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    CompareRunsResponse,
    DashboardResponse,
    DatasetInfo,
    JobCreatedResponse,
    RunDetailResponse,
    RunListResponse,
    TriggerRunRequest,
)

router = APIRouter(prefix="/eval")

# Injected at startup by app.py
_job_manager = None


def init_router(job_manager) -> None:
    global _job_manager
    _job_manager = job_manager


def _jm():
    if _job_manager is None:
        raise HTTPException(503, "Service not initialized")
    return _job_manager


# ── POST /eval/runs — trigger ─────────────────────────────────────────────


@router.post("/runs", status_code=202, response_model=JobCreatedResponse)
def trigger_run(req: TriggerRunRequest):
    jm = _jm()
    rag_url = os.environ.get("RAG_SERVER_URL", "http://localhost:8001")
    try:
        job_id = jm.trigger(
            name=req.name,
            tier=req.tier,
            datasets=req.datasets,
            samples=req.samples,
            seed=req.seed,
            judge_enabled=req.judge_enabled,
            rag_server_url=rag_url,
        )
    except RuntimeError:
        raise HTTPException(409, "An eval job is already running")
    except ValueError as e:
        raise HTTPException(422, str(e))

    return JobCreatedResponse(
        job_id=job_id,
        status="queued",
        created_at=jm.active_created_at,
    )


# ── GET /eval/runs/active — current job status ───────────────────────────


@router.get("/runs/active")
def get_active_job():
    active = _jm().get_active_job()
    if not active:
        return None  # FastAPI returns 200 with null body
    return active


# ── DELETE /eval/runs/active — cancel running job ─────────────────────────


@router.delete("/runs/active")
def cancel_active_job():
    if not _jm().cancel_active():
        raise HTTPException(404, "No active job to cancel")
    return {"status": "cancelled"}


# ── GET /eval/runs — list completed runs ──────────────────────────────────


@router.get("/runs", response_model=RunListResponse)
def list_runs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    jm = _jm()
    runs_data, total = jm.list_runs(limit=limit, offset=offset)
    summaries = [jm.run_to_summary(d) for d in runs_data]
    return RunListResponse(runs=summaries, total=total)


# ── GET /eval/runs/{run_id} — full run detail ────────────────────────────


# ── GET /eval/runs/compare — compare runs ────────────────────────────────
# Must be registered before /runs/{run_id} or "compare" gets captured as a run_id.


@router.get("/runs/compare", response_model=CompareRunsResponse)
def compare_runs(ids: str = Query(..., description="Comma-separated run IDs")):
    jm = _jm()
    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    if len(id_list) < 2:
        raise HTTPException(422, "At least 2 run IDs required")

    details = []
    summaries = []
    for rid in id_list:
        data = jm.get_run(rid)
        if not data:
            raise HTTPException(404, f"Run {rid} not found")
        details.append(jm.run_to_detail(data))
        summaries.append(jm.run_to_summary(data))

    # Compute deltas between first and second run (second minus first)
    deltas: dict[str, float | None] = {}
    if len(details) >= 2:
        d1 = details[0].duration_seconds
        d2 = details[1].duration_seconds
        deltas["duration_seconds"] = round(d2 - d1, 1) if d1 is not None and d2 is not None else None

        w1 = summaries[0].weighted_score
        w2 = summaries[1].weighted_score
        deltas["weighted_score"] = round(w2 - w1, 4) if w1 is not None and w2 is not None else None

        metrics1 = summaries[0].metrics
        metrics2 = summaries[1].metrics
        for name in set(metrics1) | set(metrics2):
            v1 = metrics1.get(name)
            v2 = metrics2.get(name)
            deltas[name] = round(v2 - v1, 4) if v1 is not None and v2 is not None else None

    return CompareRunsResponse(runs=details, deltas=deltas)


@router.get("/runs/{run_id}", response_model=RunDetailResponse)
def get_run(run_id: str):
    data = _jm().get_run(run_id)
    if not data:
        raise HTTPException(404, f"Run {run_id} not found")
    return _jm().run_to_detail(data)


# ── GET /eval/dashboard — dashboard summary ──────────────────────────────


@router.get("/dashboard", response_model=DashboardResponse)
def get_dashboard():
    jm = _jm()
    latest = jm.get_latest_run()
    runs_data, total = jm.list_runs(limit=1)
    active = jm.get_active_job()

    return DashboardResponse(
        latest_run=jm.run_to_summary(latest) if latest else None,
        total_runs=total,
        active_job=active,
    )


# ── GET /eval/datasets — available datasets ──────────────────────────────


@router.get("/datasets", response_model=list[DatasetInfo])
def list_datasets():
    from evals.config import DATASET_TIER_SUPPORT, DatasetName
    from evals.datasets.registry import list_datasets as registry_list

    raw = registry_list()
    result = []
    for ds in raw:
        ds_name = ds["name"]
        try:
            enum_val = DatasetName(ds_name)
            tiers = [t.value for t in DATASET_TIER_SUPPORT.get(enum_val, [])]
        except ValueError:
            tiers = []

        result.append(
            DatasetInfo(
                name=ds_name,
                description=ds.get("description", ""),
                source_url=ds.get("source_url", ""),
                supported_tiers=tiers,
            )
        )
    return result
