"""FastAPI application for the eval service."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from api.job_manager import JobManager
from api.routes import init_router, router
from infrastructure.settings import init_settings

logger = logging.getLogger(__name__)

RUNS_DIR = Path("data/eval_runs")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init settings, index past runs."""
    init_settings()
    jm = JobManager(runs_dir=RUNS_DIR)
    count = jm.index_existing_runs()
    init_router(jm)
    logger.info(f"Eval API ready — {count} past runs indexed")
    yield


app = FastAPI(
    title="RAGBench Eval Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
