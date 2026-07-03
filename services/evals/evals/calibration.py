"""Judge calibration against RAGBench TRACe ground-truth annotations.

RAGBench ships human-verified (GPT-4-annotated) labels for each
(question, documents, response) triple:
- adherence_score (bool): is the response fully grounded in the documents
- relevance_score (float 0-1): fraction of context relevant to the question

This module runs our LLM judge on the *reference* responses and compares its
scores to those labels, following the paper's methodology (RMSE for continuous
scores, accuracy/AUROC-style agreement for adherence). It answers: "how much
can we trust the judge scores reported by our eval runs?"
"""

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from evals.config import JudgeConfig
from evals.judges.llm_judge import LLMJudge

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/calibration")


@dataclass
class CalibrationResult:
    """Aggregated judge-vs-ground-truth agreement."""

    sample_count: int
    adherence_accuracy: float | None
    adherence_rmse: float | None
    relevance_rmse: float | None
    judge_model: str
    per_item: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _rmse(pairs: list[tuple[float, float]]) -> float | None:
    if not pairs:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in pairs) / len(pairs))


async def calibrate_judge(
    items: list[dict[str, Any]],
    judge_config: JudgeConfig | None = None,
    concurrency: int = 10,
    progress_callback: Any | None = None,
) -> CalibrationResult:
    """Score RAGBench reference responses with the judge and compare to labels.

    Args:
        items: Raw RAGBench items (from RAGBenchLoader.load_raw_items)
        judge_config: Judge configuration (default from models config)
        concurrency: Max concurrent judge calls
        progress_callback: Called with (completed_count) after each item
    """
    judge = LLMJudge(judge_config)
    sem = asyncio.Semaphore(concurrency)
    completed = 0

    async def _judge_one(item: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal completed
        async with sem:
            try:
                question = item.get("question", "")
                response = item.get("response", "")
                documents = item.get("documents") or []
                context = "\n\n".join(
                    d if isinstance(d, str) else str(d) for d in documents
                )
                if not (question and response and context):
                    return None

                faith_task = judge.evaluate_faithfulness(answer=response, context=context)
                rel_task = judge.evaluate_context_relevance(question=question, context=context)
                faith, rel = await asyncio.gather(faith_task, rel_task)

                return {
                    "id": item.get("id"),
                    "subset": item.get("subset"),
                    "judge_faithfulness": faith.score,
                    "judge_context_relevance": rel.score,
                    "gt_adherence": item.get("adherence_score"),
                    "gt_relevance": item.get("relevance_score"),
                    "gt_utilization": item.get("utilization_score"),
                    "gt_completeness": item.get("completeness_score"),
                }
            except Exception as e:
                logger.warning(f"[CALIBRATION] Failed for item {item.get('id')}: {e}")
                return None
            finally:
                completed += 1
                if progress_callback:
                    progress_callback(completed)

    results = [r for r in await asyncio.gather(*(_judge_one(i) for i in items)) if r]

    # Adherence: ground truth is boolean; judge faithfulness thresholded at 0.5
    adherence_pairs = [
        (r["judge_faithfulness"], 1.0 if r["gt_adherence"] else 0.0)
        for r in results
        if r["gt_adherence"] is not None
    ]
    adherence_accuracy = None
    if adherence_pairs:
        correct = sum(1 for judged, gt in adherence_pairs if (judged >= 0.5) == (gt >= 0.5))
        adherence_accuracy = correct / len(adherence_pairs)

    relevance_pairs = [
        (r["judge_context_relevance"], r["gt_relevance"])
        for r in results
        if r["gt_relevance"] is not None
    ]

    return CalibrationResult(
        sample_count=len(results),
        adherence_accuracy=adherence_accuracy,
        adherence_rmse=_rmse(adherence_pairs),
        relevance_rmse=_rmse(relevance_pairs),
        judge_model=judge.config.model,
        per_item=results,
        metadata={
            "adherence_sample_count": len(adherence_pairs),
            "relevance_sample_count": len(relevance_pairs),
        },
    )


def save_calibration(result: CalibrationResult, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(
            {
                "sample_count": result.sample_count,
                "judge_model": result.judge_model,
                "adherence_accuracy": result.adherence_accuracy,
                "adherence_rmse": result.adherence_rmse,
                "relevance_rmse": result.relevance_rmse,
                "metadata": result.metadata,
                "per_item": result.per_item,
            },
            f,
            indent=2,
        )
    logger.info(f"[CALIBRATION] Saved to {path}")
    return path
