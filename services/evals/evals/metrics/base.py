"""Base class for evaluation metrics."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from evals.schemas import EvalQuestion, EvalResponse, MetricResult, MetricGroup

logger = logging.getLogger(__name__)


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this metric."""
        ...

    @property
    @abstractmethod
    def group(self) -> MetricGroup:
        """Which metric group this belongs to."""
        ...

    @property
    def description(self) -> str:
        return ""

    @property
    def requires_gold(self) -> bool:
        return True

    @property
    def requires_judge(self) -> bool:
        return False

    @abstractmethod
    def compute(
        self,
        question: EvalQuestion,
        response: EvalResponse,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute the metric for a single question/response pair."""
        ...

    async def compute_batch(
        self,
        questions: list[EvalQuestion],
        responses: list[EvalResponse],
        progress_callback: Any | None = None,
        concurrency: int = 10,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute the metric across a batch of question/response pairs.

        For judge metrics, runs concurrently with semaphore.
        For non-judge metrics, runs sequentially.
        """
        if len(questions) != len(responses):
            raise ValueError(
                f"Questions and responses must have same length: "
                f"{len(questions)} vs {len(responses)}"
            )

        if not questions:
            return MetricResult(
                name=self.name,
                value=0.0,
                group=self.group,
                sample_size=0,
            )

        results: list[MetricResult] = []

        if self.requires_judge:
            # Concurrent execution with semaphore for I/O-bound judge calls
            sem = asyncio.Semaphore(concurrency)
            completed_count = 0

            async def _run_one(q: EvalQuestion, r: EvalResponse) -> MetricResult | None:
                nonlocal completed_count
                async with sem:
                    try:
                        result = self.compute(q, r, **kwargs)
                        if asyncio.iscoroutine(result):
                            result = await result
                        return result
                    except Exception as e:
                        logger.warning(f"[METRIC] {self.name} failed for question {q.id}: {e}")
                        return None
                    finally:
                        completed_count += 1
                        if progress_callback:
                            progress_callback(completed_count)

            tasks = [_run_one(q, r) for q, r in zip(questions, responses)]
            task_results = await asyncio.gather(*tasks)
            results = [r for r in task_results if r is not None]
        else:
            # Sequential execution for CPU-bound metrics
            for i, (q, r) in enumerate(zip(questions, responses)):
                try:
                    result = self.compute(q, r, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"[METRIC] {self.name} failed for question {q.id}: {e}")
                if progress_callback:
                    progress_callback(i + 1)

        if not results:
            return MetricResult(
                name=self.name,
                value=0.0,
                group=self.group,
                sample_size=0,
                details={"error": "All computations failed"},
            )

        avg_value = sum(r.value for r in results) / len(results)

        return MetricResult(
            name=self.name,
            value=avg_value,
            group=self.group,
            sample_size=len(results),
            details={
                "individual_scores": [r.value for r in results],
                "std_dev": self._compute_std(results),
            },
        )

    def _compute_std(self, results: list[MetricResult]) -> float:
        """Compute standard deviation of metric values."""
        if len(results) < 2:
            return 0.0

        values = [r.value for r in results]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, group={self.group.value})"
