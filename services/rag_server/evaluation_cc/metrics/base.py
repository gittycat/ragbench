"""Base class for evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any

from evaluation_cc.schemas import EvalQuestion, EvalResponse, MetricResult, MetricGroup


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics.

    Each metric is responsible for:
    1. Computing a single metric value from question/response pairs
    2. Providing metadata about the metric
    3. Supporting batch computation for efficiency
    """

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
        """Human-readable description of this metric."""
        return ""

    @property
    def requires_gold(self) -> bool:
        """Whether this metric requires gold passages/answers."""
        return True

    @property
    def requires_judge(self) -> bool:
        """Whether this metric requires an LLM judge."""
        return False

    @abstractmethod
    def compute(
        self,
        question: EvalQuestion,
        response: EvalResponse,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute the metric for a single question/response pair.

        Args:
            question: The evaluation question with gold data
            response: The RAG system's response
            **kwargs: Additional metric-specific parameters

        Returns:
            MetricResult with the computed value
        """
        ...

    def compute_batch(
        self,
        questions: list[EvalQuestion],
        responses: list[EvalResponse],
        **kwargs: Any,
    ) -> MetricResult:
        """Compute the metric across a batch of question/response pairs.

        Default implementation averages individual results.
        Override for more efficient batch computation.

        Args:
            questions: List of evaluation questions
            responses: List of RAG responses

        Returns:
            Aggregated MetricResult
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

        # Compute individual results
        results = []
        for q, r in zip(questions, responses):
            try:
                result = self.compute(q, r, **kwargs)
                results.append(result)
            except Exception as e:
                # Skip failed computations but log them
                import logging
                logging.getLogger(__name__).warning(
                    f"[METRIC] {self.name} failed for question {q.id}: {e}"
                )

        if not results:
            return MetricResult(
                name=self.name,
                value=0.0,
                group=self.group,
                sample_size=0,
                details={"error": "All computations failed"},
            )

        # Aggregate
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
