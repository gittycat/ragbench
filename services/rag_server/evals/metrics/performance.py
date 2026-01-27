"""Performance metrics for latency and cost tracking."""

import statistics
from typing import Any

from evals.metrics.base import BaseMetric
from evals.config import get_model_cost
from evals.schemas import (
    EvalQuestion,
    EvalResponse,
    MetricResult,
    MetricGroup,
)


class LatencyP50(BaseMetric):
    """P50 (median) latency in milliseconds.

    Lower is better.
    """

    @property
    def name(self) -> str:
        return "latency_p50"

    @property
    def group(self) -> MetricGroup:
        return MetricGroup.PERFORMANCE

    @property
    def description(self) -> str:
        return "Median query latency in milliseconds"

    @property
    def requires_gold(self) -> bool:
        return False

    def compute(
        self,
        question: EvalQuestion,
        response: EvalResponse,
        **kwargs: Any,
    ) -> MetricResult:
        if response.metrics and response.metrics.latency_ms:
            latency = response.metrics.latency_ms
        else:
            latency = 0.0

        return MetricResult(
            name=self.name,
            value=latency,
            group=self.group,
            sample_size=1,
            details={"latency_ms": latency},
        )

    def compute_batch(
        self,
        questions: list[EvalQuestion],
        responses: list[EvalResponse],
        **kwargs: Any,
    ) -> MetricResult:
        """Compute P50 latency across batch."""
        latencies = []
        for r in responses:
            if r.metrics and r.metrics.latency_ms:
                latencies.append(r.metrics.latency_ms)

        if not latencies:
            return MetricResult(
                name=self.name,
                value=0.0,
                group=self.group,
                sample_size=0,
                details={"note": "No latency data available"},
            )

        p50 = statistics.median(latencies)

        return MetricResult(
            name=self.name,
            value=p50,
            group=self.group,
            sample_size=len(latencies),
            details={
                "p50_ms": p50,
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "mean_ms": statistics.mean(latencies),
            },
        )


class LatencyP95(BaseMetric):
    """P95 latency in milliseconds.

    Lower is better. Captures tail latency.
    """

    @property
    def name(self) -> str:
        return "latency_p95"

    @property
    def group(self) -> MetricGroup:
        return MetricGroup.PERFORMANCE

    @property
    def description(self) -> str:
        return "95th percentile query latency in milliseconds"

    @property
    def requires_gold(self) -> bool:
        return False

    def compute(
        self,
        question: EvalQuestion,
        response: EvalResponse,
        **kwargs: Any,
    ) -> MetricResult:
        # Single sample - return the latency value
        if response.metrics and response.metrics.latency_ms:
            latency = response.metrics.latency_ms
        else:
            latency = 0.0

        return MetricResult(
            name=self.name,
            value=latency,
            group=self.group,
            sample_size=1,
            details={"latency_ms": latency},
        )

    def compute_batch(
        self,
        questions: list[EvalQuestion],
        responses: list[EvalResponse],
        **kwargs: Any,
    ) -> MetricResult:
        """Compute P95 latency across batch."""
        latencies = []
        for r in responses:
            if r.metrics and r.metrics.latency_ms:
                latencies.append(r.metrics.latency_ms)

        if not latencies:
            return MetricResult(
                name=self.name,
                value=0.0,
                group=self.group,
                sample_size=0,
                details={"note": "No latency data available"},
            )

        # Calculate P95
        sorted_latencies = sorted(latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p95 = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]

        return MetricResult(
            name=self.name,
            value=p95,
            group=self.group,
            sample_size=len(latencies),
            details={
                "p95_ms": p95,
                "p50_ms": statistics.median(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
            },
        )


class CostPerQuery(BaseMetric):
    """Average cost per query in USD.

    Lower is better.
    """

    def __init__(self, model: str | None = None):
        """Initialize with optional model name for cost lookup.

        Args:
            model: Model name for cost calculation. If None, uses config.
        """
        self._model = model

    @property
    def model(self) -> str:
        if self._model is None:
            from infrastructure.config.models_config import get_models_config
            config = get_models_config()
            self._model = config.llm.model
        return self._model

    @property
    def name(self) -> str:
        return "cost_per_query"

    @property
    def group(self) -> MetricGroup:
        return MetricGroup.PERFORMANCE

    @property
    def description(self) -> str:
        return "Average cost per query in USD"

    @property
    def requires_gold(self) -> bool:
        return False

    def compute(
        self,
        question: EvalQuestion,
        response: EvalResponse,
        **kwargs: Any,
    ) -> MetricResult:
        cost = 0.0

        if response.metrics and response.metrics.token_usage:
            usage = response.metrics.token_usage
            cost = get_model_cost(
                model=self.model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )

        return MetricResult(
            name=self.name,
            value=cost,
            group=self.group,
            sample_size=1,
            details={
                "cost_usd": cost,
                "model": self.model,
                "prompt_tokens": (
                    response.metrics.token_usage.prompt_tokens
                    if response.metrics and response.metrics.token_usage
                    else 0
                ),
                "completion_tokens": (
                    response.metrics.token_usage.completion_tokens
                    if response.metrics and response.metrics.token_usage
                    else 0
                ),
            },
        )

    def compute_batch(
        self,
        questions: list[EvalQuestion],
        responses: list[EvalResponse],
        **kwargs: Any,
    ) -> MetricResult:
        """Compute average cost per query across batch."""
        costs = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for r in responses:
            if r.metrics and r.metrics.token_usage:
                usage = r.metrics.token_usage
                cost = get_model_cost(
                    model=self.model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                )
                costs.append(cost)
                total_prompt_tokens += usage.prompt_tokens
                total_completion_tokens += usage.completion_tokens

        if not costs:
            return MetricResult(
                name=self.name,
                value=0.0,
                group=self.group,
                sample_size=0,
                details={"note": "No token usage data available"},
            )

        avg_cost = sum(costs) / len(costs)
        total_cost = sum(costs)

        return MetricResult(
            name=self.name,
            value=avg_cost,
            group=self.group,
            sample_size=len(costs),
            details={
                "avg_cost_usd": avg_cost,
                "total_cost_usd": total_cost,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "model": self.model,
            },
        )
