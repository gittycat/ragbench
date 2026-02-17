"""Evaluation runner - orchestrates the complete evaluation pipeline.

This module provides:
- RAG server client for querying
- Evaluation orchestration across datasets
- Metric computation and aggregation
- Weighted scoring and Pareto analysis
- Run persistence and reporting
"""

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from evals.config import (
    EvalConfig,
    DatasetName,
    DEFAULT_WEIGHTS,
)
from evals.datasets.registry import get_dataset, load_datasets
from evals.judges.llm_judge import LLMJudge
from evals.metrics import (
    METRIC_GROUPS,
    RecallAtK,
    PrecisionAtK,
    MRR,
    NDCG,
    Faithfulness,
    AnswerCorrectness,
    AnswerRelevancy,
    CitationPrecision,
    CitationRecall,
    SectionAccuracy,
    UnanswerableAccuracy,
    FalsePositiveRate,
    FalseNegativeRate,
    LatencyP50,
    LatencyP95,
    CostPerQuery,
)
from evals.schemas import (
    EvalQuestion,
    EvalResponse,
    RetrievedChunk,
    Citation,
    QueryMetrics,
    TokenUsage,
    MetricResult,
    MetricGroup,
    Scorecard,
    WeightedScore,
    ParetoPoint,
    ConfigSnapshot,
    EvalRun,
)

logger = logging.getLogger(__name__)


class RAGClient:
    """HTTP client for the RAG server."""

    def __init__(self, base_url: str = "http://localhost:8001", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def query(self, question: str, session_id: str | None = None) -> dict[str, Any]:
        """Send a query to the RAG server.

        Args:
            question: The question text
            session_id: Optional session ID for conversation continuity

        Returns:
            Raw response from the server
        """
        payload = {"query": question}
        if session_id:
            payload["session_id"] = session_id

        response = self._client.post(f"{self.base_url}/query", json=payload)
        response.raise_for_status()
        return response.json()

    def get_config(self) -> dict[str, Any]:
        """Get the current RAG server configuration."""
        response = self._client.get(f"{self.base_url}/models/info")
        response.raise_for_status()
        return response.json()

    def health_check(self) -> bool:
        """Check if the RAG server is healthy."""
        try:
            response = self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def parse_rag_response(
    question_id: str,
    raw_response: dict[str, Any],
    latency_ms: float,
) -> EvalResponse:
    """Parse a RAG server response into an EvalResponse.

    Args:
        question_id: ID of the question
        raw_response: Raw response from the RAG server
        latency_ms: Query latency in milliseconds

    Returns:
        Parsed EvalResponse
    """
    # Extract answer
    answer = raw_response.get("response", "")

    # Extract retrieved chunks
    retrieved_chunks = []
    sources = raw_response.get("sources") or []
    for i, source in enumerate(sources):
        chunk = RetrievedChunk(
            doc_id=source.get("doc_id", source.get("document_id", "")),
            chunk_id=source.get("chunk_id", source.get("node_id", f"chunk-{i}")),
            text=source.get("text", source.get("content", "")),
            score=source.get("score"),
            rank=i + 1,
            metadata=source.get("metadata", {}),
        )
        retrieved_chunks.append(chunk)

    # Extract citations (if present in response)
    citations = []
    raw_citations = raw_response.get("citations") or []
    for cit in raw_citations:
        citation = Citation(
            source_index=cit.get("source_index", 0),
            doc_id=cit.get("doc_id"),
            chunk_id=cit.get("chunk_id"),
            chunk_index=cit.get("chunk_index"),
            text_span=cit.get("text_span"),
        )
        citations.append(citation)

    # Extract token usage if available
    token_usage = None
    usage = raw_response.get("usage", {})
    if usage:
        token_usage = TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    metrics = QueryMetrics(latency_ms=latency_ms, token_usage=token_usage)

    return EvalResponse(
        question_id=question_id,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        citations=citations,
        session_id=raw_response.get("session_id"),
        metrics=metrics,
        raw_response=raw_response,
    )


class EvaluationRunner:
    """Orchestrates evaluation runs.

    Usage:
        config = EvalConfig(datasets=["ragbench", "squad_v2"])
        runner = EvaluationRunner(config)
        results = runner.run()
    """

    def __init__(self, config: EvalConfig | None = None):
        """Initialize the runner.

        Args:
            config: Evaluation configuration. Uses defaults if not provided.
        """
        self.config = config or EvalConfig()
        self._client: RAGClient | None = None
        self._judge: LLMJudge | None = None
        self._metrics: dict[MetricGroup, list] = {}
        self._model_config: dict[str, Any] = {}  # Store model config from /models/info

    @property
    def client(self) -> RAGClient:
        """Get or create the RAG client."""
        if self._client is None:
            self._client = RAGClient(
                base_url=self.config.rag_server_url,
                timeout=120.0,
            )
        return self._client

    @property
    def judge(self) -> LLMJudge:
        """Get or create the LLM judge."""
        if self._judge is None:
            self._judge = LLMJudge(self.config.judge)
        return self._judge

    def _init_metrics(self) -> None:
        """Initialize metric instances based on config."""
        self._metrics = {}

        # Retrieval metrics
        if self.config.metrics.retrieval:
            self._metrics[MetricGroup.RETRIEVAL] = [
                RecallAtK(k=k) for k in self.config.metrics.recall_k_values
            ] + [
                PrecisionAtK(k=k) for k in self.config.metrics.precision_k_values
            ] + [
                MRR(),
                NDCG(k=10),
            ]

        # Generation metrics (require judge)
        if self.config.metrics.generation and self.config.judge.enabled:
            self._metrics[MetricGroup.GENERATION] = [
                Faithfulness(),
                AnswerCorrectness(),
                AnswerRelevancy(),
            ]

        # Citation metrics
        if self.config.metrics.citation:
            self._metrics[MetricGroup.CITATION] = [
                CitationPrecision(),
                CitationRecall(),
                SectionAccuracy(),
            ]

        # Abstention metrics
        if self.config.metrics.abstention:
            self._metrics[MetricGroup.ABSTENTION] = [
                UnanswerableAccuracy(),
                FalsePositiveRate(),
                FalseNegativeRate(),
            ]

        # Performance metrics are computed separately from latencies

    def run(self, name: str | None = None) -> EvalRun:
        """Execute a complete evaluation run.

        Args:
            name: Optional name for this run

        Returns:
            Complete EvalRun with all results
        """
        run_id = str(uuid.uuid4())[:8]
        run_name = name or f"eval-{run_id}"
        created_at = datetime.now()

        logger.info(f"[EVAL] Starting run: {run_name}")

        # Check RAG server health
        if not self.client.health_check():
            raise ConnectionError(
                f"RAG server not available at {self.config.rag_server_url}"
            )

        # Get RAG server config for snapshot
        try:
            rag_config = self.client.get_config()
            self._model_config = rag_config  # Store for cost calculation
        except Exception as e:
            logger.warning(f"[EVAL] Could not get RAG config: {e}")
            rag_config = {}
            self._model_config = {}

        config_snapshot = self._create_config_snapshot(rag_config)

        # Initialize metrics
        self._init_metrics()

        # Load datasets
        logger.info(f"[EVAL] Loading datasets: {self.config.datasets}")
        datasets = load_datasets(
            self.config.datasets,
            max_samples=self.config.samples_per_dataset,
            seed=self.config.seed,
        )

        # Run queries and collect responses
        all_questions: list[EvalQuestion] = []
        all_responses: list[EvalResponse] = []
        latencies: list[float] = []
        error_count = 0

        for dataset in datasets:
            logger.info(f"[EVAL] Processing dataset: {dataset.name} ({len(dataset)} questions)")

            for question in dataset:
                try:
                    start_time = time.perf_counter()
                    raw_response = self.client.query(question.question)
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    response = parse_rag_response(
                        question_id=question.id,
                        raw_response=raw_response,
                        latency_ms=latency_ms,
                    )

                    all_questions.append(question)
                    all_responses.append(response)
                    latencies.append(latency_ms)

                except Exception as e:
                    logger.error(f"[EVAL] Query failed for {question.id}: {e}")
                    error_count += 1

        logger.info(
            f"[EVAL] Completed {len(all_responses)} queries, {error_count} errors"
        )

        # Compute metrics
        scorecard = self._compute_metrics(all_questions, all_responses, latencies)

        # Compute weighted score
        weighted_score = self._compute_weighted_score(scorecard)

        # Create run result
        eval_run = EvalRun(
            id=run_id,
            name=run_name,
            created_at=created_at,
            completed_at=datetime.now(),
            config=config_snapshot,
            datasets=[ds.value for ds in self.config.datasets],
            scorecard=scorecard,
            weighted_score=weighted_score,
            question_count=len(all_questions) + error_count,
            error_count=error_count,
            metadata={
                "samples_per_dataset": self.config.samples_per_dataset,
                "seed": self.config.seed,
            },
        )

        # Save run
        self._save_run(eval_run)

        logger.info(
            f"[EVAL] Run complete. Weighted score: {weighted_score.score:.3f}"
        )

        return eval_run

    def _create_config_snapshot(self, rag_config: dict) -> ConfigSnapshot:
        """Create a snapshot of the current RAG configuration."""
        # /models/info returns flat structure now
        return ConfigSnapshot(
            llm_model=rag_config.get("llm_model", "unknown"),
            llm_provider=rag_config.get("llm_provider", "unknown"),
            embedding_model=rag_config.get("embedding_model", "unknown"),
            reranker_model=rag_config.get("reranker_model"),
            retrieval_top_k=10,  # Not available in /models/info
            hybrid_search_enabled=False,  # Not available in /models/info
            contextual_retrieval_enabled=False,  # Not available in /models/info
            additional=rag_config,
        )

    def _compute_metrics(
        self,
        questions: list[EvalQuestion],
        responses: list[EvalResponse],
        latencies: list[float],
    ) -> Scorecard:
        """Compute all metrics for the evaluation.

        Args:
            questions: List of questions
            responses: List of RAG responses
            latencies: List of query latencies in ms

        Returns:
            Complete Scorecard with all metrics
        """
        scorecard = Scorecard()

        # Compute each metric group
        for group, metrics in self._metrics.items():
            logger.info(f"[EVAL] Computing {group.value} metrics")

            for metric in metrics:
                try:
                    # Pass judge for generation metrics
                    kwargs = {}
                    if metric.requires_judge:
                        kwargs["judge"] = self.judge

                    result = metric.compute_batch(questions, responses, **kwargs)
                    scorecard.add_metric(result)
                    logger.debug(f"[EVAL] {result.name}: {result.value:.3f}")

                except Exception as e:
                    logger.error(f"[EVAL] Failed to compute {metric.name}: {e}")

        # Add performance metrics
        if self.config.metrics.performance:
            # Latency metrics
            if latencies:
                # Latency P50
                sorted_latencies = sorted(latencies)
                p50_idx = int(len(sorted_latencies) * 0.5)
                p50 = sorted_latencies[p50_idx] if sorted_latencies else 0
                scorecard.add_metric(MetricResult(
                    name="latency_p50_ms",
                    value=p50,
                    group=MetricGroup.PERFORMANCE,
                    sample_size=len(latencies),
                ))

                # Latency P95
                p95_idx = int(len(sorted_latencies) * 0.95)
                p95 = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)] if sorted_latencies else 0
                scorecard.add_metric(MetricResult(
                    name="latency_p95_ms",
                    value=p95,
                    group=MetricGroup.PERFORMANCE,
                    sample_size=len(latencies),
                ))

                # Average latency
                avg_latency = sum(latencies) / len(latencies) if latencies else 0
                scorecard.add_metric(MetricResult(
                    name="latency_avg_ms",
                    value=avg_latency,
                    group=MetricGroup.PERFORMANCE,
                    sample_size=len(latencies),
                ))

            # Cost metrics (if model config available)
            if self._model_config:
                try:
                    cost_metric = CostPerQuery(
                        model=self._model_config.get("llm_model", "unknown"),
                        cost_per_1m_input_tokens=self._model_config.get("cost_per_1m_input_tokens", 0.0),
                        cost_per_1m_output_tokens=self._model_config.get("cost_per_1m_output_tokens", 0.0),
                    )
                    result = cost_metric.compute_batch(questions, responses)
                    scorecard.add_metric(result)
                except Exception as e:
                    logger.error(f"[EVAL] Failed to compute cost metric: {e}")

        return scorecard

    def _compute_weighted_score(self, scorecard: Scorecard) -> WeightedScore:
        """Compute the weighted overall score.

        Maps metrics to objectives and weights them according to config.
        """
        weights = self.config.weights
        contributions = {}
        objectives = {}

        # Map metric groups to objectives
        group_to_objective = {
            MetricGroup.RETRIEVAL: "retrieval",
            MetricGroup.GENERATION: "accuracy",
            MetricGroup.CITATION: "citation",
            MetricGroup.ABSTENTION: "accuracy",  # Contributes to accuracy
        }

        # Collect average scores per objective
        objective_scores: dict[str, list[float]] = {obj: [] for obj in weights}

        for metric in scorecard.metrics:
            if metric.group == MetricGroup.PERFORMANCE:
                continue  # Performance metrics handled separately

            objective = group_to_objective.get(metric.group)
            if objective and objective in objective_scores:
                # Normalize metric value to 0-1 range
                value = max(0.0, min(1.0, metric.value))
                objective_scores[objective].append(value)

        # Compute average per objective
        for obj, scores in objective_scores.items():
            if scores:
                objectives[obj] = sum(scores) / len(scores)
            else:
                objectives[obj] = 0.0

        # Handle performance objectives (latency, cost) - invert since lower is better
        # For now, assume normalized values; could be enhanced with target thresholds
        latency_metric = scorecard.get_metric("latency_p50_ms")
        if latency_metric and "latency" in weights:
            # Normalize: assume 5000ms is worst case, 0ms is best
            normalized_latency = 1.0 - min(latency_metric.value / 5000, 1.0)
            objectives["latency"] = normalized_latency

        # Cost objective (if we have token usage)
        if "cost" in weights:
            # Placeholder: would need token usage aggregation
            objectives["cost"] = 1.0  # Assume free/local by default

        # Compute weighted score
        total_weight = sum(weights.get(obj, 0) for obj in objectives)
        if total_weight == 0:
            return WeightedScore(score=0.0, weights=weights, objectives=objectives)

        weighted_sum = 0.0
        for obj, value in objectives.items():
            weight = weights.get(obj, 0)
            contribution = value * weight
            contributions[obj] = contribution
            weighted_sum += contribution

        final_score = weighted_sum / total_weight

        return WeightedScore(
            score=final_score,
            weights=weights,
            contributions=contributions,
            objectives=objectives,
        )

    def _save_run(self, run: EvalRun) -> Path:
        """Save evaluation run to disk.

        Args:
            run: The evaluation run to save

        Returns:
            Path to the saved file
        """
        self.config.runs_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{run.id}_{run.created_at.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.config.runs_dir / filename

        # Convert to serializable dict
        run_dict = self._run_to_dict(run)

        with open(filepath, "w") as f:
            json.dump(run_dict, f, indent=2, default=str)

        logger.info(f"[EVAL] Saved run to {filepath}")
        return filepath

    def _run_to_dict(self, run: EvalRun) -> dict[str, Any]:
        """Convert EvalRun to a JSON-serializable dictionary."""
        return {
            "id": run.id,
            "name": run.name,
            "created_at": run.created_at.isoformat(),
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "config": {
                "llm_model": run.config.llm_model,
                "llm_provider": run.config.llm_provider,
                "embedding_model": run.config.embedding_model,
                "reranker_model": run.config.reranker_model,
                "retrieval_top_k": run.config.retrieval_top_k,
                "hybrid_search_enabled": run.config.hybrid_search_enabled,
                "contextual_retrieval_enabled": run.config.contextual_retrieval_enabled,
            },
            "datasets": run.datasets,
            "scorecard": self._scorecard_to_dict(run.scorecard) if run.scorecard else None,
            "weighted_score": {
                "score": run.weighted_score.score,
                "weights": run.weighted_score.weights,
                "contributions": run.weighted_score.contributions,
                "objectives": run.weighted_score.objectives,
            } if run.weighted_score else None,
            "question_count": run.question_count,
            "error_count": run.error_count,
            "metadata": run.metadata,
        }

    def _scorecard_to_dict(self, scorecard: Scorecard) -> dict[str, Any]:
        """Convert Scorecard to a JSON-serializable dictionary."""
        return {
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "group": m.group.value,
                    "sample_size": m.sample_size,
                    "details": m.details,
                }
                for m in scorecard.metrics
            ],
            "by_group": {
                group.value: [m.name for m in metrics]
                for group, metrics in scorecard.by_group.items()
            },
        }

    def close(self):
        """Clean up resources."""
        if self._client:
            self._client.close()


def run_evaluation(config: EvalConfig | None = None) -> EvalRun:
    """Convenience function to run an evaluation.

    Args:
        config: Evaluation configuration

    Returns:
        Complete EvalRun with results
    """
    runner = EvaluationRunner(config)
    try:
        return runner.run()
    finally:
        runner.close()


def compute_pareto_frontier(runs: list[EvalRun]) -> list[ParetoPoint]:
    """Compute the Pareto frontier across multiple runs.

    A run is Pareto-optimal if no other run dominates it
    (better in at least one objective without being worse in any).

    Args:
        runs: List of evaluation runs to compare

    Returns:
        List of ParetoPoints, with is_dominated flag set appropriately
    """
    points = []

    for run in runs:
        if not run.weighted_score:
            continue

        point = ParetoPoint(
            run_id=run.id,
            config_name=run.name,
            objectives=run.weighted_score.objectives.copy(),
        )
        points.append(point)

    # Determine dominance
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i == j:
                continue

            # Check if p2 dominates p1
            better_in_one = False
            worse_in_one = False

            for obj in p1.objectives:
                v1 = p1.objectives.get(obj, 0)
                v2 = p2.objectives.get(obj, 0)

                if v2 > v1:
                    better_in_one = True
                elif v2 < v1:
                    worse_in_one = True

            if better_in_one and not worse_in_one:
                p1.is_dominated = True
                p2.dominates.append(p1.run_id)

    return points
