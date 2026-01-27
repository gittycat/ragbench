"""
evals - RAG Evaluation Framework

A comprehensive evaluation framework for RAG systems supporting:
- Multiple evaluation datasets (RAGBench, Qasper, SQuAD v2, HotpotQA, MS MARCO)
- Retrieval, generation, citation, and abstention metrics
- LLM-as-judge evaluation with configurable models
- Weighted scoring with Pareto optimization
- Manual review export for high-stakes domains

Usage:
    from evals import EvalConfig, run_evaluation

    config = EvalConfig(
        datasets=["ragbench", "squad_v2"],
        samples_per_dataset=100,
    )
    results = run_evaluation(config)

CLI Usage:
    # Run evaluation
    python -m evals.cli eval --datasets ragbench --samples 10

    # Show dataset stats
    python -m evals.cli stats

    # Export for manual review
    python -m evals.cli export --run-id abc123 --format markdown
"""

from evals.config import (
    EvalConfig,
    DatasetName,
    MetricConfig,
    JudgeConfig,
    DEFAULT_WEIGHTS,
    DATASET_ASPECTS,
    get_model_cost,
)
from evals.schemas import (
    # Dataset schemas
    EvalQuestion,
    GoldPassage,
    EvalDataset,
    QueryType,
    Difficulty,
    # Response schemas
    EvalResponse,
    Citation,
    RetrievedChunk,
    TokenUsage,
    QueryMetrics,
    # Result schemas
    MetricResult,
    MetricGroup,
    Scorecard,
    WeightedScore,
    ParetoPoint,
    ConfigSnapshot,
    EvalRun,
)
from evals.runner import (
    run_evaluation,
    EvaluationRunner,
    RAGClient,
    compute_pareto_frontier,
)
from evals.export import (
    export_for_review,
    export_scorecard,
    export_run_report,
)
from evals.datasets.registry import (
    register as register_dataset,
    get_loader as get_dataset_loader,
    list_available as list_available_datasets,
    get_metadata as get_dataset_metadata,
    get_dataset,
    list_datasets,
    load_datasets,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "EvalConfig",
    "DatasetName",
    "MetricConfig",
    "JudgeConfig",
    "DEFAULT_WEIGHTS",
    "DATASET_ASPECTS",
    "get_model_cost",
    # Dataset schemas
    "EvalQuestion",
    "GoldPassage",
    "EvalDataset",
    "QueryType",
    "Difficulty",
    # Response schemas
    "EvalResponse",
    "Citation",
    "RetrievedChunk",
    "TokenUsage",
    "QueryMetrics",
    # Result schemas
    "MetricResult",
    "MetricGroup",
    "Scorecard",
    "WeightedScore",
    "ParetoPoint",
    "ConfigSnapshot",
    "EvalRun",
    # Runner
    "run_evaluation",
    "EvaluationRunner",
    "RAGClient",
    "compute_pareto_frontier",
    # Export
    "export_for_review",
    "export_scorecard",
    "export_run_report",
    # Dataset utilities
    "register_dataset",
    "get_dataset_loader",
    "list_available_datasets",
    "get_dataset_metadata",
    "get_dataset",
    "list_datasets",
    "load_datasets",
]
