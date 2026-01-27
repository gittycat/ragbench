"""Dataset loaders for evaluation."""

from evals.datasets.base import BaseDatasetLoader
from evals.datasets.registry import (
    register,
    get_loader,
    list_available,
    get_metadata,
    get_dataset,
    list_datasets,
    load_datasets,
)
from evals.datasets.ragbench import RAGBenchLoader
from evals.datasets.qasper import QasperLoader
from evals.datasets.squad_v2 import SquadV2Loader
from evals.datasets.hotpotqa import HotpotQALoader
from evals.datasets.msmarco import MSMarcoLoader

__all__ = [
    "BaseDatasetLoader",
    "register",
    "get_loader",
    "list_available",
    "get_metadata",
    "get_dataset",
    "list_datasets",
    "load_datasets",
    "RAGBenchLoader",
    "QasperLoader",
    "SquadV2Loader",
    "HotpotQALoader",
    "MSMarcoLoader",
]
