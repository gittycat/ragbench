"""Dataset loaders for evaluation."""

from evaluation_cc.datasets.base import BaseDatasetLoader
from evaluation_cc.datasets.registry import (
    register,
    get_loader,
    list_available,
    get_metadata,
    get_dataset,
    list_datasets,
    load_datasets,
)
from evaluation_cc.datasets.ragbench import RAGBenchLoader
from evaluation_cc.datasets.qasper import QasperLoader
from evaluation_cc.datasets.squad_v2 import SquadV2Loader
from evaluation_cc.datasets.hotpotqa import HotpotQALoader
from evaluation_cc.datasets.msmarco import MSMarcoLoader

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
