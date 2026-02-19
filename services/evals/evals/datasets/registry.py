"""Dataset registry for managing available datasets."""

import logging
from typing import TYPE_CHECKING

from evals.config import DatasetName
from evals.schemas import EvalDataset

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from evals.datasets.base import BaseDatasetLoader

# Module-level registry state
_loaders: dict[DatasetName, type["BaseDatasetLoader"]] = {}
_instances: dict[DatasetName, "BaseDatasetLoader"] = {}


def register(name: DatasetName, loader_class: type["BaseDatasetLoader"]) -> None:
    """Register a dataset loader class."""
    _loaders[name] = loader_class


def get_loader(name: DatasetName) -> "BaseDatasetLoader":
    """Get a dataset loader instance (cached)."""
    if name not in _instances:
        if name not in _loaders:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(_loaders.keys())}")
        _instances[name] = _loaders[name]()
    return _instances[name]


def list_available() -> list[DatasetName]:
    """List all registered datasets."""
    return list(_loaders.keys())


def get_metadata(name: DatasetName) -> dict:
    """Get metadata for a dataset."""
    loader = get_loader(name)
    return loader.get_metadata()


def _register_default_loaders() -> None:
    """Register all default dataset loaders."""
    from evals.datasets.ragbench import RAGBenchLoader
    from evals.datasets.qasper import QasperLoader
    from evals.datasets.squad_v2 import SquadV2Loader
    from evals.datasets.hotpotqa import HotpotQALoader
    from evals.datasets.msmarco import MSMarcoLoader
    from evals.datasets.golden import GoldenDatasetLoader

    register(DatasetName.RAGBENCH, RAGBenchLoader)
    register(DatasetName.QASPER, QasperLoader)
    register(DatasetName.SQUAD_V2, SquadV2Loader)
    register(DatasetName.HOTPOTQA, HotpotQALoader)
    register(DatasetName.MSMARCO, MSMarcoLoader)
    register(DatasetName.GOLDEN, GoldenDatasetLoader)


# Register loaders on import
_register_default_loaders()


def get_dataset(
    name: DatasetName | str,
    split: str = "test",
    max_samples: int | None = None,
    seed: int | None = None,
) -> EvalDataset:
    """Load a dataset by name."""
    if isinstance(name, str):
        name = DatasetName(name)

    loader = get_loader(name)
    return loader.load(split=split, max_samples=max_samples, seed=seed)


def list_datasets() -> list[dict]:
    """List all available datasets with metadata."""
    return [get_metadata(name) for name in list_available()]


def load_datasets(
    names: list[DatasetName | str],
    split: str = "test",
    max_samples: int | None = None,
    seed: int | None = None,
) -> list[EvalDataset]:
    """Load multiple datasets, skipping any that fail to load."""
    results = []
    for name in names:
        try:
            results.append(get_dataset(name, split=split, max_samples=max_samples, seed=seed))
        except Exception as e:
            logger.warning(f"[REGISTRY] Skipping dataset '{name}': {e}")
            print(f"\nWARNING: Skipping dataset '{name}': {e}\n")
    return results
