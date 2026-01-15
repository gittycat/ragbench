"""Dataset registry for managing available datasets."""

from typing import TYPE_CHECKING

from evaluation_cc.config import DatasetName
from evaluation_cc.schemas import EvalDataset

if TYPE_CHECKING:
    from evaluation_cc.datasets.base import BaseDatasetLoader

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
    from evaluation_cc.datasets.ragbench import RAGBenchLoader
    from evaluation_cc.datasets.qasper import QasperLoader
    from evaluation_cc.datasets.squad_v2 import SquadV2Loader
    from evaluation_cc.datasets.hotpotqa import HotpotQALoader
    from evaluation_cc.datasets.msmarco import MSMarcoLoader

    register(DatasetName.RAGBENCH, RAGBenchLoader)
    register(DatasetName.QASPER, QasperLoader)
    register(DatasetName.SQUAD_V2, SquadV2Loader)
    register(DatasetName.HOTPOTQA, HotpotQALoader)
    register(DatasetName.MSMARCO, MSMarcoLoader)


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
    """Load multiple datasets."""
    return [
        get_dataset(name, split=split, max_samples=max_samples, seed=seed)
        for name in names
    ]
