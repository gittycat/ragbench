"""Dataset registry for managing available datasets."""

from typing import TYPE_CHECKING

from evaluation_cc.config import DatasetName
from evaluation_cc.schemas import EvalDataset

if TYPE_CHECKING:
    from evaluation_cc.datasets.base import BaseDatasetLoader


class DatasetRegistry:
    """Registry for dataset loaders.

    Provides a central place to register and retrieve dataset loaders.
    Loaders are registered lazily to avoid importing heavy dependencies.
    """

    _loaders: dict[DatasetName, type["BaseDatasetLoader"]] = {}
    _instances: dict[DatasetName, "BaseDatasetLoader"] = {}

    @classmethod
    def register(cls, name: DatasetName, loader_class: type["BaseDatasetLoader"]) -> None:
        """Register a dataset loader class."""
        cls._loaders[name] = loader_class

    @classmethod
    def get_loader(cls, name: DatasetName) -> "BaseDatasetLoader":
        """Get a dataset loader instance (cached)."""
        if name not in cls._instances:
            if name not in cls._loaders:
                raise ValueError(f"Unknown dataset: {name}. Available: {list(cls._loaders.keys())}")
            cls._instances[name] = cls._loaders[name]()
        return cls._instances[name]

    @classmethod
    def list_available(cls) -> list[DatasetName]:
        """List all registered datasets."""
        return list(cls._loaders.keys())

    @classmethod
    def get_metadata(cls, name: DatasetName) -> dict:
        """Get metadata for a dataset."""
        loader = cls.get_loader(name)
        return loader.get_metadata()


def _register_default_loaders() -> None:
    """Register all default dataset loaders."""
    from evaluation_cc.datasets.ragbench import RAGBenchLoader
    from evaluation_cc.datasets.qasper import QasperLoader
    from evaluation_cc.datasets.squad_v2 import SquadV2Loader
    from evaluation_cc.datasets.hotpotqa import HotpotQALoader
    from evaluation_cc.datasets.msmarco import MSMarcoLoader

    DatasetRegistry.register(DatasetName.RAGBENCH, RAGBenchLoader)
    DatasetRegistry.register(DatasetName.QASPER, QasperLoader)
    DatasetRegistry.register(DatasetName.SQUAD_V2, SquadV2Loader)
    DatasetRegistry.register(DatasetName.HOTPOTQA, HotpotQALoader)
    DatasetRegistry.register(DatasetName.MSMARCO, MSMarcoLoader)


# Register loaders on import
_register_default_loaders()


def get_dataset(
    name: DatasetName | str,
    split: str = "test",
    max_samples: int | None = None,
    seed: int | None = None,
) -> EvalDataset:
    """Load a dataset by name.

    Args:
        name: Dataset name (e.g., "ragbench", "squad_v2")
        split: Which split to load
        max_samples: Maximum samples to load
        seed: Random seed for sampling

    Returns:
        Loaded EvalDataset
    """
    if isinstance(name, str):
        name = DatasetName(name)

    loader = DatasetRegistry.get_loader(name)
    return loader.load(split=split, max_samples=max_samples, seed=seed)


def list_datasets() -> list[dict]:
    """List all available datasets with metadata."""
    return [
        DatasetRegistry.get_metadata(name)
        for name in DatasetRegistry.list_available()
    ]


def load_datasets(
    names: list[DatasetName | str],
    split: str = "test",
    max_samples: int | None = None,
    seed: int | None = None,
) -> list[EvalDataset]:
    """Load multiple datasets.

    Args:
        names: List of dataset names
        split: Which split to load
        max_samples: Maximum samples per dataset
        seed: Random seed for sampling

    Returns:
        List of loaded EvalDatasets
    """
    return [
        get_dataset(name, split=split, max_samples=max_samples, seed=seed)
        for name in names
    ]
