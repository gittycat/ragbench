"""Dataset registry for managing available datasets."""

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evals.config import DatasetName
from evals.schemas import EvalDataset
from evals.schemas.dataset import EvalQuestion, GoldPassage, QueryType, Difficulty

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from evals.datasets.base import BaseDatasetLoader

# Module-level registry state
_loaders: dict[DatasetName, type["BaseDatasetLoader"]] = {}
_instances: dict[DatasetName, "BaseDatasetLoader"] = {}

CACHE_DIR = Path("data/dataset_cache")


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


def _cache_key(name: str, split: str, max_samples: int | None, seed: int | None) -> str:
    raw = f"{name}|{split}|{max_samples}|{seed}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _dataset_to_dict(ds: EvalDataset) -> dict[str, Any]:
    return {
        "name": ds.name,
        "version": ds.version,
        "description": ds.description,
        "source_url": ds.source_url,
        "domains": ds.domains,
        "metadata": ds.metadata,
        "questions": [
            {
                "id": q.id,
                "question": q.question,
                "expected_answer": q.expected_answer,
                "query_type": q.query_type.value,
                "difficulty": q.difficulty.value,
                "domain": q.domain,
                "is_unanswerable": q.is_unanswerable,
                "metadata": q.metadata,
                "gold_passages": [
                    {
                        "doc_id": gp.doc_id,
                        "chunk_id": gp.chunk_id,
                        "text": gp.text,
                        "relevance_score": gp.relevance_score,
                    }
                    for gp in q.gold_passages
                ],
            }
            for q in ds.questions
        ],
    }


def _dataset_from_dict(d: dict[str, Any]) -> EvalDataset:
    questions = []
    for q in d["questions"]:
        gold_passages = [
            GoldPassage(
                doc_id=gp["doc_id"],
                chunk_id=gp["chunk_id"],
                text=gp["text"],
                relevance_score=gp.get("relevance_score", 1.0),
            )
            for gp in q.get("gold_passages", [])
        ]
        questions.append(EvalQuestion(
            id=q["id"],
            question=q["question"],
            expected_answer=q.get("expected_answer"),
            gold_passages=gold_passages,
            query_type=QueryType(q.get("query_type", "factoid")),
            difficulty=Difficulty(q.get("difficulty", "medium")),
            domain=q.get("domain", "general"),
            is_unanswerable=q.get("is_unanswerable", False),
            metadata=q.get("metadata", {}),
        ))
    return EvalDataset(
        name=d["name"],
        version=d["version"],
        questions=questions,
        description=d.get("description", ""),
        source_url=d.get("source_url", ""),
        domains=d.get("domains", []),
        metadata=d.get("metadata", {}),
    )


def _read_cache(key: str) -> EvalDataset | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return _dataset_from_dict(json.load(f))
    except Exception as e:
        logger.warning(f"[REGISTRY] Corrupt cache file {path}, ignoring: {e}")
        path.unlink(missing_ok=True)
        return None


def _write_cache(key: str, ds: EvalDataset) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"{key}.json"
        with open(path, "w") as f:
            json.dump(_dataset_to_dict(ds), f, separators=(",", ":"))
        logger.info(f"[REGISTRY] Cached dataset to {path}")
    except Exception as e:
        logger.warning(f"[REGISTRY] Failed to write cache: {e}")


def clear_cache() -> int:
    """Delete all cached datasets. Returns number of files removed."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
        count += 1
    return count


def get_dataset(
    name: DatasetName | str,
    split: str = "test",
    max_samples: int | None = None,
    seed: int | None = None,
    use_cache: bool = True,
) -> EvalDataset:
    """Load a dataset by name, with disk caching."""
    if isinstance(name, str):
        name = DatasetName(name)

    key = _cache_key(name.value, split, max_samples, seed)

    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            logger.info(f"[REGISTRY] Cache hit for {name.value} ({key})")
            return cached

    loader = get_loader(name)
    ds = loader.load(split=split, max_samples=max_samples, seed=seed)

    if use_cache:
        _write_cache(key, ds)

    return ds


def list_datasets() -> list[dict]:
    """List all available datasets with metadata."""
    return [get_metadata(name) for name in list_available()]


def load_datasets(
    names: list[DatasetName | str],
    split: str = "test",
    max_samples: int | None = None,
    seed: int | None = None,
    use_cache: bool = True,
) -> list[EvalDataset]:
    """Load multiple datasets, skipping any that fail to load."""
    results = []
    for name in names:
        try:
            results.append(get_dataset(
                name, split=split, max_samples=max_samples, seed=seed, use_cache=use_cache,
            ))
        except Exception as e:
            logger.warning(f"[REGISTRY] Skipping dataset '{name}': {e}")
            print(f"\nWARNING: Skipping dataset '{name}': {e}\n")
    return results
