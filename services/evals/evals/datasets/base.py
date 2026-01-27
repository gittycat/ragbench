"""Base class for dataset loaders."""

from abc import ABC, abstractmethod
from typing import Any

from evals.schemas import EvalDataset, EvalQuestion, QueryType


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders.

    Each dataset loader is responsible for:
    1. Loading data from HuggingFace or other sources
    2. Converting to the unified EvalDataset/EvalQuestion schema
    3. Providing metadata about the dataset

    Subclasses must implement:
    - load(): Load and return the full dataset
    - get_metadata(): Return dataset metadata
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this dataset."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the dataset."""
        ...

    @property
    @abstractmethod
    def source_url(self) -> str:
        """URL to the original dataset source."""
        ...

    @property
    def primary_aspects(self) -> list[str]:
        """Primary evaluation aspects this dataset is good for."""
        return ["generation"]

    @property
    def domains(self) -> list[str]:
        """Domains covered by this dataset."""
        return ["general"]

    @abstractmethod
    def load(
        self,
        split: str = "test",
        max_samples: int | None = None,
        seed: int | None = None,
    ) -> EvalDataset:
        """Load the dataset.

        Args:
            split: Which split to load (train, validation, test)
            max_samples: Maximum number of samples to load (None = all)
            seed: Random seed for sampling

        Returns:
            EvalDataset with loaded questions
        """
        ...

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata about this dataset."""
        return {
            "name": self.name,
            "description": self.description,
            "source_url": self.source_url,
            "primary_aspects": self.primary_aspects,
            "domains": self.domains,
        }

    def _infer_query_type(self, question: str, metadata: dict) -> QueryType:
        """Infer query type from question text and metadata.

        Can be overridden by subclasses for dataset-specific logic.
        """
        question_lower = question.lower()

        # Check for summary/report patterns
        summary_keywords = ["summarize", "summary", "report", "list all", "describe"]
        if any(kw in question_lower for kw in summary_keywords):
            return QueryType.SUMMARY

        # Check for comparison patterns
        comparison_keywords = ["compare", "difference", "versus", "vs", "contrast"]
        if any(kw in question_lower for kw in comparison_keywords):
            return QueryType.COMPARISON

        # Check for procedural patterns
        procedural_keywords = ["how to", "how do", "how can", "steps to", "process"]
        if any(kw in question_lower for kw in procedural_keywords):
            return QueryType.PROCEDURAL

        # Default to factoid
        return QueryType.FACTOID

    def _create_question_id(self, dataset_name: str, index: int, orig_id: str | None = None) -> str:
        """Create a unique question ID."""
        if orig_id:
            return f"{dataset_name}:{orig_id}"
        return f"{dataset_name}:{index}"
