"""Golden dataset loader for local curated Q&A pairs."""

import json
import random
from pathlib import Path
from typing import Any

from evaluation_cc.datasets.base import BaseDatasetLoader
from evaluation_cc.schemas import EvalDataset, EvalQuestion, QueryType, Difficulty


class GoldenDatasetLoader(BaseDatasetLoader):
    """Loader for the local golden Q&A dataset.

    Loads curated question-answer pairs from eval_data/golden_qa.json.
    This dataset is used for quick local testing without requiring
    external dataset downloads.
    """

    # Path for local development
    GOLDEN_PATH = Path("eval_data/golden_qa.json")
    # Path inside Docker container
    GOLDEN_PATH_DOCKER = Path("/app/eval_data/golden_qa.json")

    @property
    def name(self) -> str:
        return "golden"

    @property
    def description(self) -> str:
        return "Curated Q&A pairs from your indexed documents"

    @property
    def source_url(self) -> str:
        return "local"

    @property
    def primary_aspects(self) -> list[str]:
        return ["generation", "retrieval"]

    @property
    def domains(self) -> list[str]:
        return ["user documents"]

    def _get_path(self) -> Path:
        """Get the appropriate path based on environment."""
        if self.GOLDEN_PATH_DOCKER.exists():
            return self.GOLDEN_PATH_DOCKER
        if self.GOLDEN_PATH.exists():
            return self.GOLDEN_PATH
        raise FileNotFoundError(
            f"Golden dataset not found at {self.GOLDEN_PATH} or {self.GOLDEN_PATH_DOCKER}"
        )

    def _map_query_type(self, qt: str) -> QueryType:
        """Map golden dataset query types to QueryType enum."""
        mapping = {
            "factual": QueryType.FACTOID,
            "factoid": QueryType.FACTOID,
            "reasoning": QueryType.MULTI_HOP,
            "multi_hop": QueryType.MULTI_HOP,
            "summary": QueryType.SUMMARY,
            "procedural": QueryType.PROCEDURAL,
            "comparison": QueryType.COMPARISON,
            "unanswerable": QueryType.UNANSWERABLE,
        }
        return mapping.get(qt.lower(), QueryType.FACTOID)

    def load(
        self,
        split: str = "test",
        max_samples: int | None = None,
        seed: int | None = None,
    ) -> EvalDataset:
        """Load the golden dataset.

        Args:
            split: Ignored for golden dataset (only one split)
            max_samples: Maximum number of samples to load
            seed: Random seed for sampling

        Returns:
            EvalDataset with loaded questions
        """
        path = self._get_path()

        with open(path) as f:
            data = json.load(f)

        questions = []
        for idx, item in enumerate(data):
            question = EvalQuestion(
                id=self._create_question_id("golden", idx),
                question=item["question"],
                expected_answer=item.get("answer"),
                gold_passages=[],  # Golden dataset doesn't include gold passages
                query_type=self._map_query_type(item.get("query_type", "factual")),
                difficulty=Difficulty.MEDIUM,
                domain=item.get("document", "unknown"),
                is_unanswerable=False,
                metadata={
                    "document": item.get("document"),
                    "context_hint": item.get("context_hint"),
                },
            )
            questions.append(question)

        # Sample if max_samples specified
        if max_samples and len(questions) > max_samples:
            if seed is not None:
                random.seed(seed)
            questions = random.sample(questions, max_samples)

        return EvalDataset(
            name=self.name,
            version="1.0",
            questions=questions,
            description=self.description,
            source_url=self.source_url,
            domains=self.domains,
            metadata={"path": str(path)},
        )

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata about this dataset."""
        size = 0
        try:
            path = self._get_path()
            with open(path) as f:
                size = len(json.load(f))
        except FileNotFoundError:
            pass

        return {
            "id": self.name,
            "name": "Golden Dataset (Local)",
            "description": self.description,
            "size": size,
            "domains": self.domains,
            "primary_aspects": self.primary_aspects,
            "requires_download": False,
            "download_size_mb": 0,
        }
