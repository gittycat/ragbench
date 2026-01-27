"""MS MARCO dataset loader.

MS MARCO (Microsoft Machine Reading Comprehension) is a large-scale
reading comprehension and question answering dataset. Excellent for
testing retrieval ranking quality.

Dataset: https://huggingface.co/datasets/microsoft/ms_marco
"""

import logging
import random
from typing import Any

from datasets import load_dataset

from evals.datasets.base import BaseDatasetLoader
from evals.schemas import (
    EvalDataset,
    EvalQuestion,
    GoldPassage,
    QueryType,
    Difficulty,
)

logger = logging.getLogger(__name__)


class MSMarcoLoader(BaseDatasetLoader):
    """Loader for the MS MARCO dataset."""

    @property
    def name(self) -> str:
        return "msmarco"

    @property
    def description(self) -> str:
        return "Large-scale reading comprehension dataset for retrieval ranking evaluation"

    @property
    def source_url(self) -> str:
        return "https://huggingface.co/datasets/microsoft/ms_marco"

    @property
    def primary_aspects(self) -> list[str]:
        return ["retrieval"]

    @property
    def domains(self) -> list[str]:
        return ["general", "web"]

    def load(
        self,
        split: str = "test",
        max_samples: int | None = None,
        seed: int | None = None,
        version: str = "v2.1",
    ) -> EvalDataset:
        """Load MS MARCO dataset.

        Args:
            split: Which split to load (train, validation, test)
            max_samples: Maximum samples to load
            seed: Random seed for sampling
            version: Dataset version ("v1.1" or "v2.1")

        Returns:
            EvalDataset with loaded questions
        """
        logger.info(f"Loading MS MARCO dataset (split={split}, max_samples={max_samples})")

        if seed is not None:
            random.seed(seed)

        # MS MARCO has different configs - we use v2.1 for QA
        hf_split = "validation" if split in ("test", "validation", "val") else "train"

        try:
            dataset = load_dataset(
                "microsoft/ms_marco",
                "v2.1",
                split=hf_split,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Failed to load MS MARCO v2.1, trying v1.1: {e}")
            dataset = load_dataset(
                "microsoft/ms_marco",
                "v1.1",
                split=hf_split,
                trust_remote_code=True,
            )

        # Convert items
        questions: list[EvalQuestion] = []
        for idx, item in enumerate(dataset):
            question = self._convert_item(item, idx)
            if question:
                questions.append(question)

            # Early exit
            if max_samples and len(questions) >= max_samples:
                break

        # Final sampling
        if max_samples and len(questions) > max_samples:
            questions = random.sample(questions, max_samples)

        logger.info(f"Loaded {len(questions)} questions from MS MARCO")

        return EvalDataset(
            name=self.name,
            version=version,
            questions=questions,
            description=self.description,
            source_url=self.source_url,
            domains=self.domains,
            metadata={"version": version},
        )

    def _convert_item(
        self,
        item: dict[str, Any],
        index: int,
    ) -> EvalQuestion | None:
        """Convert an MS MARCO item to EvalQuestion."""
        try:
            query = item.get("query", "")
            query_type_raw = item.get("query_type", "")

            if not query:
                return None

            # Get passages
            passages = item.get("passages", {})
            passage_texts = passages.get("passage_text", [])
            is_selected = passages.get("is_selected", [])

            if not passage_texts:
                return None

            # Get answers (may be empty for unanswerable)
            answers = item.get("answers", [])
            has_answer = bool(answers and answers[0])

            # Build gold passages from selected passages
            gold_passages = []
            for idx, (text, selected) in enumerate(zip(passage_texts, is_selected)):
                if selected == 1:  # is_selected is 0 or 1
                    gold_passages.append(
                        GoldPassage(
                            doc_id=f"msmarco:{item.get('query_id', index)}:passage:{idx}",
                            chunk_id=f"msmarco:{item.get('query_id', index)}:passage:{idx}",
                            text=text,
                            relevance_score=1.0,
                        )
                    )

            # If no selected passages, use all as potential (lower relevance)
            if not gold_passages:
                for idx, text in enumerate(passage_texts[:5]):  # Limit to 5
                    gold_passages.append(
                        GoldPassage(
                            doc_id=f"msmarco:{item.get('query_id', index)}:passage:{idx}",
                            chunk_id=f"msmarco:{item.get('query_id', index)}:passage:{idx}",
                            text=text,
                            relevance_score=0.5,  # Lower relevance for unselected
                        )
                    )

            # Determine query type
            query_type = self._map_query_type(query_type_raw, query)

            # Expected answer
            expected_answer = answers[0] if has_answer else None
            is_unanswerable = not has_answer or expected_answer == "No Answer Present."

            if is_unanswerable:
                expected_answer = None
                query_type = QueryType.UNANSWERABLE

            return EvalQuestion(
                id=self._create_question_id("msmarco", index, str(item.get("query_id"))),
                question=query,
                expected_answer=expected_answer,
                gold_passages=gold_passages,
                query_type=query_type,
                difficulty=Difficulty.MEDIUM,
                domain="web",
                is_unanswerable=is_unanswerable,
                metadata={
                    "query_id": item.get("query_id"),
                    "query_type_raw": query_type_raw,
                    "passage_count": len(passage_texts),
                    "selected_passage_count": sum(is_selected) if is_selected else 0,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to convert MS MARCO item: {e}")
            return None

    def _map_query_type(self, raw_type: str, query: str) -> QueryType:
        """Map MS MARCO query type to our QueryType enum."""
        raw_lower = raw_type.lower()

        # MS MARCO query types: DESCRIPTION, NUMERIC, ENTITY, LOCATION, PERSON
        if raw_lower in ("description",):
            return QueryType.SUMMARY
        elif raw_lower in ("numeric", "entity", "location", "person"):
            return QueryType.FACTOID

        # Fall back to inference
        return self._infer_query_type(query, {})
