"""SQuAD v2 dataset loader.

SQuAD v2 extends SQuAD 1.1 with unanswerable questions, making it
ideal for testing abstention handling (knowing when NOT to answer).

Dataset: https://huggingface.co/datasets/rajpurkar/squad_v2
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


class SquadV2Loader(BaseDatasetLoader):
    """Loader for the SQuAD v2 dataset."""

    @property
    def name(self) -> str:
        return "squad_v2"

    @property
    def description(self) -> str:
        return "Reading comprehension with unanswerable questions for abstention testing"

    @property
    def source_url(self) -> str:
        return "https://huggingface.co/datasets/rajpurkar/squad_v2"

    @property
    def primary_aspects(self) -> list[str]:
        return ["abstention"]

    @property
    def domains(self) -> list[str]:
        return ["general", "wikipedia"]

    def load(
        self,
        split: str = "test",
        max_samples: int | None = None,
        seed: int | None = None,
        unanswerable_ratio: float | None = None,
    ) -> EvalDataset:
        """Load SQuAD v2 dataset.

        Args:
            split: Which split to load (train, validation)
            max_samples: Maximum samples to load
            seed: Random seed for sampling
            unanswerable_ratio: Target ratio of unanswerable questions (None = natural)

        Returns:
            EvalDataset with loaded questions
        """
        logger.info(f"Loading SQuAD v2 dataset (split={split}, max_samples={max_samples})")

        if seed is not None:
            random.seed(seed)

        # SQuAD v2 only has train and validation splits (no test with labels)
        hf_split = "validation" if split in ("test", "validation", "val") else "train"

        dataset = load_dataset("rajpurkar/squad_v2", split=hf_split, trust_remote_code=True)

        # Convert all items
        all_questions: list[EvalQuestion] = []
        for idx, item in enumerate(dataset):
            question = self._convert_item(item, idx)
            if question:
                all_questions.append(question)

        # Balance unanswerable ratio if requested
        if unanswerable_ratio is not None:
            all_questions = self._balance_unanswerable(
                all_questions, unanswerable_ratio, max_samples, seed
            )
        elif max_samples and len(all_questions) > max_samples:
            all_questions = random.sample(all_questions, max_samples)

        logger.info(
            f"Loaded {len(all_questions)} questions from SQuAD v2 "
            f"({sum(1 for q in all_questions if q.is_unanswerable)} unanswerable)"
        )

        return EvalDataset(
            name=self.name,
            version="2.0",
            questions=all_questions,
            description=self.description,
            source_url=self.source_url,
            domains=self.domains,
            metadata={
                "unanswerable_count": sum(1 for q in all_questions if q.is_unanswerable),
                "answerable_count": sum(1 for q in all_questions if not q.is_unanswerable),
            },
        )

    def _convert_item(
        self,
        item: dict[str, Any],
        index: int,
    ) -> EvalQuestion | None:
        """Convert a SQuAD v2 item to EvalQuestion."""
        try:
            question_text = item.get("question", "")
            context = item.get("context", "")
            answers = item.get("answers", {})

            if not question_text or not context:
                return None

            # Check if unanswerable
            answer_texts = answers.get("text", [])
            is_unanswerable = len(answer_texts) == 0

            # Get expected answer
            if is_unanswerable:
                expected_answer = None
                query_type = QueryType.UNANSWERABLE
            else:
                expected_answer = answer_texts[0]  # Use first answer
                query_type = self._infer_query_type(question_text, {})

            # Create gold passage from context
            doc_id = item.get("id", f"squad_v2:{index}")
            gold_passages = [
                GoldPassage(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}:context",
                    text=context,
                )
            ]

            # Determine difficulty based on answer position/type
            difficulty = self._assess_difficulty(item, is_unanswerable)

            return EvalQuestion(
                id=self._create_question_id("squad_v2", index, item.get("id")),
                question=question_text,
                expected_answer=expected_answer,
                gold_passages=gold_passages,
                query_type=query_type,
                difficulty=difficulty,
                domain="wikipedia",
                is_unanswerable=is_unanswerable,
                metadata={
                    "title": item.get("title", ""),
                    "answer_start": answers.get("answer_start", [None])[0] if not is_unanswerable else None,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to convert SQuAD v2 item: {e}")
            return None

    def _assess_difficulty(
        self,
        item: dict[str, Any],
        is_unanswerable: bool,
    ) -> Difficulty:
        """Assess question difficulty."""
        if is_unanswerable:
            # Unanswerable questions are harder (need to recognize lack of evidence)
            return Difficulty.HARD

        # Simple heuristic: longer contexts and questions are harder
        context_len = len(item.get("context", ""))
        question_len = len(item.get("question", ""))

        if context_len > 1500 or question_len > 100:
            return Difficulty.HARD
        elif context_len > 800 or question_len > 50:
            return Difficulty.MEDIUM
        else:
            return Difficulty.EASY

    def _balance_unanswerable(
        self,
        questions: list[EvalQuestion],
        target_ratio: float,
        max_samples: int | None,
        seed: int | None,
    ) -> list[EvalQuestion]:
        """Balance the ratio of unanswerable questions."""
        if seed is not None:
            random.seed(seed)

        answerable = [q for q in questions if not q.is_unanswerable]
        unanswerable = [q for q in questions if q.is_unanswerable]

        if not answerable or not unanswerable:
            return questions

        # Calculate target counts
        total = max_samples or len(questions)
        target_unanswerable = int(total * target_ratio)
        target_answerable = total - target_unanswerable

        # Sample to achieve target ratio
        sampled_unanswerable = random.sample(
            unanswerable, min(target_unanswerable, len(unanswerable))
        )
        sampled_answerable = random.sample(
            answerable, min(target_answerable, len(answerable))
        )

        result = sampled_answerable + sampled_unanswerable
        random.shuffle(result)

        return result
