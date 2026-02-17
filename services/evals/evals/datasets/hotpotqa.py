"""HotpotQA dataset loader.

HotpotQA is a multi-hop question answering dataset where questions
require reasoning across multiple supporting documents. Good for
testing retrieval precision and multi-hop reasoning.

Dataset: https://huggingface.co/datasets/hotpotqa/hotpot_qa
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


class HotpotQALoader(BaseDatasetLoader):
    """Loader for the HotpotQA dataset."""

    @property
    def name(self) -> str:
        return "hotpotqa"

    @property
    def description(self) -> str:
        return "Multi-hop question answering requiring reasoning across multiple documents"

    @property
    def source_url(self) -> str:
        return "https://huggingface.co/datasets/hotpotqa/hotpot_qa"

    @property
    def primary_aspects(self) -> list[str]:
        return ["retrieval", "generation"]

    @property
    def domains(self) -> list[str]:
        return ["general", "wikipedia"]

    def load(
        self,
        split: str = "test",
        max_samples: int | None = None,
        seed: int | None = None,
        difficulty_filter: str | None = None,
    ) -> EvalDataset:
        """Load HotpotQA dataset.

        Args:
            split: Which split to load (train, validation)
            max_samples: Maximum samples to load
            seed: Random seed for sampling
            difficulty_filter: Filter by difficulty ("hard", "medium", "easy")

        Returns:
            EvalDataset with loaded questions
        """
        logger.info(f"Loading HotpotQA dataset (split={split}, max_samples={max_samples})")

        if seed is not None:
            random.seed(seed)

        # HotpotQA has 'distractor' and 'fullwiki' configs
        # We use 'distractor' which includes supporting facts
        hf_split = "validation" if split in ("test", "validation", "val") else "train"

        dataset = load_dataset(
            "hotpotqa/hotpot_qa",
            "distractor",
            split=hf_split,
        )

        # Convert all items
        questions: list[EvalQuestion] = []
        for idx, item in enumerate(dataset):
            question = self._convert_item(item, idx)
            if question:
                # Apply difficulty filter
                if difficulty_filter:
                    if question.difficulty.value != difficulty_filter:
                        continue
                questions.append(question)

            # Early exit
            if max_samples and len(questions) >= max_samples:
                break

        # Final sampling if needed
        if max_samples and len(questions) > max_samples:
            questions = random.sample(questions, max_samples)

        logger.info(f"Loaded {len(questions)} questions from HotpotQA")

        return EvalDataset(
            name=self.name,
            version="1.0",
            questions=questions,
            description=self.description,
            source_url=self.source_url,
            domains=self.domains,
            metadata={
                "config": "distractor",
                "has_supporting_facts": True,
            },
        )

    def _convert_item(
        self,
        item: dict[str, Any],
        index: int,
    ) -> EvalQuestion | None:
        """Convert a HotpotQA item to EvalQuestion."""
        try:
            question_text = item.get("question", "")
            answer = item.get("answer", "")
            level = item.get("level", "medium")  # 'hard', 'medium', 'easy'

            if not question_text:
                return None

            # Extract gold passages from supporting facts
            gold_passages = self._extract_supporting_passages(item)

            # Map HotpotQA level to our Difficulty enum
            difficulty_map = {
                "hard": Difficulty.HARD,
                "medium": Difficulty.MEDIUM,
                "easy": Difficulty.EASY,
            }
            difficulty = difficulty_map.get(level, Difficulty.MEDIUM)

            # HotpotQA questions are multi-hop by design
            query_type = QueryType.MULTI_HOP

            return EvalQuestion(
                id=self._create_question_id("hotpotqa", index, item.get("id")),
                question=question_text,
                expected_answer=answer,
                gold_passages=gold_passages,
                query_type=query_type,
                difficulty=difficulty,
                domain="wikipedia",
                is_unanswerable=False,
                metadata={
                    "level": level,
                    "type": item.get("type", ""),  # 'bridge' or 'comparison'
                    "supporting_facts_count": len(gold_passages),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to convert HotpotQA item: {e}")
            return None

    def _extract_supporting_passages(
        self,
        item: dict[str, Any],
    ) -> list[GoldPassage]:
        """Extract gold passages from supporting facts."""
        gold_passages = []

        # Get context paragraphs
        context = item.get("context", {})
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])

        # Get supporting facts (title, sentence_idx)
        supporting_facts = item.get("supporting_facts", {})
        sf_titles = supporting_facts.get("title", [])
        sf_sent_ids = supporting_facts.get("sent_id", [])

        # Build title -> sentences map
        title_to_sents = {}
        for title, sents in zip(titles, sentences_list):
            title_to_sents[title] = sents

        # Extract supporting sentences
        seen_passages = set()
        for sf_title, sf_sent_id in zip(sf_titles, sf_sent_ids):
            if sf_title in title_to_sents:
                sents = title_to_sents[sf_title]
                if 0 <= sf_sent_id < len(sents):
                    sent_text = sents[sf_sent_id]
                    passage_key = (sf_title, sf_sent_id)

                    if passage_key not in seen_passages:
                        seen_passages.add(passage_key)
                        gold_passages.append(
                            GoldPassage(
                                doc_id=sf_title,
                                chunk_id=f"{sf_title}:sent:{sf_sent_id}",
                                text=sent_text,
                            )
                        )

        return gold_passages
