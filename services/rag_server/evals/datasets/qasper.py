"""Qasper dataset loader.

Qasper is a dataset for question answering over scientific papers,
with evidence annotations for each answer. Good for testing
section-level citation correctness.

Dataset: https://huggingface.co/datasets/qasper
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


class QasperLoader(BaseDatasetLoader):
    """Loader for the Qasper dataset."""

    @property
    def name(self) -> str:
        return "qasper"

    @property
    def description(self) -> str:
        return "Question answering over scientific papers with evidence annotations"

    @property
    def source_url(self) -> str:
        return "https://huggingface.co/datasets/allenai/qasper"

    @property
    def primary_aspects(self) -> list[str]:
        return ["citation", "generation"]

    @property
    def domains(self) -> list[str]:
        return ["scientific"]

    def load(
        self,
        split: str = "test",
        max_samples: int | None = None,
        seed: int | None = None,
    ) -> EvalDataset:
        """Load Qasper dataset.

        Args:
            split: Which split to load (train, validation, test)
            max_samples: Maximum samples to load
            seed: Random seed for sampling

        Returns:
            EvalDataset with loaded questions
        """
        logger.info(f"Loading Qasper dataset (split={split}, max_samples={max_samples})")

        if seed is not None:
            random.seed(seed)

        # Load from HuggingFace
        # Qasper uses 'validation' and 'test' splits
        if split == "test":
            hf_split = "test"
        elif split == "validation" or split == "val":
            hf_split = "validation"
        else:
            hf_split = "train"

        dataset = load_dataset("allenai/qasper", split=hf_split)

        questions: list[EvalQuestion] = []
        question_idx = 0

        for paper_idx, paper in enumerate(dataset):
            paper_questions = self._extract_questions_from_paper(paper, paper_idx)
            questions.extend(paper_questions)
            question_idx += len(paper_questions)

            # Early exit if we have enough
            if max_samples and len(questions) >= max_samples:
                break

        # Sample if we have more than needed
        if max_samples and len(questions) > max_samples:
            questions = random.sample(questions, max_samples)

        logger.info(f"Loaded {len(questions)} questions from Qasper")

        return EvalDataset(
            name=self.name,
            version="1.0",
            questions=questions,
            description=self.description,
            source_url=self.source_url,
            domains=self.domains,
        )

    def _extract_questions_from_paper(
        self,
        paper: dict[str, Any],
        paper_idx: int,
    ) -> list[EvalQuestion]:
        """Extract all questions from a single paper."""
        questions = []

        paper_id = paper.get("id", f"paper:{paper_idx}")
        full_text = paper.get("full_text", {})

        # Build section map for evidence lookup
        section_map = self._build_section_map(full_text, paper_id)

        # Process each QA pair
        qas = paper.get("qas", [])
        for qa_idx, qa in enumerate(qas):
            question = self._convert_qa(qa, paper_id, section_map, qa_idx)
            if question:
                questions.append(question)

        return questions

    def _build_section_map(
        self,
        full_text: dict[str, Any],
        paper_id: str,
    ) -> dict[str, GoldPassage]:
        """Build a map of section names to passages."""
        section_map = {}

        section_names = full_text.get("section_name", [])
        paragraphs = full_text.get("paragraphs", [])

        for idx, (name, para_list) in enumerate(zip(section_names, paragraphs)):
            if not para_list:
                continue

            text = "\n".join(para_list) if isinstance(para_list, list) else str(para_list)
            section_id = f"{paper_id}:section:{idx}"

            section_map[name] = GoldPassage(
                doc_id=paper_id,
                chunk_id=section_id,
                text=text[:2000],
            )

        return section_map

    def _convert_qa(
        self,
        qa: dict[str, Any],
        paper_id: str,
        section_map: dict[str, GoldPassage],
        qa_idx: int,
    ) -> EvalQuestion | None:
        """Convert a Qasper QA pair to EvalQuestion."""
        try:
            question_text = qa.get("question", "")
            if not question_text:
                return None

            # Get answers and evidence
            answers = qa.get("answers", [])
            if not answers:
                return None

            # Use first answer (Qasper may have multiple annotators)
            first_answer = answers[0]
            answer_info = first_answer.get("answer", {})

            # Check answer type
            is_unanswerable = answer_info.get("unanswerable", False)
            is_yes_no = answer_info.get("yes_no") is not None
            extractive = answer_info.get("extractive_spans", [])
            free_form = answer_info.get("free_form_answer", "")

            # Determine expected answer
            if is_unanswerable:
                expected_answer = None
            elif is_yes_no:
                expected_answer = "Yes" if answer_info.get("yes_no") else "No"
            elif extractive:
                expected_answer = " ".join(extractive)
            else:
                expected_answer = free_form

            # Extract gold passages from evidence
            gold_passages = []
            evidence = answer_info.get("evidence", [])

            for ev_text in evidence:
                # Try to match evidence to sections
                matched = False
                for section_name, passage in section_map.items():
                    if ev_text in passage.text:
                        gold_passages.append(passage)
                        matched = True
                        break

                # If no match, create a standalone passage
                if not matched and ev_text:
                    gold_passages.append(
                        GoldPassage(
                            doc_id=paper_id,
                            chunk_id=f"{paper_id}:evidence:{len(gold_passages)}",
                            text=ev_text[:2000],
                        )
                    )

            # Infer query type
            if is_unanswerable:
                query_type = QueryType.UNANSWERABLE
            elif is_yes_no:
                query_type = QueryType.FACTOID
            else:
                query_type = self._infer_query_type(question_text, {})

            return EvalQuestion(
                id=self._create_question_id("qasper", qa_idx, qa.get("question_id")),
                question=question_text,
                expected_answer=expected_answer,
                gold_passages=gold_passages,
                query_type=query_type,
                difficulty=Difficulty.HARD,  # Scientific papers are generally hard
                domain="scientific",
                is_unanswerable=is_unanswerable,
                metadata={
                    "paper_id": paper_id,
                    "is_yes_no": is_yes_no,
                    "has_extractive": bool(extractive),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to convert Qasper QA: {e}")
            return None
