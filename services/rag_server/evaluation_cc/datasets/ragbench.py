"""RAGBench dataset loader.

RAGBench is a multi-domain RAG benchmark with industry subsets including
legal, finance, tech, and medical domains. Purpose-built for RAG evaluation.

Dataset: https://huggingface.co/datasets/rungalileo/ragbench
Paper: https://arxiv.org/abs/2407.11005
"""

import logging
import random
from typing import Any

from datasets import load_dataset

from evaluation_cc.datasets.base import BaseDatasetLoader
from evaluation_cc.schemas import (
    EvalDataset,
    EvalQuestion,
    GoldPassage,
    QueryType,
    Difficulty,
)

logger = logging.getLogger(__name__)

# RAGBench subsets and their domain mappings
RAGBENCH_SUBSETS = {
    "covidqa": "medical",
    "cuad": "legal",
    "delucionqa": "general",
    "emanual": "technical",
    "expertqa": "general",
    "finqa": "finance",
    "hagrid": "general",
    "hotpotqa": "general",
    "msmarco": "general",
    "narrativeqa": "general",
    "natural_questions": "general",
    "pubmedqa": "medical",
    "squad": "general",
    "tatqa": "finance",
    "techqa": "technical",
}


class RAGBenchLoader(BaseDatasetLoader):
    """Loader for the RAGBench dataset."""

    @property
    def name(self) -> str:
        return "ragbench"

    @property
    def description(self) -> str:
        return "Multi-domain RAG benchmark with legal, finance, tech, and medical subsets"

    @property
    def source_url(self) -> str:
        return "https://huggingface.co/datasets/rungalileo/ragbench"

    @property
    def primary_aspects(self) -> list[str]:
        return ["generation", "retrieval"]

    @property
    def domains(self) -> list[str]:
        return list(set(RAGBENCH_SUBSETS.values()))

    def load(
        self,
        split: str = "test",
        max_samples: int | None = None,
        seed: int | None = None,
        subsets: list[str] | None = None,
    ) -> EvalDataset:
        """Load RAGBench dataset.

        Args:
            split: Which split to load (train, test)
            max_samples: Maximum total samples to load
            seed: Random seed for sampling
            subsets: Specific subsets to load (None = all)

        Returns:
            EvalDataset with loaded questions
        """
        logger.info(f"Loading RAGBench dataset (split={split}, max_samples={max_samples})")

        if seed is not None:
            random.seed(seed)

        # Determine which subsets to load
        target_subsets = subsets or list(RAGBENCH_SUBSETS.keys())
        questions: list[EvalQuestion] = []

        # Calculate samples per subset for balanced loading
        samples_per_subset = None
        if max_samples:
            samples_per_subset = max(1, max_samples // len(target_subsets))

        for subset_name in target_subsets:
            try:
                subset_questions = self._load_subset(
                    subset_name,
                    split=split,
                    max_samples=samples_per_subset,
                    seed=seed,
                )
                questions.extend(subset_questions)
                logger.info(f"Loaded {len(subset_questions)} questions from {subset_name}")
            except Exception as e:
                logger.warning(f"Failed to load subset {subset_name}: {e}")
                continue

        # Final sampling if we have more than max_samples
        if max_samples and len(questions) > max_samples:
            questions = random.sample(questions, max_samples)

        logger.info(f"Loaded {len(questions)} total questions from RAGBench")

        return EvalDataset(
            name=self.name,
            version="1.0",
            questions=questions,
            description=self.description,
            source_url=self.source_url,
            domains=self.domains,
            metadata={"subsets_loaded": target_subsets},
        )

    def _load_subset(
        self,
        subset_name: str,
        split: str,
        max_samples: int | None,
        seed: int | None,
    ) -> list[EvalQuestion]:
        """Load a single RAGBench subset."""
        # Load from HuggingFace
        dataset = load_dataset(
            "rungalileo/ragbench",
            subset_name,
            split=split,
            trust_remote_code=True,
        )

        # Sample if needed
        if max_samples and len(dataset) > max_samples:
            indices = random.sample(range(len(dataset)), max_samples)
            dataset = dataset.select(indices)

        domain = RAGBENCH_SUBSETS.get(subset_name, "general")
        questions = []

        for idx, item in enumerate(dataset):
            question = self._convert_item(item, subset_name, domain, idx)
            if question:
                questions.append(question)

        return questions

    def _convert_item(
        self,
        item: dict[str, Any],
        subset_name: str,
        domain: str,
        index: int,
    ) -> EvalQuestion | None:
        """Convert a RAGBench item to EvalQuestion."""
        try:
            # RAGBench schema has: question, response, documents, all_documents
            question_text = item.get("question", "")
            answer = item.get("response", "")

            if not question_text:
                return None

            # Extract gold passages from documents
            gold_passages = []
            documents = item.get("documents", [])

            for doc_idx, doc in enumerate(documents):
                # Documents can be strings or dicts
                if isinstance(doc, str):
                    doc_text = doc
                    doc_id = f"{subset_name}:doc:{doc_idx}"
                else:
                    doc_text = doc.get("text", str(doc))
                    doc_id = doc.get("id", f"{subset_name}:doc:{doc_idx}")

                gold_passages.append(
                    GoldPassage(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}:chunk:0",
                        text=doc_text[:2000],  # Truncate very long passages
                    )
                )

            # Infer query type
            query_type = self._infer_query_type(question_text, {"domain": domain})

            return EvalQuestion(
                id=self._create_question_id(f"ragbench:{subset_name}", index),
                question=question_text,
                expected_answer=answer,
                gold_passages=gold_passages,
                query_type=query_type,
                difficulty=Difficulty.MEDIUM,
                domain=domain,
                is_unanswerable=False,
                metadata={
                    "subset": subset_name,
                    "original_index": index,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to convert RAGBench item: {e}")
            return None
