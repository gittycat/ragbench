"""RAGBench dataset loader.

RAGBench is a multi-domain RAG benchmark with industry subsets including
legal, finance, tech, and medical domains. Purpose-built for RAG evaluation.

Each example carries TRACe annotations (adherence, context relevance,
utilization, completeness) produced by a GPT-4-family annotator. The loader
uses the sentence-level relevance keys to split each example's documents into
gold passages (relevant) and distractor context passages, and exposes the
TRACe scores in question metadata.

Dataset: https://huggingface.co/datasets/galileo-ai/ragbench
Paper: https://arxiv.org/abs/2407.11005
"""

import hashlib
import logging
import random
import re
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

HF_REPO_ID = "galileo-ai/ragbench"

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
    "pubmedqa": "medical",
    "tatqa": "finance",
    "techqa": "technical",
}

# Curated default mix: one subset per domain, keeps runs fast and diverse
DEFAULT_SUBSETS = ["covidqa", "finqa", "cuad", "techqa"]

# TRACe annotation fields copied into question metadata
TRACE_FIELDS = (
    "adherence_score",
    "relevance_score",
    "utilization_score",
    "completeness_score",
)

_SENTENCE_KEY_RE = re.compile(r"^(\d+)")


def _doc_index_from_key(key: str) -> int | None:
    """Sentence keys look like '0a', '12c' — leading digits are the doc index."""
    m = _SENTENCE_KEY_RE.match(key)
    return int(m.group(1)) if m else None


def relevant_doc_indices(item: dict[str, Any]) -> set[int]:
    """Doc indices containing at least one annotated-relevant sentence."""
    indices: set[int] = set()
    for key in item.get("all_relevant_sentence_keys") or []:
        idx = _doc_index_from_key(key)
        if idx is not None:
            indices.add(idx)
    return indices


def _content_doc_id(subset_name: str, text: str) -> str:
    """Stable content-based doc id so identical docs dedupe across questions."""
    digest = hashlib.sha1(text.encode()).hexdigest()[:12]
    return f"ragbench:{subset_name}:doc:{digest}"


class RAGBenchLoader(BaseDatasetLoader):
    """Loader for the RAGBench dataset."""

    @property
    def name(self) -> str:
        return "ragbench"

    @property
    def description(self) -> str:
        return "Multi-domain RAG benchmark with TRACe annotations (legal, finance, tech, medical)"

    @property
    def source_url(self) -> str:
        return f"https://huggingface.co/datasets/{HF_REPO_ID}"

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
            split: Which split to load (train, validation, test)
            max_samples: Maximum total samples to load
            seed: Random seed for sampling
            subsets: Specific subsets to load (None = curated default mix)
        """
        logger.info(f"Loading RAGBench dataset (split={split}, max_samples={max_samples})")

        if seed is not None:
            random.seed(seed)

        target_subsets = subsets or DEFAULT_SUBSETS
        questions: list[EvalQuestion] = []

        # Balanced sampling across subsets
        samples_per_subset = None
        if max_samples:
            samples_per_subset = max(1, max_samples // len(target_subsets))

        for subset_name in target_subsets:
            try:
                subset_questions = self._load_subset(
                    subset_name,
                    split=split,
                    max_samples=samples_per_subset,
                )
                questions.extend(subset_questions)
                logger.info(f"Loaded {len(subset_questions)} questions from {subset_name}")
            except Exception as e:
                logger.warning(f"Failed to load subset {subset_name}: {e}")
                continue

        if max_samples and len(questions) > max_samples:
            questions = random.sample(questions, max_samples)

        logger.info(f"Loaded {len(questions)} total questions from RAGBench")

        return EvalDataset(
            name=self.name,
            version="2.0",
            questions=questions,
            description=self.description,
            source_url=self.source_url,
            domains=self.domains,
            metadata={"subsets_loaded": target_subsets},
        )

    def load_raw_items(
        self,
        subsets: list[str] | None = None,
        split: str = "test",
        max_samples: int | None = None,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Load raw RAGBench items (with full TRACe annotations) for judge calibration."""
        if seed is not None:
            random.seed(seed)

        target_subsets = subsets or DEFAULT_SUBSETS
        per_subset = None
        if max_samples:
            per_subset = max(1, max_samples // len(target_subsets))

        items: list[dict[str, Any]] = []
        for subset_name in target_subsets:
            try:
                dataset = load_dataset(HF_REPO_ID, subset_name, split=split)
            except Exception as e:
                logger.warning(f"Failed to load subset {subset_name}: {e}")
                continue
            indices = range(len(dataset))
            if per_subset and len(dataset) > per_subset:
                indices = random.sample(range(len(dataset)), per_subset)
            for i in indices:
                item = dict(dataset[i])
                item["subset"] = subset_name
                items.append(item)

        if max_samples and len(items) > max_samples:
            items = random.sample(items, max_samples)
        return items

    def _load_subset(
        self,
        subset_name: str,
        split: str,
        max_samples: int | None,
    ) -> list[EvalQuestion]:
        """Load a single RAGBench subset."""
        dataset = load_dataset(HF_REPO_ID, subset_name, split=split)

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
        """Convert a RAGBench item to EvalQuestion.

        Documents with annotated-relevant sentences become gold passages;
        the rest become distractor context passages. If the example has no
        relevance annotations, all documents are treated as gold.
        """
        try:
            question_text = item.get("question", "")
            answer = item.get("response", "")

            if not question_text:
                return None

            relevant = relevant_doc_indices(item)

            gold_passages: list[GoldPassage] = []
            context_passages: list[GoldPassage] = []
            documents = item.get("documents") or []

            for doc_idx, doc in enumerate(documents):
                doc_text = doc if isinstance(doc, str) else doc.get("text", str(doc))
                if not doc_text.strip():
                    continue
                doc_id = _content_doc_id(subset_name, doc_text)
                passage = GoldPassage(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}:chunk:0",
                    text=doc_text[:4000],  # Truncate very long passages
                    relevance_score=1.0 if (not relevant or doc_idx in relevant) else 0.0,
                )
                if not relevant or doc_idx in relevant:
                    gold_passages.append(passage)
                else:
                    context_passages.append(passage)

            query_type = self._infer_query_type(question_text, {"domain": domain})

            metadata: dict[str, Any] = {
                "subset": subset_name,
                "original_id": item.get("id"),
                "original_index": index,
                "has_relevance_annotations": bool(relevant),
            }
            for trace_field in TRACE_FIELDS:
                if item.get(trace_field) is not None:
                    metadata[trace_field] = item[trace_field]

            return EvalQuestion(
                id=self._create_question_id(
                    f"ragbench:{subset_name}", index, orig_id=item.get("id")
                ),
                question=question_text,
                expected_answer=answer,
                gold_passages=gold_passages,
                context_passages=context_passages,
                query_type=query_type,
                difficulty=Difficulty.MEDIUM,
                domain=domain,
                is_unanswerable=False,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Failed to convert RAGBench item: {e}")
            return None
