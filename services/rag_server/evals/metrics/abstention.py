"""Abstention metrics for unanswerable question handling.

Measures how well the RAG system handles questions that cannot be answered
from the available context.
"""

from typing import Any

from evals.metrics.base import BaseMetric
from evals.schemas import (
    EvalQuestion,
    EvalResponse,
    MetricResult,
    MetricGroup,
)


# Default phrases indicating abstention
DEFAULT_ABSTENTION_PHRASES = [
    "i don't have enough information",
    "i do not have enough information",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "not enough information",
    "insufficient information",
    "no information available",
    "don't know",
    "do not know",
    "cannot be determined",
    "can't be determined",
    "not mentioned",
    "not found",
    "no relevant information",
]


def is_abstention(answer: str, phrases: list[str] | None = None) -> bool:
    """Check if an answer is an abstention (refusal to answer).

    Args:
        answer: The generated answer
        phrases: Custom abstention phrases (uses defaults if None)

    Returns:
        True if the answer appears to be an abstention
    """
    if not answer:
        return True

    answer_lower = answer.lower().strip()
    phrases = phrases or DEFAULT_ABSTENTION_PHRASES

    for phrase in phrases:
        if phrase.lower() in answer_lower:
            return True

    return False


class UnanswerableAccuracy(BaseMetric):
    """Accuracy on unanswerable questions.

    Measures whether the system correctly abstains on unanswerable questions
    and correctly answers answerable questions.

    Higher is better. 1.0 means perfect handling.
    """

    def __init__(self, abstention_phrases: list[str] | None = None):
        self.abstention_phrases = abstention_phrases or DEFAULT_ABSTENTION_PHRASES

    @property
    def name(self) -> str:
        return "unanswerable_accuracy"

    @property
    def group(self) -> MetricGroup:
        return MetricGroup.ABSTENTION

    @property
    def description(self) -> str:
        return "Accuracy on handling unanswerable vs answerable questions"

    def compute(
        self,
        question: EvalQuestion,
        response: EvalResponse,
        **kwargs: Any,
    ) -> MetricResult:
        # Determine if system abstained
        did_abstain = is_abstention(response.answer, self.abstention_phrases)

        # Determine if should have abstained
        should_abstain = question.is_unanswerable

        # Check if correct
        is_correct = did_abstain == should_abstain

        return MetricResult(
            name=self.name,
            value=1.0 if is_correct else 0.0,
            group=self.group,
            sample_size=1,
            details={
                "should_abstain": should_abstain,
                "did_abstain": did_abstain,
                "is_correct": is_correct,
            },
        )

    def compute_batch(
        self,
        questions: list[EvalQuestion],
        responses: list[EvalResponse],
        **kwargs: Any,
    ) -> MetricResult:
        """Compute accuracy with detailed breakdown."""
        if len(questions) != len(responses):
            raise ValueError("Questions and responses must have same length")

        # Track confusion matrix
        true_positive = 0  # Correctly abstained
        true_negative = 0  # Correctly answered
        false_positive = 0  # Incorrectly abstained (should have answered)
        false_negative = 0  # Incorrectly answered (should have abstained)

        for q, r in zip(questions, responses):
            did_abstain = is_abstention(r.answer, self.abstention_phrases)
            should_abstain = q.is_unanswerable

            if should_abstain and did_abstain:
                true_positive += 1
            elif not should_abstain and not did_abstain:
                true_negative += 1
            elif not should_abstain and did_abstain:
                false_positive += 1
            else:  # should_abstain and not did_abstain
                false_negative += 1

        total = len(questions)
        accuracy = (true_positive + true_negative) / total if total > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=accuracy,
            group=self.group,
            sample_size=total,
            details={
                "true_positive": true_positive,
                "true_negative": true_negative,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "unanswerable_count": true_positive + false_negative,
                "answerable_count": true_negative + false_positive,
            },
        )


class FalsePositiveRate(BaseMetric):
    """False positive rate for abstention.

    Measures how often the system incorrectly abstains when it should answer.

    Lower is better. 0.0 means never incorrectly abstains.
    """

    def __init__(self, abstention_phrases: list[str] | None = None):
        self.abstention_phrases = abstention_phrases or DEFAULT_ABSTENTION_PHRASES

    @property
    def name(self) -> str:
        return "abstention_false_positive_rate"

    @property
    def group(self) -> MetricGroup:
        return MetricGroup.ABSTENTION

    @property
    def description(self) -> str:
        return "Rate of incorrect abstention on answerable questions"

    def compute(
        self,
        question: EvalQuestion,
        response: EvalResponse,
        **kwargs: Any,
    ) -> MetricResult:
        # Only meaningful for answerable questions
        if question.is_unanswerable:
            return MetricResult(
                name=self.name,
                value=0.0,  # N/A for unanswerable
                group=self.group,
                sample_size=0,
                details={"note": "Not applicable for unanswerable questions"},
            )

        did_abstain = is_abstention(response.answer, self.abstention_phrases)

        return MetricResult(
            name=self.name,
            value=1.0 if did_abstain else 0.0,
            group=self.group,
            sample_size=1,
            details={"did_abstain": did_abstain},
        )


class FalseNegativeRate(BaseMetric):
    """False negative rate for abstention.

    Measures how often the system incorrectly answers when it should abstain.
    This is the hallucination risk for unanswerable questions.

    Lower is better. 0.0 means never hallucinated on unanswerable questions.
    """

    def __init__(self, abstention_phrases: list[str] | None = None):
        self.abstention_phrases = abstention_phrases or DEFAULT_ABSTENTION_PHRASES

    @property
    def name(self) -> str:
        return "abstention_false_negative_rate"

    @property
    def group(self) -> MetricGroup:
        return MetricGroup.ABSTENTION

    @property
    def description(self) -> str:
        return "Rate of incorrect answering on unanswerable questions (hallucination risk)"

    def compute(
        self,
        question: EvalQuestion,
        response: EvalResponse,
        **kwargs: Any,
    ) -> MetricResult:
        # Only meaningful for unanswerable questions
        if not question.is_unanswerable:
            return MetricResult(
                name=self.name,
                value=0.0,  # N/A for answerable
                group=self.group,
                sample_size=0,
                details={"note": "Not applicable for answerable questions"},
            )

        did_abstain = is_abstention(response.answer, self.abstention_phrases)

        return MetricResult(
            name=self.name,
            value=0.0 if did_abstain else 1.0,
            group=self.group,
            sample_size=1,
            details={
                "did_abstain": did_abstain,
                "hallucinated": not did_abstain,
            },
        )
