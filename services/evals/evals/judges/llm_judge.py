"""LLM-as-judge for evaluation metrics.

Uses a configurable LLM to evaluate answers for:
- Faithfulness (grounding in context)
- Answer correctness (vs expected answer)
- Answer relevancy (to the question)
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from llama_index.core.llms import LLM

from evals.config import JudgeConfig

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Result from an LLM judge evaluation.

    Attributes:
        metric_name: Name of the metric being evaluated
        score: Numeric score (typically 0-1)
        reasoning: LLM's reasoning for the score
        raw_response: Raw LLM response text
        metadata: Additional result metadata
    """

    metric_name: str
    score: float
    reasoning: str = ""
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMJudge:
    """LLM-as-judge for evaluating RAG responses.

    Uses a separate LLM instance (not the main RAG LLM) to evaluate
    responses for various quality metrics.
    """

    def __init__(self, config: JudgeConfig | None = None):
        """Initialize the judge.

        Args:
            config: Judge configuration. If None, loads from models config.
        """
        self.config = config or self._load_default_config()
        self._llm: LLM | None = None

    def _load_default_config(self) -> JudgeConfig:
        """Load judge config from models.yml."""
        from infrastructure.config.models_config import get_models_config

        models_config = get_models_config()
        eval_config = models_config.eval

        return JudgeConfig(
            enabled=True,
            provider=eval_config.provider,
            model=eval_config.model,
            temperature=0.0,
            max_retries=3,
        )

    @property
    def llm(self) -> LLM:
        """Get or create the judge LLM instance."""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _create_llm(self) -> LLM:
        """Create a new LLM client for the judge."""
        from infrastructure.llm.config import LLMConfig, LLMProvider
        from infrastructure.llm.factory import create_llm_client
        from infrastructure.config.models_config import get_models_config

        models_config = get_models_config()
        api_key = models_config.eval.api_key

        try:
            provider = LLMProvider(self.config.provider)
        except ValueError:
            raise ValueError(f"Unsupported judge provider: {self.config.provider}")

        llm_config = LLMConfig(
            provider=provider,
            model=self.config.model,
            api_key=api_key,
            timeout=120.0,
        )

        logger.info(f"[JUDGE] Creating {provider.value} LLM: {self.config.model}")
        return create_llm_client(llm_config)

    def evaluate_faithfulness(
        self,
        answer: str,
        context: str,
    ) -> JudgeResult:
        """Evaluate whether the answer is faithful to the context.

        Args:
            answer: The generated answer
            context: The retrieved context

        Returns:
            JudgeResult with faithfulness score (0-1)
        """
        prompt = f"""You are evaluating the faithfulness of an answer to the provided context.
Faithfulness measures whether all claims in the answer are supported by the context.

Context:
{context}

Answer:
{answer}

Evaluate the faithfulness of this answer on a scale of 0 to 1:
- 1.0: All claims are fully supported by the context
- 0.5: Some claims are supported, but others are not or are partially supported
- 0.0: The answer contains claims that contradict or are not supported by the context

Provide your response in the following format:
SCORE: [0.0-1.0]
REASONING: [Your explanation]"""

        return self._evaluate(prompt, "faithfulness")

    def evaluate_correctness(
        self,
        answer: str,
        expected_answer: str,
        question: str,
    ) -> JudgeResult:
        """Evaluate whether the answer is correct compared to expected.

        Args:
            answer: The generated answer
            expected_answer: The expected/reference answer
            question: The original question

        Returns:
            JudgeResult with correctness score (0-1)
        """
        prompt = f"""You are evaluating the correctness of an answer compared to a reference answer.
Correctness measures whether the answer conveys the same information as the reference.

Question:
{question}

Reference Answer:
{expected_answer}

Generated Answer:
{answer}

Evaluate the correctness on a scale of 0 to 1:
- 1.0: The answer is fully correct and equivalent to the reference
- 0.5: The answer is partially correct or missing some key information
- 0.0: The answer is incorrect or contradicts the reference

Provide your response in the following format:
SCORE: [0.0-1.0]
REASONING: [Your explanation]"""

        return self._evaluate(prompt, "correctness")

    def evaluate_relevancy(
        self,
        answer: str,
        question: str,
    ) -> JudgeResult:
        """Evaluate whether the answer is relevant to the question.

        Args:
            answer: The generated answer
            question: The original question

        Returns:
            JudgeResult with relevancy score (0-1)
        """
        prompt = f"""You are evaluating the relevancy of an answer to a question.
Relevancy measures whether the answer addresses what the question is asking.

Question:
{question}

Answer:
{answer}

Evaluate the relevancy on a scale of 0 to 1:
- 1.0: The answer directly and completely addresses the question
- 0.5: The answer partially addresses the question or includes irrelevant information
- 0.0: The answer does not address the question at all

Provide your response in the following format:
SCORE: [0.0-1.0]
REASONING: [Your explanation]"""

        return self._evaluate(prompt, "relevancy")

    def _evaluate(self, prompt: str, metric_name: str) -> JudgeResult:
        """Run evaluation with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                response = self.llm.complete(prompt)
                raw_response = str(response)

                # Parse response
                score, reasoning = self._parse_response(raw_response)

                return JudgeResult(
                    metric_name=metric_name,
                    score=score,
                    reasoning=reasoning,
                    raw_response=raw_response,
                    metadata={"attempt": attempt + 1},
                )

            except Exception as e:
                logger.warning(f"[JUDGE] Attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    logger.error(f"[JUDGE] All attempts failed for {metric_name}")
                    return JudgeResult(
                        metric_name=metric_name,
                        score=0.0,
                        reasoning=f"Evaluation failed: {e}",
                        raw_response="",
                        metadata={"error": str(e)},
                    )

        # Should not reach here
        return JudgeResult(metric_name=metric_name, score=0.0, reasoning="Unknown error")

    def _parse_response(self, response: str) -> tuple[float, str]:
        """Parse the score and reasoning from LLM response."""
        lines = response.strip().split("\n")
        score = 0.0
        reasoning = ""

        for line in lines:
            line = line.strip()
            if line.upper().startswith("SCORE:"):
                try:
                    score_str = line.split(":", 1)[1].strip()
                    # Handle various formats: "0.8", "0.8/1", "80%"
                    score_str = score_str.replace("/1", "").replace("%", "")
                    if "%" in line:
                        score = float(score_str) / 100
                    else:
                        score = float(score_str)
                    score = max(0.0, min(1.0, score))  # Clamp to 0-1
                except ValueError:
                    logger.warning(f"[JUDGE] Failed to parse score from: {line}")
            elif line.upper().startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        # If reasoning not found in REASONING: line, use remainder of response
        if not reasoning:
            score_line_found = False
            for line in lines:
                if line.upper().startswith("SCORE:"):
                    score_line_found = True
                elif score_line_found:
                    reasoning += line + " "
            reasoning = reasoning.strip()

        return score, reasoning
