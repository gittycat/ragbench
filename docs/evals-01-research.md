# RAG Evaluation: Research & Industry State (2025–2026)

## The 3-Tier Model

Industry consensus across major benchmarks and research groups has converged on evaluating RAG systems at three distinct tiers. A system can pass one tier and completely fail another — they are not redundant.

### Tier 1 — Generation Quality

**What it tests:** Given pre-retrieved passages, does the LLM produce a good answer?

Setup: provide `(question, gold passages, expected answer)` directly to the LLM. The retrieval pipeline is bypassed. Gold passages are injected as context.

Measures:
- **Faithfulness** — is the answer grounded in the provided context?
- **Answer correctness** — does it match the expected answer?
- **Answer relevancy** — does it address the question?
- **Abstention** — does it correctly refuse unanswerable questions?

This tier isolates the LLM and prompt from retrieval quality. Useful for: comparing LLM providers, tuning prompts, detecting hallucination patterns.

### Tier 2 — End-to-End Pipeline

**What it tests:** Does the full pipeline — chunking, embedding, indexing, retrieval, reranking, generation — produce good outputs?

Setup: ingest a known document corpus via the upload pipeline, wait for processing to complete, then send queries where the correct source documents are known.

Measures everything in Tier 1, plus:
- **Recall@K, Precision@K, MRR, NDCG** — retrieval quality
- **Real pipeline latency** — including retrieval and reranking

Key principle: *"If you can't retrieve it, you can't generate it."* Tier 2 will fail faster and more informatively than Tier 1 when the retrieval pipeline breaks. (Superlinked VectorHub, NVIDIA NeMo Evaluator)

### Tier 3 — Domain-Specific Synthetic Evaluation

**What it tests:** Does the system work on your actual documents at production quality?

Setup: use an LLM to generate QA pairs from a sample of real uploaded documents, filter for difficulty, optionally validate with human review, then run end-to-end (same as Tier 2) against these domain-specific questions.

Measures: everything in Tier 2, but on real-world documents and query distributions that match actual usage. Most actionable signal for production deployments.

Tools: RAGAS `TestsetGenerator`, DeepEval `Synthesizer`, NVIDIA NeMo Curator SDG pipeline.

Practitioner consensus:
- Minimum 100 questions for MVP / CI gate
- Minimum 200 questions for pre-production sign-off
- ~40% of the dataset should be adversarial or difficult questions
- Freeze dataset versions for comparable results across runs

---

## Key Research Findings

### RAGBench is a Generation Benchmark, Not an End-to-End Benchmark

RAGBench ([rungalileo/ragbench](https://huggingface.co/datasets/rungalileo/ragbench), paper: [arxiv.org/abs/2407.11005](https://arxiv.org/abs/2407.11005)) provides `(question, documents, answer)` triples where `documents` contains the gold passages. These passages should be **injected directly as context** (Tier 1) — not ingested into the RAG system and retrieved (Tier 2). Using RAGBench for end-to-end retrieval evaluation is a category error.

For Tier 2 evaluation, RAGBench passages *can* be uploaded as documents and then queried, using the gold doc_ids to assess retrieval quality — but this requires explicit ingestion infrastructure.

### LLM-as-Judge is Dominant but Requires Care

LLM judges are the standard for generation quality metrics. They are sensitive to:
- Prompt wording (minor changes produce large score shifts)
- Temperature (always use 0.0 for determinism)
- Judge model version (pin the version)
- Judge consistency (the ConsJudge approach, Feb 2026, specifically addresses this for RAG)

Multi-provider support is important: on-prem deployments need to judge with local models (Ollama), while cloud deployments can use frontier models.

### Multi-Turn Evaluation is Emerging

IBM's mtRAG benchmark and SemEval 2026 Task 8 (MTRAGEval) specifically target conversational RAG. Since this project supports chat history via `condense_plus_context` mode, multi-turn evaluation is directly relevant. However, multi-turn eval requires maintaining session state across queries — this is a Tier 3 extension, not a Day 1 requirement.

### DeepEval vs RAGAS

| Aspect | DeepEval | RAGAS |
|--------|----------|-------|
| CI/CD integration | pytest-native ✓ | Requires custom harness |
| Synthetic generation | Flexible (multi-provider) | Tied to LangChain ecosystem |
| Metric extensibility | Good | Good |
| Production maturity | Higher | Moderate |

This project already uses DeepEval (custom metrics, not DeepEval's built-in LLM test cases). The existing custom metric implementation is the right choice — it avoids framework coupling and allows direct control over metric computation.

### Established Datasets by Tier

| Dataset | HuggingFace ID | Tier | Primary Signal |
|---------|----------------|------|----------------|
| RAGBench | `rungalileo/ragbench` | Tier 1 (inject), Tier 2 (ingest) | Multi-domain generation + retrieval |
| SQuAD v2 | `rajpurkar/squad_v2` | Tier 1 (abstention) | Unanswerable question handling |
| HotpotQA | `hotpotqa/hotpot_qa` | Tier 2 | Multi-hop retrieval |
| Qasper | `allenai/qasper` | Tier 2 | Long-doc citation accuracy |
| MS MARCO | `microsoft/ms_marco` | Tier 2 | Retrieval ranking |
| golden (local) | n/a | Tier 3 | Domain-specific QA |

---

## Sources

- [RAGBench: Explainable Benchmark for RAG Systems](https://arxiv.org/abs/2407.11005) — Friel et al., 2024
- [Evaluation of Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2405.07437) — 2024
- [BEIR: Heterogeneous Benchmark for Zero-shot Evaluation](https://github.com/beir-cellar/beir)
- [RAGAS Documentation](https://docs.ragas.io/)
- [DeepEval](https://github.com/confident-ai/deepeval)
- [SemEval 2026 Task 8: MTRAGEval](https://semeval.github.io/SemEval2026/tasks)
- [NVIDIA NeMo Evaluator](https://developer.nvidia.com/nemo)
- [Superlinked VectorHub: RAG Evaluation Guide](https://superlinked.com/vectorhub)
- [Vectara Open RAGBench](https://huggingface.co/datasets/vectara/open_ragbench)
