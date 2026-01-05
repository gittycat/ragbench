# RAG Eval Research Requirements, Assumptions, and Draft Plan

Generated with Codex / GPT 5.2

This document captures the requirements for the evaluation component of ragbench. It includes:
- assumptions and clarifications made
- a survey of evaluation frameworks and datasets
- and a draft plan for rebuilding evaluation from scratch to support objective-weighted model/search/prompting decisions.

## Requirements

### Product Goal
- The system must help users choose the best configuration (LLM model, retrieval, reranking, prompting) for their documents.
- "Best" depends on user-defined objectives and weights (accuracy, cost, speed, privacy/on-prem, hallucination risk, focus, etc.).
- The UI must show both a single weighted score and a detailed scorecard across evaluation dimensions.

### Evaluation Outputs (must be reported)
- Retrieval quality
- Answer quality and correctness
- Faithfulness / hallucination risk
- Citation accuracy (correct document + section/chunk)
- Latency (p50/p95)
- Cost (LLM + retrieval + reranking + embedding if applicable)
- Objective-weighted overall score + Pareto-optimal options

### Citation Requirements
- Citation correctness is required for all answers (source document and section).
- Evaluation must support both:
  - LLM-as-judge citation scoring
  - Manual inspection workflows
- Future UI setting: choose whether citations are derived from all retrieved chunks or only explicitly cited sources.

### Model Constraints
- Evaluation models are user-configured:
  - On-prem only (privacy-first) OR cloud-based frontier models
  - Must be selectable in settings

### Document Domains
- Exact document types are not known ahead of time and depend on the user and deployment.
- The evaluation system must be generic by default, with the ability to add deeper domain-specific evaluations later.
- No multilingual content required.

### Query Types
- Mostly factoid questions
- Some long-form summary/reporting tasks (e.g., "report all expenses to department X")

## Assumptions and Clarifications Made
- Evaluation must be robust enough to compare different combinations of LLM, retrieval, and prompting.
- The evaluation dataset cannot be limited to a small, single-domain golden set.
- The initial evaluation suite must be domain-agnostic, with extension points for domain-specific datasets later.
- A citation-focused evaluation layer is required in addition to standard RAG metrics.
- The evaluation system should support offline/manual review for high-stakes domains.
- Evaluation must handle unanswerable questions (abstention vs hallucination).
- User-defined objective weights should produce a single score, but raw metrics must remain visible for tradeoff analysis.

## Evaluation Framework Survey (extensive, best-fit oriented)

### 1) RAGAS
- URL: https://docs.ragas.io/
- Strengths:
  - Designed for RAG evaluation (faithfulness, context precision/recall, answer relevance)
  - Supports LLM-as-judge + reference-free metrics
  - Extensible, widely adopted
- Weaknesses:
  - Requires careful prompt/LLM selection to judge reliability
  - Still needs explicit citation metrics for doc/section accuracy
- Fit:
  - Good core framework for RAG scorecard
  - Needs additions for citation correctness and unanswerable handling

### 2) DeepEval (current)
- URL: https://github.com/confident-ai/deepeval (already in repo)
- Strengths:
  - Simple integration; existing code already wired
  - Multiple RAG metrics available
- Weaknesses:
  - Lacks built-in citation accuracy metrics
  - Requires gold passages for reliable retrieval metrics
- Fit:
  - Acceptable if extended with gold-aware retrieval and citation scoring

### 3) BEIR / Retrieval Benchmarks (for retrieval-only)
- URL: https://github.com/beir-cellar/beir
- Strengths:
  - Strong retrieval evaluation suite
  - Retrieval metrics are robust and widely comparable
- Weaknesses:
  - Not a full RAG eval framework (no generation/citation metrics)
- Fit:
  - Best for isolating search/reranking quality

### 4) Promptfoo / OpenAI Evals / LLM Evals Suites
- Strengths:
  - Good for A/B testing, prompt regression
- Weaknesses:
  - Not RAG-specific; lacks citation and retrieval alignment by default
- Fit:
  - Optional; not core to this project's goals

### 5) LlamaIndex / LangChain evaluation toolkits
- Strengths:
  - Built-in RAG evaluation modules
  - Often include judge-based scoring and dataset utilities
- Weaknesses:
  - Framework coupling; less controllable for custom RAG stacks
- Fit:
  - Only if integration convenience outweighs coupling costs

### Framework Recommendation
- Primary: DeepEval (selected). Extend with gold-aware retrieval metrics and citation correctness scoring.
- Retrieval benchmarking: use BEIR-style metrics (Recall@K, MRR, NDCG) with gold passages.
- Citation correctness: add a dedicated evaluation layer (custom metric) using chunk/section IDs.

## Dataset Survey (public, best-fit for technical/legal/tax)

### High Priority (generic-first baseline)

1) RAGBench (multi-domain, RAG-specific, industry subsets)
   - Dataset card: https://huggingface.co/datasets/rungalileo/ragbench/resolve/main/README.md
   - Paper: https://arxiv.org/abs/2407.11005
   - Why: purpose-built for RAG; includes legal, finance, tech, medical domains
   - Value: cross-domain comparisons and domain-specific slices
   - Recommended approach:
     - Start with a balanced mix across subsets for domain-agnostic coverage.
     - Add targeted subsets later as deployments reveal dominant domains.

2) Qasper (long documents, evidence-grounded QA)
   - Dataset card: https://huggingface.co/datasets/qasper/resolve/main/README.md
   - Why: evidence annotations in long docs; good proxy for section-level citation correctness
   - Value: long-form and evidence-heavy answers

3) SQuAD v2 (unanswerable handling)
   - Dataset card: https://huggingface.co/datasets/rajpurkar/squad_v2/resolve/main/README.md
   - Why: unanswerable questions
   - Value: abstention/false answer analysis

### Medium Priority (specific gaps)

4) HotpotQA (multi-hop reasoning)
   - Dataset card: https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/README.md
   - Why: supporting facts allow precise retrieval evaluation
   - Value: stress-test retrieval + multi-hop reasoning

5) MS MARCO (retrieval ranking)
   - Dataset card: https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/README.md
   - Why: retrieval benchmark standard
   - Value: ranking metrics for search and rerankers

### Notes on Gated Datasets
- Some datasets (e.g., MultiDoc2Dial, IBM TechQA) may require gated access. Consider them only if access is feasible and licensing permits.

## Recommended Metrics (aligned with goals)

### Retrieval Metrics
- Recall@K
- Precision@K
- MRR
- NDCG
- Coverage of gold passages (doc_id + chunk_id)

### Generation Metrics
- Faithfulness (answer grounded in retrieved context)
- Answer correctness (judge vs reference)
- Answer relevancy
- Long-form completeness (for summary/report tasks)

### Citation Metrics (core requirement)
- Citation precision (cited chunks are relevant)
- Citation recall (all gold evidence cited)
- Section-level accuracy (doc_id + section_id/chunk_id)

### Abstention Metrics
- Unanswerable accuracy (correct refusal)
- False positive answer rate
- False negative refusal rate

### Performance Metrics
- Latency p50/p95
- Cost per query (LLM + embedding + reranking)

## Draft Plan (rebuild eval from scratch)

### Phase 0: Core Requirements + Data Contracts
1) Define a unified evaluation schema:
   - Questions, expected answers, gold passages (doc_id + chunk_id), metadata (domain, difficulty)
2) Define a response schema:
   - Answer, citations (doc_id + chunk_id), retrieval_context with IDs
3) Ensure RAG server returns stable doc_id and chunk_id for every cited source.

### Phase 1: Dataset Integration (generic-first)
1) Implement loaders for:
   - RAGBench (multi-domain baseline; sample across subsets)
   - Qasper (long-doc evidence grounding)
   - SQuAD v2 (unanswerable handling)
   - HotpotQA (multi-hop reasoning)
   - MS MARCO (retrieval ranking realism)
2) Add dataset registry and sampling strategies (by difficulty, query type).
3) Store dataset metadata for reporting and reproducibility.

### Phase 2: Evaluation Framework Selection
1) Use DeepEval as the core evaluation framework.
2) Extend with gold-aware retrieval metrics and citation correctness scoring.
3) Implement LLM-as-judge pipeline with user-configurable evaluator models.
4) Add manual review export for citation checks (CSV/JSON with answer + cited chunks).

### Phase 3: Metrics Implementation
1) Retrieval metrics (Recall@K, Precision@K, MRR, NDCG) using gold passage IDs.
2) Answer metrics:
   - Faithfulness, relevance, correctness
3) Citation metrics:
   - Precision/recall at chunk and section level
4) Unanswerable handling metrics (SQuAD v2).
5) Long-form completeness scoring for summary/report questions.

### Phase 4: Benchmark Orchestration
1) End-to-end pipeline:
   - Load dataset
   - Ingest documents
   - Run queries
   - Capture answer + citations + retrieval contexts
2) Store runs with full config snapshots (LLM, embedding, retrieval, reranker, prompts).
3) Provide per-domain and per-query-type breakdowns.

### Phase 5: Scoring
1) Compute weighted overall score based on user-specified objectives.
2) Provide a scorecard view (all raw metrics) + Pareto front.
3) Show tradeoffs across models, search strategies, and prompting variants.

### Phase 6: Validation and Guardrails
1) Add regression thresholds for critical metrics (citation precision, faithfulness).
2) Add evaluation sampling to CI for quick regression checks.
3) Add manual review workflow for high-risk configurations.

## Next Steps (implementation-ready)
- Decide framework choice (RAGAS vs DeepEval extension).
- Confirm dataset priority order and subset sizes.
- Define citation schema for doc/section/chunk IDs in API responses.
- Agree on default objective weights and UI display format.
- Consider future work: replace heuristic long-form completeness with an LLM-judge or semantic matching approach.

## Completed Work Summary (this implementation)
- New evaluation_v2 module (datasets, runner, metrics, CLI): 13 new files under `services/rag_server/evaluation_v2`.
- Legacy eval removal + CLI shim: 27 files removed under `services/rag_server/evaluation` and `services/rag_server/tests/evaluation`, plus 2 shim files under `services/rag_server/evaluation`.
- Query API chunk/citation support: 3 files updated under `services/rag_server/schemas`, `services/rag_server/api/routes`, `services/rag_server/pipelines`.
- Metrics and recommendation updates (retrieval/citation/abstention/long-form definitions): 4 files updated under `services/rag_server/services`, `services/rag_server/schemas`, `services/rag_server/tests`.
- Explicit numeric citations and prompting: 2 files updated under `services/rag_server/infrastructure/llm` and `services/rag_server/pipelines` (plus query response wiring already counted).
- Manual review export workflow: 1 new file + 2 updates under `services/rag_server/evaluation_v2`.
- Eval config expansion (citation scope/format + abstention phrases): 3 files updated under `config/` and 2 under `services/rag_server`.
- API migration mapping: 1 new file under `docs/`.

## Appendix: Common Domains With Strong Public Datasets

- Legal: CUAD (contracts), LegalBench-RAG
- Finance / Accounting: FinQA, TAT-QA
- Biomedical / Clinical: PubMedQA
- Technical / Product Docs: TechQA
- Scientific Papers: Qasper
- Government / Policy: GovReport
- Customer Support / Multi-doc QA: MultiDoc2Dial (may require gated access)
