# Evaluation Implementation Status

**Last Updated**: 2026-01-21
**Purpose**: Context document for continuing eval implementation work

---

## Background

The RAG evaluation system uses **DeepEval** (not RAGAS). The implementation lives in `services/rag_server/evals/`. The goal is to evaluate RAG quality across retrieval, generation, citation, abstention, and performance metrics.

## Current State

### What Exists

**Framework** (`services/rag_server/evals/`):
- `cli.py` - Commands: `eval`, `stats`, `datasets`, `export`, `compare`
- `runner.py` - Orchestrates queries → metrics → scoring
- `config.py` - `EvalConfig`, `JudgeConfig`, `MetricConfig` dataclasses
- `export.py` - JSON/CSV/Markdown export for manual review

**Metrics** (`evals/metrics/`):
- `retrieval.py` - RecallAtK, PrecisionAtK, MRR, NDCG
- `generation.py` - Faithfulness, AnswerCorrectness, AnswerRelevancy (LLM-as-judge)
- `citation.py` - CitationPrecision, CitationRecall, SectionAccuracy
- `abstention.py` - UnanswerableAccuracy, FalsePositiveRate, FalseNegativeRate
- `performance.py` - LatencyP50, LatencyP95, CostPerQuery

**Datasets** (`evals/datasets/`):
- Loaders for: RAGBench, Qasper, SQuAD v2, HotpotQA, MS MARCO
- Registry pattern in `registry.py`

**Test Data** (`services/rag_server/evals/data/`):
- `golden_qa.json` - 10 Q&A pairs (Paul Graham essays)
- `documents/` - 3 HTML source documents
- `golden_baseline.json` - Baseline config snapshot

**CI** (`.forgejo/workflows/ci.yml`):
- Optional `test-eval` job triggered by `[eval]` in commit message

**Tests** (`services/rag_server/tests/test_rag_eval.py`):
- 42 unit tests for metrics, citation extraction, query endpoint, cost calculation, Pareto analysis
- 17 eval-marked tests for dataset loaders and integration

---

## Tasks by Group

### Group 1: Core Framework (Make It Work) ✅ COMPLETE

- [x] Create `tests/test_rag_eval.py` with `@pytest.mark.eval` marker
- [x] Add `include_chunks` parameter to `/query` endpoint in `main.py`
- [x] Return `chunk_id` and `chunk_index` in query response sources
- [x] Verify CLI commands work end-to-end (`eval`, `stats`, `datasets`)

### Group 2: Dataset Integration ✅ COMPLETE

- [x] Test HuggingFace dataset loaders (RAGBench, Qasper, SQuAD v2, HotpotQA, MS MARCO)
- [x] Verify sampling and stratification work correctly
- [x] Handle dataset download/caching gracefully

### Group 3: Citation Metrics ✅ COMPLETE

- [x] Implement citation extraction from LLM answers (parse `[1]`, `[2]` format)
- [x] Wire citation metrics to extracted citations
- [x] Test citation precision/recall calculations

### Group 4: Performance Metrics ✅ COMPLETE

- [x] Token counting via `TokenCountingHandler` (implemented in `pipelines/inference.py`)
- [x] Calculate cost based on model pricing from config (`get_model_cost` in `config.py` with `MODEL_COSTS` lookup)
- [x] Complete Pareto analysis in `compare --pareto` command (`_compute_pareto_from_dicts` in `cli.py`)

### Group 5: Enhancement Plan (Future)

- [ ] Baseline management API (`/metrics/baseline` CRUD)
- [ ] Run comparison API (`/metrics/compare/{run_a}/{run_b}`)
- [ ] Recommendation engine (`/metrics/recommend`)
- [ ] Dashboard components (Svelte)

---

## Key Files Reference

| Purpose | Path |
|---------|------|
| CLI entry | `services/rag_server/evals/cli.py` |
| Runner | `services/rag_server/evals/runner.py` |
| Config | `services/rag_server/evals/config.py` |
| Metrics | `services/rag_server/evals/metrics/` |
| Datasets | `services/rag_server/evals/datasets/` |
| Schemas | `services/rag_server/evals/schemas/` |
| Test data | `services/rag_server/evals/data/golden_qa.json` |
| **Tests** | `services/rag_server/tests/test_rag_eval.py` |
| API migration spec | `docs/eval_api_migration.json` |
| Enhancement plan | `docs/RAG_EVAL_ENHANCEMENT_PLAN.md` |

---

## Running Tests

```bash
cd services/rag_server

# Run unit tests (no external dependencies)
.venv/bin/pytest tests/test_rag_eval.py -v

# Run eval tests (downloads HuggingFace datasets)
.venv/bin/pytest tests/test_rag_eval.py -v --run-eval

# Run all tests
.venv/bin/pytest tests/test_rag_eval.py -v --run-eval
```

---

## Quick Verification

```bash
cd services/rag_server

# Check module imports
python -c "from evals import EvalConfig, run_evaluation"

# List datasets
python -m evals.cli datasets

# Run minimal eval (needs RAG server running + ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python -m evals.cli eval --datasets ragbench --samples 5 --no-judge
```

---

## Notes

- Module is `evals`, not `evaluation` (docs sometimes wrong)
- LLM-as-judge requires `ANTHROPIC_API_KEY` in environment
- `--no-judge` flag skips LLM metrics for faster testing
- Results saved to `data/eval_runs/` directory
- Citation extraction supports `[1]`, `[1,2]`, `[1-3]`, and `(1)` formats
