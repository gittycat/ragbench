# Evaluation Implementation Status

**Last Updated**: 2025-01-21
**Purpose**: Context document for continuing eval implementation work

---

## Background

The RAG evaluation system uses **DeepEval** (not RAGAS). The implementation lives in `services/rag_server/evaluation_cc/`. The goal is to evaluate RAG quality across retrieval, generation, citation, abstention, and performance metrics.

## Current State

### What Exists

**Framework** (`services/rag_server/evaluation_cc/`):
- `cli.py` - Commands: `eval`, `stats`, `datasets`, `export`, `compare`
- `runner.py` - Orchestrates queries → metrics → scoring
- `config.py` - `EvalConfig`, `JudgeConfig`, `MetricConfig` dataclasses
- `export.py` - JSON/CSV/Markdown export for manual review

**Metrics** (`evaluation_cc/metrics/`):
- `retrieval.py` - RecallAtK, PrecisionAtK, MRR, NDCG
- `generation.py` - Faithfulness, AnswerCorrectness, AnswerRelevancy (LLM-as-judge)
- `citation.py` - CitationPrecision, CitationRecall, SectionAccuracy
- `abstention.py` - UnanswerableAccuracy, FalsePositiveRate, FalseNegativeRate
- `performance.py` - LatencyP50, LatencyP95, CostPerQuery

**Datasets** (`evaluation_cc/datasets/`):
- Loaders for: RAGBench, Qasper, SQuAD v2, HotpotQA, MS MARCO
- Registry pattern in `registry.py`

**Test Data** (`services/rag_server/eval_data/`):
- `golden_qa.json` - 10 Q&A pairs (Paul Graham essays)
- `documents/` - 3 HTML source documents
- `golden_baseline.json` - Baseline config snapshot

**CI** (`.forgejo/workflows/ci.yml`):
- Optional `test-eval` job triggered by `[eval]` in commit message

### What's Missing

1. **`tests/test_rag_eval.py`** - Referenced in CLAUDE.md and CI but doesn't exist
2. **`include_chunks` API param** - `/query` endpoint needs this per `docs/eval_api_migration.json`
3. **Citation extraction** - Parsing `[1]`, `[2]` references from LLM answers
4. **End-to-end testing** - CLI commands and HuggingFace dataset loaders untested
5. **Enhancement features** - Baseline API, comparison API, recommendation engine (see `docs/RAG_EVAL_ENHANCEMENT_PLAN.md`)

---

## Tasks by Group

### Group 1: Core Framework (Make It Work)

- [ ] Create `tests/test_rag_eval.py` with `@pytest.mark.eval` marker
- [ ] Add `include_chunks` parameter to `/query` endpoint in `main.py`
- [ ] Return `chunk_id` and `chunk_index` in query response sources
- [ ] Verify CLI commands work end-to-end (`eval`, `stats`, `datasets`)

### Group 2: Dataset Integration

- [ ] Test HuggingFace dataset loaders (RAGBench, Qasper, SQuAD v2, HotpotQA, MS MARCO)
- [ ] Verify sampling and stratification work correctly
- [ ] Handle dataset download/caching gracefully

### Group 3: Citation Metrics

- [ ] Implement citation extraction from LLM answers (parse `[1]`, `[2]` format)
- [ ] Wire citation metrics to extracted citations
- [ ] Test citation precision/recall calculations

### Group 4: Performance Metrics

- [ ] Implement actual token counting (currently placeholder)
- [ ] Calculate cost based on model pricing from config
- [ ] Complete Pareto analysis in `compare --pareto` command

### Group 5: Enhancement Plan (Future)

- [ ] Baseline management API (`/metrics/baseline` CRUD)
- [ ] Run comparison API (`/metrics/compare/{run_a}/{run_b}`)
- [ ] Recommendation engine (`/metrics/recommend`)
- [ ] Dashboard components (Svelte)

---

## Key Files Reference

| Purpose | Path |
|---------|------|
| CLI entry | `services/rag_server/evaluation_cc/cli.py` |
| Runner | `services/rag_server/evaluation_cc/runner.py` |
| Config | `services/rag_server/evaluation_cc/config.py` |
| Metrics | `services/rag_server/evaluation_cc/metrics/` |
| Datasets | `services/rag_server/evaluation_cc/datasets/` |
| Schemas | `services/rag_server/evaluation_cc/schemas/` |
| Test data | `services/rag_server/eval_data/golden_qa.json` |
| API migration spec | `docs/eval_api_migration.json` |
| Enhancement plan | `docs/RAG_EVAL_ENHANCEMENT_PLAN.md` |

---

## Quick Verification

```bash
cd services/rag_server

# Check module imports
python -c "from evaluation_cc import EvalConfig, run_evaluation"

# List datasets
python -m evaluation_cc.cli datasets

# Run minimal eval (needs RAG server running + ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python -m evaluation_cc.cli eval --datasets ragbench --samples 5 --no-judge
```

---

## Notes

- Module is `evaluation_cc`, not `evaluation` (docs sometimes wrong)
- LLM-as-judge requires `ANTHROPIC_API_KEY` in environment
- `--no-judge` flag skips LLM metrics for faster testing
- Results saved to `data/eval_runs/` directory
