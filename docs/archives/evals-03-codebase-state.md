# RAG Evaluation: Current Codebase State

Last reviewed: 2026-02-19 (by Opus 4.6 + manual verification)

---

## Service Structure

```
services/evals/
├── evals/
│   ├── runner.py         # EvaluationRunner + RAGClient — main orchestrator
│   ├── config.py         # EvalConfig, DatasetName enum, weights
│   ├── cli.py            # CLI entry point (eval, stats, datasets, export, compare)
│   ├── export.py         # JSON/CSV/Markdown export — works correctly
│   ├── datasets/
│   │   ├── base.py       # BaseDatasetLoader abstract class
│   │   ├── registry.py   # Dataset registry / factory
│   │   ├── ragbench.py   # RAGBench HuggingFace loader ✓
│   │   ├── golden.py     # Local golden QA pairs (evals/data/golden_qa.json) ✓
│   │   ├── qasper.py     # Qasper long-doc loader ✓
│   │   ├── hotpotqa.py   # HotpotQA multi-hop loader ✓
│   │   ├── msmarco.py    # MSMARCO retrieval loader ✓
│   │   └── squad_v2.py   # SQuAD v2 loader ✓
│   ├── metrics/
│   │   ├── retrieval.py  # RecallAtK, PrecisionAtK, MRR, NDCG — BUG: returns 1.0 on no gold
│   │   ├── generation.py # Faithfulness, AnswerCorrectness, AnswerRelevancy ✓
│   │   ├── citation.py   # CitationPrecision, CitationRecall, SectionAccuracy ✓
│   │   ├── abstention.py # UnanswerableAccuracy, FPR, FNR ✓
│   │   └── performance.py# LatencyP50, LatencyP95, CostPerQuery — minor issue
│   ├── judges/
│   │   └── llm_judge.py  # LLM-as-judge (multi-provider) — brittle parsing
│   └── schemas/
│       ├── dataset.py    # EvalQuestion, GoldPassage, EvalDataset ✓
│       ├── response.py   # EvalResponse, RetrievedChunk, Citation ✓
│       └── results.py    # MetricResult, Scorecard, EvalRun, WeightedScore ✓
└── infrastructure/
    └── llm/
        └── factory.py    # Multi-provider LLM client ✓
```

---

## What's Solid — Do Not Change

**schemas/** — Clean dataclasses, well-typed, tier-agnostic. `EvalQuestion.gold_passages` already holds passage text, which is what Tier 1 injection and Tier 2 ingestion both need.

**metrics/** — Mathematically correct implementations of all metric groups. Tier-agnostic. Only the no-gold default values need fixing (see bugs below).

**judges/llm_judge.py** — Works for all tiers. The `_parse_response()` method at line ~241 handles `SCORE:` and `REASONING:` prefixes with case-insensitive matching. Brittle but functional — do not rewrite, just be aware.

**datasets/** — HuggingFace integration is correct, proper sampling, gold passage text extracted. `ragbench.py` is particularly solid. The 3 invalid RAGBench subsets (`narrativeqa`, `natural_questions`, `squad`) were already removed; 12 valid subsets remain.

**export.py** — Tier-agnostic JSON/CSV/Markdown export. No changes needed.

**config.py** — Well-designed with YAML support. Needs a new `EvalTier` enum and a `tier` field added — that's an extension, not a rewrite.

**infrastructure/llm/factory.py** — Multi-provider LLM client already present. Will be reused for Tier 3 synthesis (future).

---

## What's Broken — Needs Fixing

### runner.py — Missing Ingestion Phase

This is the core problem. `EvaluationRunner.run()` has no ingestion step:

```
Current flow:
  health_check()
  → get_config()
  → load_datasets()        # Downloads from HuggingFace or reads local files
  → for question in dataset:
        client.query(question.question)   # ← NO ingestion, documents never uploaded
        parse_rag_response()
  → _compute_metrics()     # Compares retrieved_chunks against question.gold_passages
  → _compute_weighted_score()
  → save_run()
```

Result: retrieval metrics return 0.0 for every question (nothing was ingested, nothing is ever retrieved). Generation metrics also 0.0 because "Empty Response" answers don't match expected answers.

### RAGClient — Missing Methods

Current `RAGClient` has only:
- `query(question, session_id)` — POST /query
- `get_config()` — GET /models/info
- `health_check()` — GET /health
- `close()`

Missing:
- `upload_text_as_document(text, filename)` — POST /upload with in-memory file
- `wait_for_batch(batch_id, timeout)` — polls GET /tasks/{batch_id}/status
- `delete_document(document_id)` — DELETE /documents/{id}

Without these, Tier 2 cannot be implemented.

### cli.py — No Tier Flag

The CLI hardcodes RAG server assumptions. There is no `--tier` flag. All datasets are treated the same way. Tier-conditional behavior (health checks, ingestion) is missing.

---

## Known Bugs

### Bug 1: Retrieval Metrics Return 1.0 When No Gold Passages

In `metrics/retrieval.py`:
- `RecallAtK.compute()` — returns `1.0` when `question.gold_passages` is empty
- `MRR.compute()` — returns `1.0` when `question.gold_passages` is empty
- `NDCG.compute()` — returns `1.0` when `question.gold_passages` is empty

These should return `0.0` (or skip/None). The current behavior masks all retrieval failures. This is why `abstention_false_positive_rate: 1.000` appeared in a previous run — the metric was being hit with questions that had no gold passages.

Fix: change early-return `return 1.0` to `return 0.0` in the no-gold-passages branch of each metric.

### Bug 2: Performance Metrics Include Zero-Query Runs

In `metrics/performance.py`: latency and cost metrics are computed even when there are 0 successful queries. This produces `0.000` across the board rather than `None` / skipped. Minor but misleading.

Fix: guard with `if not latencies: return None`.

### Bug 3: citations Returns null Instead of []

RAG server returns JSON `null` for the `citations` field when `include_chunks=False`. This is already worked around in `runner.py` with `raw_response.get("citations") or []`. No further fix needed.

---

## Missing Pieces (New Code Required)

### 1. POST /query/with-context on rag-server (Tier 1)

No endpoint currently exists to accept pre-retrieved context and feed it directly to the LLM. This must be added to `services/rag_server/api/routes/query.py` (or a new route file). It should use the same LLM, system prompt, and chat engine as the regular `/query` endpoint.

### 2. RAGClient ingestion methods (Tier 2)

`upload_text_as_document()`, `wait_for_batch()`, `delete_document()` must be added to `RAGClient` in `runner.py`.

### 3. Ingestion phase in EvaluationRunner.run() (Tier 2)

After loading datasets and before querying:
1. Group all gold passages by `doc_id` across all questions
2. Upload each unique document as a `.txt` file via the upload API
3. Poll until batch completes
4. Build mapping: `gold_doc_id → rag_server_doc_id` (by matching filename in `GET /documents`)
5. Before metric computation: patch `question.gold_passages` to use rag_server doc_ids

The cleanup phase must run in a `finally` block: delete all uploaded doc_ids regardless of eval success or failure.

### 4. EvalTier enum and tier-based routing

`config.py` needs `EvalTier` enum. The runner needs to branch on tier to select the appropriate query method (inject context vs. regular query) and whether to run the ingestion phase.

---

## Dataset Tier Compatibility

| Dataset | gold_passages | text available | Tier 1 | Tier 2 | Notes |
|---------|:---:|:---:|:---:|:---:|-------|
| `ragbench` | ✓ | ✓ (~2000 chars) | ✓ | ✓ | 12 subsets |
| `golden` | ✗ | ✗ | ✓ (no gold needed) | ✗ | Local curated, no source docs |
| `qasper` | ✓ | ✓ | — | ✓ | Long scientific papers |
| `hotpotqa` | ✓ | ✓ | — | ✓ | Wikipedia passages |
| `msmarco` | ✓ | ✓ | — | ✓ | Retrieval ranking |
| `squad_v2` | ✓ | ✓ | ✓ (abstention) | — | Unanswerable questions |

Note: `golden` dataset (`evals/data/golden_qa.json`) currently has 10 manually curated Q&A pairs with no gold passage text — it works for Tier 3 (run end-to-end against already-indexed docs) but requires human-indexed documents to already be in the system.

---

## Pre-Existing Infrastructure Issues (Not Caused by Eval Code)

**BM25 broken:** `function bm25_search(unknown, tsquery) does not exist` — the `pg_textsearch` extension BM25 index is not created. Hybrid search silently falls back to vector-only. This is a RAG server infrastructure bug that predates the eval service. Tier 2 retrieval metrics will reflect vector-only retrieval until this is fixed separately.

---

## Test Coverage

`tests/test_rag_eval.py` covers:
- Metric computation (unit tests with mock data)
- Schema validation
- Dataset loading (with mocked HuggingFace responses)

Missing:
- Tests for Tier 2 ingestion flow (will be added with implementation)
- Tests for `POST /query/with-context` endpoint (will be added with Tier 1)
- Integration tests for full Tier 1 and Tier 2 runs
