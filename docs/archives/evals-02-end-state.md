# RAG Evaluation: End State Design

This document describes what the eval system will look like once all three tiers are fully implemented. It does not describe files or code — it describes behavior, APIs, datasets, and metrics.

---

## Overview

The eval system runs as a standalone CLI service (`services/evals/`) that calls the RAG server over HTTP. Three evaluation modes (tiers) are available:

```
just test-eval --tier tier1    # Generation isolation (fast, no ingestion)
just test-eval --tier tier2    # Full pipeline (requires running RAG server + worker)
just test-eval --tier tier3    # Domain-specific synthetic dataset
```

Each tier produces a scorecard with the same metric groups, enabling direct comparison across tiers.

---

## Tier 1 — Generation Quality

**Goal:** Measure LLM + prompt quality in isolation from retrieval.

**How it works:**
1. Load a dataset that includes gold passages (RAGBench, SQuAD v2)
2. For each question, POST to `POST /query/with-context` on the RAG server — passing gold passages as the context
3. The RAG server feeds those passages directly to the LLM, skipping retrieval entirely
4. Evaluate the generated answer against expected answer and gold passages

**RAG Server Endpoint (new):**
```
POST /query/with-context
Body: {
  query: str,
  context_passages: [{ text: str, doc_id: str }],
  session_id?: str
}
Response: same shape as POST /query
```

This endpoint uses the same LLM, same system prompt, same chat engine as production — so Tier 1 results are directly comparable to Tier 2 results.

**Datasets:**
- `ragbench` (HuggingFace: `rungalileo/ragbench`) — 12 subsets across tech, legal, medical, finance
- `squad_v2` (HuggingFace: `rajpurkar/squad_v2`) — unanswerable questions for abstention testing

**Metrics computed:**
- Faithfulness
- Answer correctness
- Answer relevancy
- Abstention accuracy (squad_v2 only)
- False positive rate, false negative rate (squad_v2 only)
- Latency (LLM-only, no retrieval)
- Cost per query

**No ingestion required.** Documents are never uploaded to the RAG system.

---

## Tier 2 — End-to-End Pipeline

**Goal:** Measure the complete pipeline including chunking, embedding, indexing, retrieval, reranking, and generation.

**How it works:**
1. Group gold passages by document; upload each as a text file via `POST /upload`
2. Poll `GET /tasks/{batch_id}/status` until all documents are processed
3. Track the mapping from gold doc IDs to the RAG server's assigned document IDs
4. For each question, POST to `POST /query` with `include_chunks: true`
5. Map retrieved chunk doc_ids back to gold doc_ids using the tracking mapping
6. Evaluate retrieval quality (did the right docs come back?) and generation quality
7. Delete all uploaded documents in a `finally` block to avoid polluting the RAG system

**Datasets:**
- `ragbench` — gold passages ingested as documents, queried by question
- `hotpotqa` (HuggingFace: `hotpotqa/hotpot_qa`) — multi-hop retrieval, Wikipedia passages
- `qasper` (HuggingFace: `allenai/qasper`) — long scientific documents, citation accuracy
- `msmarco` (HuggingFace: `microsoft/ms_marco`) — passage ranking benchmark

**Metrics computed (all of Tier 1 plus):**
- Recall@K (K = 1, 3, 5, 10)
- Precision@K (K = 1, 3, 5)
- MRR (Mean Reciprocal Rank)
- NDCG@10
- Citation precision, citation recall, section accuracy
- Latency (full pipeline including retrieval and reranking)

**Requires:** Running `rag-server` + `task-worker` + `postgres` + `chromadb` + Ollama.

---

## Tier 3 — Domain-Specific Synthetic Evaluation

**Goal:** Measure system quality on real uploaded documents using generated QA pairs.

**How it works (two sub-phases):**

**Phase A — Synthesis (one-time, not per eval run):**
1. Take a sample of real uploaded documents from the RAG system
2. Use the configured LLM to generate QA pairs: questions answerable from specific passages
3. Filter by difficulty (embedding similarity threshold)
4. Optionally: human review to promote from "silver" to "gold"
5. Save to `evals/data/golden_qa.json` with versioning metadata

Command:
```
uv run python -m evals.cli synthesize --source-dir /path/to/docs --samples 200
```

**Phase B — Evaluation (same as Tier 2):**
- Run end-to-end eval using the golden dataset (loaded from `evals/data/golden_qa.json`)
- Same metrics as Tier 2

**External resources needed:**
- The configured LLM (via `infrastructure/llm/factory.py`) for synthesis
- No new HuggingFace datasets — uses local documents

---

## Metrics Summary

| Metric Group | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| Faithfulness | ✓ | ✓ | ✓ |
| Answer correctness | ✓ | ✓ | ✓ |
| Answer relevancy | ✓ | ✓ | ✓ |
| Abstention | ✓ (squad_v2) | — | opt |
| Recall@K | — | ✓ | ✓ |
| Precision@K | — | ✓ | ✓ |
| MRR | — | ✓ | ✓ |
| NDCG | — | ✓ | ✓ |
| Citation precision/recall | — | ✓ | ✓ |
| Latency | ✓ | ✓ | ✓ |
| Cost | ✓ | ✓ | ✓ |

---

## Weighted Scoring

Each run produces a single weighted score computed from metric group averages:

| Weight Dimension | Default Weight |
|-----------------|----------------|
| Accuracy (answer correctness) | 30% |
| Faithfulness | 20% |
| Citation quality | 20% |
| Retrieval quality | 15% |
| Cost | 10% |
| Latency | 5% |

Weights are user-configurable in `EvalConfig`. Tier 1 runs redistribute retrieval weight to other dimensions automatically (no retrieval metrics).

---

## External Dependencies

| Resource | Purpose | Required for |
|----------|---------|--------------|
| HuggingFace `datasets` lib | Loading public benchmarks | Tier 1, Tier 2 |
| `rungalileo/ragbench` | Multi-domain QA | Tier 1, Tier 2 |
| `rajpurkar/squad_v2` | Abstention testing | Tier 1 |
| `hotpotqa/hotpot_qa` | Multi-hop retrieval | Tier 2 |
| `allenai/qasper` | Long-doc citation | Tier 2 |
| `microsoft/ms_marco` | Retrieval ranking | Tier 2 |
| Anthropic / Ollama API | LLM-as-judge | All tiers (generation metrics) |
| Running RAG stack | Target system | Tier 2, Tier 3 |

Datasets are cached locally after first download via the HuggingFace `datasets` cache.

---

## Configuration

Tier is specified per run via CLI `--tier` flag or in `EvalConfig`:

```yaml
tier: tier2
datasets:
  - ragbench
  - hotpotqa
samples_per_dataset: 100
judge:
  enabled: true
  provider: anthropic
  model: claude-sonnet-4-20250514
cleanup_on_failure: true    # always delete ingested docs, even on error
```

Each dataset declares which tiers it supports. Selecting a dataset for a tier it doesn't support raises a configuration error with a clear message.

---

## Result Storage

Each run saves to `data/eval_runs/{run_id}.json`. The JSON includes:
- Config snapshot (LLM model, embedding model, retrieval settings, contextual retrieval on/off, hybrid search on/off)
- Per-metric results with sample sizes and confidence
- Weighted score
- Timestamps and error counts

CLI commands `eval stats`, `eval compare`, and `eval export` operate on these stored runs.
