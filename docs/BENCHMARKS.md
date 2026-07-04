# Pipeline Benchmarks

Measured with `services/rag_server/scripts/benchmark_pipeline.py` against the
full docker compose stack.

## 2026-07-04 — Phase 1 before/after

**Hardware:** MacBook (Apple M2, 16GB RAM), macOS 27.0.

**Config:** `active.inference=gpt5-mini` (OpenAI), `active.embedding=nomic-embed`
(Ollama, local), `active.reranker=minilm-l6`. `enable_hybrid_search=true`,
`enable_contextual_retrieval=false` (default — unchanged by this benchmark),
`reranker.enabled=true`.

**Before:** commit `57017d9` (pre-refactor, main before Phase 1).
**After:** commit `4c4fd0c` (Phase 1 tasks 1.1–1.4 + benchmark script; 1.6/1.7
query-path work not yet applied).

Corpus: `sample_documents/` (3 files, 6 chunks total — `python_basics.txt` 1
chunk, `machine_learning.txt` 3 chunks, `docker_guide.md` 2 chunks). 10 fixed
questions, run sequentially then concurrently.

| Stage | Before (mean) | After (mean) |
|---|---|---|
| Ingestion (end-to-end, per doc) | 1070.5ms | 1065.9ms |
| Queries (sequential, mean / p95) | 6852.6ms / 8607.1ms | 6781.7ms / 9559.4ms |
| Queries (concurrent, mean / p95) | 13320.4ms / 15610.0ms | 12862.4ms / 18674.4ms |

**Result: no measurable difference at this corpus size.** This is expected,
not a negative finding about Phase 1 — the sample corpus doesn't exercise the
part of the pipeline the completed tasks target:

- Task 1.2 (batched embedding) collapses N per-chunk HTTP calls into
  `ceil(N/32)` batched calls. With only 6 chunks total, that's 6 calls → 1
  call per document — real, but too small a base to move *total* ingestion
  wall-clock time, which is dominated by chunking, DB writes, and the
  document-copy step, not the embedding HTTP round-trips.
- Task 1.3 (concurrent contextual retrieval) only runs when
  `enable_contextual_retrieval: true`. It stays off by default (per the
  refactor plan), so this benchmark doesn't exercise it at all.
- Query latency is essentially unchanged because the query path itself
  (Task 1.6 — async query path) hasn't been implemented yet; time is
  dominated by the unchanged synchronous `gpt5-mini` generation call.
- Task 1.4 (parallel task workers) helps when multiple documents are
  ingested concurrently under load; a single 3-document batch doesn't
  saturate `WORKER_CONCURRENCY=2`.

**Takeaway:** re-run this benchmark against a corpus with hundreds of chunks
per document (and `enable_contextual_retrieval: true`) to see the intended
effect, and again after Task 1.6 lands to see query-path gains. The per-stage
`[EMBEDDING]`/`[CONTEXTUAL]` log lines (suppressed here by
`LOG_LEVEL=WARNING`) give a truer stage-by-stage breakdown than end-to-end
wall time on a small corpus — set `LOG_LEVEL=INFO` for a more granular rerun.
