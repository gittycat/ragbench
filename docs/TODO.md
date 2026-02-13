# TODO

## Speedup contextual retrieval

Contextual retrieval is slow (~3s per chunk, sequential LLM calls in `services/rag_server/pipelines/ingestion.py:310`).

- [ ] **Batch/parallel LLM calls** — Replace sequential `for` loop with `asyncio.gather` using concurrency limit (5-10 parallel calls). Expected 3-5x speedup.
- [ ] **Use a faster/local model for context generation** — Use a local Ollama model (e.g., `llama3.2:3b`) specifically for contextual prefix generation to eliminate network latency. 2-10x faster.
- [ ] **Cache contextual prefixes** — Hash chunk content, skip LLM call if prefix already exists for that hash. Helps on re-uploads.
- [ ] **Disable as fallback** — `enable_contextual_retrieval: false` in config.yml. Hybrid search (BM25 + vector + RRF + reranking) is already strong without it.
