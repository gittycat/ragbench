# Architecture Refactor Plan — Performance + Deployment Tiers

Created: 2026-07-03
Status: Approved for implementation
Implementer notes: follow tasks in order within each phase. One commit per task
(short one-liner messages). Use `uv` for all dependency changes. Prefer
module-level functions over classes (see CLAUDE.md). Run `just test-unit` after
every task; integration-affecting tasks list extra verification steps.

## Product decisions (context — do not re-litigate)

The product serves privacy-sensitive users (SMEs and individuals) who do NOT
trust third parties — including OpenAI/Anthropic — with document content.
Three deployment tiers share this one codebase:

1. **Local-only tier** — everything on the user's machine (current default).
   Slow today largely due to self-inflicted pipeline problems, fixed in Phase 1.
2. **Confidential-compute VM tier** — the existing Docker stack deployed to a
   rented Linux GPU VM (optionally with confidential-compute attestation),
   serving open-weight models via vLLM. Thin clients connect over the network.
3. **Opt-in cloud tier** — frontier LLM APIs (OpenAI/Anthropic) used for
   **generation only**, with reversible PII masking. The document corpus
   (embedding, enrichment, reranking) never leaves the local/VM boundary.

Explicitly out of scope:
- LAN GPU server as a product tier (ruled out)
- Native desktop packaging / Tauri rewrite
- Replacing LlamaIndex, ChromaDB, or pg_textsearch
- Cloud reranking APIs (reranker stays local in all tiers)

Key insight driving Phase 1: current slowness is mostly sequential code, not
model speed. Ingestion embeds one chunk per HTTP round-trip; contextual
enrichment is one blocking LLM call per chunk; the task worker processes one
task at a time; every query pays an extra condense LLM round-trip.

---

## Phase 1 — Pipeline performance (benefits all tiers)

### Task 1.1 — Embedding provider factory

**Problem:** `services/rag_server/infrastructure/llm/embeddings.py` hard-codes
`OllamaEmbedding`. `config.yml` already defines OpenAI embedding models and
`EmbeddingConfig` (in `infrastructure/config/models_config.py`) already carries
`provider`, `api_key`, `base_url`, `requires_api_key` — the factory ignores them.

**Change:**
- Rewrite `get_embedding_function()` to dispatch on `config.embedding.provider`,
  mirroring the `_PROVIDER_CONFIG` mapping pattern in
  `infrastructure/llm/factory.py`:
  - `ollama` → `llama_index.embeddings.ollama.OllamaEmbedding(base_url, model_name)`
  - `openai` → `llama_index.embeddings.openai.OpenAIEmbedding(model, api_key, api_base)`
- Add dependency: `cd services/rag_server && uv add llama-index-embeddings-openai`
  (search for the current released version first).
- Support an optional `embed_batch_size` field on `EmbeddingConfig` (default 64
  for ollama, 100 for openai) and pass it to the embedding constructor
  (`embed_batch_size` kwarg is supported by all LlamaIndex embedding classes).
- Add `embed_batch_size` to the embedding entries in `config.yml` (optional key).

**Dimension-mismatch guard:** switching embedding models changes vector
dimensions (nomic-embed = 768, openai-3-small = 1536) and silently corrupts
retrieval. At startup (where `Settings.embed_model` is set in
`services/rag_server/core/config.py`), if the ChromaDB collection is non-empty,
compare its stored embedding dimension against the active model's dimension
(embed the string "dim-probe" once to learn it). On mismatch, raise a clear
error: the user must delete + re-index, or switch the model back.

**Tests:** unit tests with `@patch` verifying provider dispatch and that
`requires_api_key` validation still fires. Existing tests in
`services/rag_server/tests/` show the mock style.

### Task 1.2 — Batched embedding + bulk insert in ingestion

**Problem:** `embed_and_index_chunks()` in
`services/rag_server/pipelines/ingestion.py` loops node-by-node calling
`index.insert_nodes([node])` — one embedding HTTP call and one Chroma insert
per chunk.

**Change:**
- Rewrite `embed_and_index_chunks()` to process nodes in batches
  (`INGEST_BATCH_SIZE = 32`, module constant):
  1. For each batch: `texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in batch]`
  2. `embeddings = Settings.embed_model.get_text_embedding_batch(texts)`
  3. Assign `node.embedding = emb` for each pair.
  4. `index.insert_nodes(batch)` — nodes with pre-set embeddings are not
     re-embedded by LlamaIndex.
  5. Call `progress_callback(processed_so_far, total)` once per batch.
- Adapt `_insert_node_with_retry` into `_process_batch_with_retry` — keep the
  exponential backoff and connection-error detection, retry at batch
  granularity.
- Keep the per-batch timing logs (same log format, `[EMBEDDING]` prefix) so the
  performance breakdown in `ingest_document()` still works.

**Verification:** `just test-unit`, then integration: upload a sample document
(`sample_documents/machine_learning.txt`) via the running stack and confirm
chunk count and query results unchanged. Log output should show batches, not
per-chunk lines.

### Task 1.3 — Concurrent contextual enrichment

**Problem:** `add_contextual_retrieval()` in `pipelines/ingestion.py` makes one
blocking `llm.complete()` call per chunk, sequentially. This is why
`enable_contextual_retrieval` is off in `config.yml`.

**Change:**
- Add async variant `add_contextual_prefix_to_chunk_async()` using
  `llm.acomplete(prompt)`; same prompt, same fallback-to-original-node on error.
- Rewrite `add_contextual_retrieval()` to run all chunks through
  `asyncio.gather()` bounded by `asyncio.Semaphore` — the exact pattern already
  used in `services/evals/evals/runner.py`.
- Concurrency from config: add `contextual_concurrency: 8` under `retrieval:`
  in `config.yml` and expose it via the retrieval config model in
  `models_config.py`.
- `ingest_document()` is sync but is called from async context
  (`infrastructure/tasks/worker.py::process_document_async`). Check how it is
  invoked there (likely via executor thread). Inside the sync pipeline, wrap
  with `asyncio.run(...)` — safe because the executor thread has no running
  loop. Do NOT call `asyncio.run` from a thread that already has a loop.
- Preserve chunk order in the returned list (gather preserves order).

**Verification:** enable `enable_contextual_retrieval: true` locally, ingest a
multi-chunk document, confirm prefixes present and elapsed time roughly
`ceil(chunks / concurrency) * per-call latency` instead of `chunks * latency`.
Leave the setting `false` in committed `config.yml`.

### Task 1.4 — Parallel task workers

**Problem:** `infrastructure/tasks/task_worker.py::run_worker()` runs a single
claim-process loop; N uploaded documents process serially. The SKIP LOCKED
queue (`db_jobs.claim_next_task`) already guarantees safe concurrent claims.

**Change:**
- Add `WORKER_CONCURRENCY` env var (default `2`). In `run_worker()`, spawn that
  many independent claim loops with `asyncio.gather()` — each loop is the
  current `while not _shutdown` body. Keep the single `check_stuck_tasks()`
  coroutine.
- Move the retry back-off (`await asyncio.sleep(delay)` before
  `reset_task_for_retry`) out of the claim loop's critical path: re-queue
  immediately but have `claim_next_task` respect a `next_attempt_at`-style
  delay ONLY IF the schema already supports it; otherwise keep the sleep —
  with multiple loops the blocking matters less. Do not change DB schema in
  this task.
- Add `WORKER_CONCURRENCY` to the `task-worker` service environment in
  `docker-compose.yml`.
- Connection budget: each loop uses short-lived sessions from the shared pool
  (`pool_size=10, max_overflow=20` per `config.yml`) — fine up to ~8 loops.
  Cap `WORKER_CONCURRENCY` at 8 with a warning log.

**Verification:** upload 3+ documents at once; `docker compose logs task-worker`
shows interleaved `[WORKER] Claimed task` lines; all tasks complete; no
"too many clients" errors from PostgreSQL.

### Task 1.5 — Skip the condense round-trip on empty history

**Problem:** `create_chat_engine()` in `pipelines/inference.py` uses
`CondensePlusContextChatEngine`, which rewrites the query via an extra LLM call
before answering. For the first message of a session there is no history to
condense — pure wasted latency.

**Change:**
- First, verify behavior of the installed `llama-index-core` version: read
  `CondensePlusContextChatEngine._acondense_question` /
  `_condense_question` source in the venv. If it already returns the raw query
  when `chat_history` is empty, no code change — document the finding in the
  commit message and close the task.
- If it does condense unconditionally: subclass in `pipelines/inference.py`
  (`class SkipEmptyCondenseChatEngine(CondensePlusContextChatEngine)`) and
  override the condense method(s) to return the query unchanged when history
  is empty. Use the subclass in `create_chat_engine()`.

**Verification:** first query in a fresh session makes exactly one LLM
generation call (watch logs / token counter); follow-up queries still condense.

### Task 1.6 — Async query path

**Problem:** `query_rag` / `query_rag_stream` are sync; API routes in
`services/rag_server/api/routes/query.py` push them onto executor threads.
Thread hops add latency and cap concurrency at the thread pool size.

**Change (keep scoped — this is the riskiest Phase 1 task):**
- Add `async def query_rag_async(...)` and an async streaming variant in
  `pipelines/inference.py`, using `chat_engine.achat()` and
  `chat_engine.astream_chat()`.
- Session metadata calls (`touch_session`, `get_session_metadata`,
  `update_session_title`) and `PostgresChatStore` may be sync — check each; if
  sync, keep those specific calls on `run_in_executor` inside the async
  function rather than converting the repositories.
- Update `api/routes/query.py` to await the async variants directly (remove
  the outer `run_in_executor` for query endpoints only — leave other routes
  untouched).
- Keep the sync `query_rag` exported and functional: the evals service and
  tests may call it. Do not delete or rename existing functions.

**Verification:** `just test-unit`; integration: non-streaming and streaming
query endpoints return correct answers + sources; run 5 concurrent queries
(e.g. `xargs -P5` curl loop) and confirm no event-loop blocking (health
endpoint stays responsive during queries).

### Task 1.7 — Benchmark before/after

- Write `scripts/benchmark_pipeline.py` (under `services/rag_server/scripts/`):
  times (a) ingestion of `sample_documents/` end-to-end via the upload API and
  (b) 10 sequential + 10 concurrent queries, printing a small table
  (mean/p95 per stage, using log-derived or API-returned `latency_ms`).
- Record results in `docs/BENCHMARKS.md` with date, config (active models,
  hybrid on/off, reranker on/off), and hardware. Run once on the pre-refactor
  commit (git stash or checkout) and once after Phase 1; commit both numbers.

---

## Phase 2 — Tier enablement

### Task 2.1 — vLLM / OpenAI-compatible inference provider

Enables the confidential-VM tier: vLLM exposes an OpenAI-compatible endpoint.

- Add `vllm` to `LLMProvider` enum (`infrastructure/llm/config.py`) and to
  `_PROVIDER_CONFIG` in `factory.py`, mapped to `OpenAILike` from
  `llama-index-llms-openai-like` (`uv add` it; check current version). Params:
  `model`, `api_base` (from `base_url`), `api_key` (optional — vLLM often runs
  keyless behind the network boundary; default to `"none"` when absent),
  `is_chat_model=True`.
- Add a commented example entry under `models.inference` in `config.yml`:
  ```yaml
  # qwen-vllm:
  #   provider: vllm
  #   model: Qwen/Qwen2.5-14B-Instruct
  #   base_url: http://vllm:8000/v1
  #   timeout: 120
  ```
- No API-key file requirement when `requires_api_key` is false — verify the
  validation path in `models_config.py` allows this.

**Tests:** unit test that the factory builds an `OpenAILike` with mapped params.

### Task 2.2 — Deployment overlay for the server tier (auth + TLS)

Thin clients connecting over a network need transport security and auth that
localhost never needed.

- **Auth:** add optional bearer-token auth to `rag-server` (FastAPI
  dependency): if env `RAG_SERVER_AUTH_TOKEN_FILE` is set (Docker secret),
  every request except `/health` must send
  `Authorization: Bearer <token>`. Constant-time comparison (`secrets.compare_digest`).
  Webapp forwards the token from its own env var. When unset, behavior is
  unchanged (local tier).
- **TLS:** add `docker-compose.server.yml` overlay placing
  [Caddy](https://caddyserver.com) (current 2.x image) in front of the webapp
  on the `public` network with automatic HTTPS (or internal CA for
  LAN-less-DNS setups). Only Caddy publishes ports in the overlay; webapp/
  rag-server/evals stop publishing to host.
- Document usage in `DEVELOPMENT.md`: `docker compose -f docker-compose.yml -f
  docker-compose.server.yml up`.

**Tests:** unit test the auth dependency (missing/wrong/right token). Manual:
overlay boots, requests without token get 401, with token 200.

### Task 2.3 — PII masking for the cloud tier (generation only)

Implement `docs/PII_MASKING_IMPLEMENTATION_PLAN.md` **with a narrowed scope**:
masking applies to the generation path only. Embedding, contextual enrichment,
and reranking are local/VM-side by product decision, so the corpus never
transits the mask. Where this plan and the PII doc conflict, this plan wins.

Scope by data path (supersedes the 4-path table in the PII doc):
- **Mask:** user query + retrieved context + chat history sent to the
  generation LLM (`query_rag*` in `pipelines/inference.py`); first message
  sent for session-title generation (`services/session_titles.py`).
- **Do NOT mask (stays inside trust boundary):** contextual enrichment
  (`add_contextual_prefix_to_chunk` — local LLM by decision), embeddings,
  reranker input, eval-judge traffic (evals service, separate concern).

Implementation follows the PII doc otherwise: `infrastructure/pii/` module,
Presidio analyzer + reversible session-scoped token mapping
(`[[[PERSON_0]]]`), `validate_tokens_preserved()` + fuzzy recovery on the
response, output guardrail scan, config block (`pii.enabled`, entity list,
threshold) in `config.yml`, master toggle default `false`.

Streaming caveat: token-by-token unmasking can split a `[[[PERSON_0]]]` token
across SSE events. When `pii.enabled` is true, buffer the stream and emit
unmasked text in sentence-sized flushes (split on `. `, `\n`) — simpler and
safer than partial-token matching. Note the behavior change in the config
comment.

Dependencies: `uv add presidio-analyzer presidio-anonymizer` plus the spaCy
model download in the Dockerfile (`en_core_web_lg`) — check current released
versions and image-size impact; prefer `en_core_web_md` if quality is
acceptable in tests.

**Tests:** unit tests for mask/unmask round-trip, token preservation
validation, fuzzy recovery, and the streaming buffer. One integration test
gated behind `--run-eval` (needs a cloud key) proving an end-to-end masked
query returns an unmasked, correct answer.

### Task 2.4 — Config guardrail: corpus-local enforcement

Add a startup validation (in `models_config.py`) enforcing the privacy
posture: if `pii.enabled` is true (cloud generation tier), the active
**embedding** provider must be a local one (`ollama`, or future local
providers). Fail fast with a clear message otherwise — masking embeddings is
not supported and sending the corpus to a cloud embedder defeats the tier's
purpose. When `pii.enabled` is false, any configured embedding provider is
allowed (user's explicit choice).

---

## Suggested implementation order

1.1 → 1.2 → 1.4 → 1.5 → 1.3 → 1.7 (first pass numbers) → 1.6 → 2.1 → 2.4 → 2.2 → 2.3

Rationale: 1.1/1.2/1.4/1.5 are independent quick wins; 1.3 depends on nothing
but is only observable with contextual retrieval enabled; 1.6 is the riskiest
and benefits from the benchmark existing first; Phase 2 tasks are independent
of each other except 2.4 referencing 2.3's config block (create the `pii:`
config stub in 2.4 if 2.3 hasn't landed yet).

## Global acceptance criteria

- `just test-unit` passes after every task; integration tests
  (`docker compose exec -T rag-server .venv/bin/pytest tests/integration -v
  --run-integration`) pass after 1.2, 1.4, 1.6, 2.2.
- No breaking changes to the public API surface of `rag-server` (webapp and
  evals service are consumers).
- `config.yml` stays backward compatible: every new key optional with a
  default; a pre-refactor config file must still boot the stack.
- Ingestion of `sample_documents/` measurably faster (expect ≥5x on embedding
  stage); first-query latency drops by roughly one LLM round-trip.
