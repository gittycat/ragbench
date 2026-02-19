# Testing

## Test Categories

| Category | Tests | Dependencies | Command |
|----------|-------|-------------|---------|
| **Unit** | ~32 | None (fully mocked) | `just test-unit` |
| **Integration** | 25 | Docker services (PostgreSQL, Ollama, RAG server) | `just test-integration` |
| **Integration (full)** | 25 + slow | Same + longer timeouts | `just test-integration-full` |
| **Evaluation** | 27 | ANTHROPIC_API_KEY + Docker services | `just test-eval` |

## Test Structure

```
tests/
├── test_*.py                         # Unit tests (mocked dependencies)
├── integration/
│   ├── conftest.py                   # Shared fixtures (api_client, upload_and_wait, test_document, cleanup)
│   ├── test_infrastructure.py        # Service connectivity + database schema (8 tests)
│   ├── test_pipeline.py              # Upload → index → query → delete round-trips (9 tests)
│   ├── test_chat_sessions.py         # Session lifecycle via HTTP API (5 tests)
│   ├── test_async_upload.py          # Async task upload workflow + progress tracking (8 tests)
│   ├── test_api_documents.py         # Document management endpoints (1 test)
│   └── test_hybrid_search.py         # BROKEN: imports dead ChromaDB module (see Roadmap)
└── evaluation/                       # Evaluation tests (DeepEval + Claude judge)
```

## Pytest Markers

```python
@pytest.mark.integration  # Requires --run-integration flag
@pytest.mark.slow         # Tests taking > 30s (skipped by default, use --run-slow)
@pytest.mark.eval         # Requires --run-eval and ANTHROPIC_API_KEY
```

## Integration Test Design

Integration tests verify **execution, connectivity, and structure** — not answer quality. Quality evaluation (relevance, faithfulness, hallucination) is the eval suite's job.

This separation follows industry consensus for RAG testing (Meta's [2024 RAG systems paper](https://arxiv.org/abs/2312.10997), Anthropic's [contextual retrieval guide](https://www.anthropic.com/news/contextual-retrieval), and Docker's [testing best practices](https://docs.docker.com/build/tests/)):

- **Embeddings**: "Did ChromaDB store a 768-dim vector?" — not "Are the embeddings semantically meaningful?"
- **Retrieval**: "Did a query return >0 chunks?" — not "Were they the right chunks?"
- **Reranking**: "Did the reranker produce output?" — not "Did it improve precision?"
- **LLM response**: "Did `/query` return a non-empty string?" — not "Was the answer faithful?"

One **canary test** bridges the gap: uploads a document with a unique random marker (`MARKER_<uuid>`), queries for it, and asserts the marker appears in a returned source chunk. This is a write-then-read test that catches content loss during ingestion → embedding → retrieval, without judging quality.

### test_infrastructure.py — Service Connectivity & Database Schema (8 tests, fast)

Read-only checks that the system is wired correctly. No uploads or mutations.

**What it covers**: `/health` + `/config` + `/models/info` endpoints return expected fields. PostgreSQL has `pg_textsearch` extension. Required tables exist (`documents`, `document_chunks`, `chat_sessions`, `chat_messages`, `job_batches`, `job_tasks`). `idx_tasks_claimable` partial index exists for SKIP LOCKED. Ollama has required models (`gemma3`, `nomic-embed-text`).

### test_pipeline.py — Upload → Index → Query → Delete (9 tests, slow)

Core RAG loop: upload a document, verify it's indexed, query it, delete it.

**What it covers**: Document appears in `/documents` after upload. Document has >0 chunks. `/query` returns non-empty answer with `session_id`. Query with `include_chunks=True` returns source list. Canary test: marker string survives full round-trip. PDF upload and query. Delete removes document from list. Query after delete returns 200 (not 500). Delete nonexistent returns 404.

### test_chat_sessions.py — Session Lifecycle (5 tests, mixed speed)

Session creation, history growth, clearing, and temporary session behavior.

**What it covers**: Query creates a session with user + assistant messages in history. Second query on same session grows history. Nonexistent session returns empty (not 500). Clear empties history. Temporary sessions don't appear in session list.

### test_async_upload.py — Async Processing & Progress (8 tests, slow)

Full async upload workflow via SKIP LOCKED task queue with progress tracking and edge cases.

**What it covers**: Upload via API → task-worker processes via SKIP LOCKED → status shows completed. Progress tracking accuracy for multi-chunk documents. Single and multiple file uploads appear in document list. Large markdown and PDF status progression. Invalid batch ID handling. Concurrent uploads without race conditions.

## Integration Test Strategy

- **No separate test-runner service** — tests reuse the `rag-server` service definition to avoid config drift
- **Local/debug:** `docker compose exec -T rag-server .venv/bin/pytest tests/integration -v --run-integration`
- **CI:** `docker compose run --rm rag-server .venv/bin/pytest tests/integration -v --run-integration`
