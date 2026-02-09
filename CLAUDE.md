# CLAUDE.md

## Python

### Prefer Functions Over Classes
- Use module-level functions instead of classes for stateless operations
- Avoid singleton pattern (`_instance = None` + `get_instance()`) - just use functions
- Classes are appropriate for: stateful objects, resource lifecycle management, framework integration

### Anti-Patterns to Avoid
```python
# BAD: Stateless class with singleton
class FooService:
    def do_thing(self, x): return x * 2

_service = None
def get_foo_service():
    global _service
    if _service is None: _service = FooService()
    return _service

# GOOD: Just a function
def do_thing(x): return x * 2
```

### Documentation
- Skip docstrings on private helpers - use inline comments if non-obvious
- Type hints replace parameter/return documentation
- Keep public API docstrings to one line when possible

## Project Overview

Local RAG system: FastAPI + Docling + LlamaIndex + PostgreSQL (pgvector + pg_search BM25) + Ollama. Implements Hybrid Search (BM25 + vector + RRF) and Contextual Retrieval.

## Architecture

- `rag-server` (port 8001): RAG API — `public` network, exposed to host
- `pgmq-worker`: Async document processing worker — `private` network
- `postgres`: PostgreSQL 18 with pgvector, pg_search (ParadeDB), pgmq — `private` network
- Ollama: runs on host at `http://host.docker.internal:11434`

### Document Processing

**Upload:** POST /upload → files saved to `/tmp/shared` → pgmq tasks queued → worker processes: DoclingReader → contextual prefix → DoclingNodeParser → embeddings → PGVectorStore + pg_search BM25

**Query:** hybrid retrieval (BM25 + pgvector + RRF, top-k=10) → SentenceTransformerRerank (top 5-10) → LLM answer → chat history saved to PostgreSQL

### Key Patterns
- **Hybrid Search**: pg_search BM25 + pgvector with RRF fusion (k=60), auto-indexes
- **Contextual Retrieval**: LLM-generated context prepended to chunks before embedding (toggle: `ENABLE_CONTEXTUAL_RETRIEVAL`)
- **Async Processing**: pgmq + PostgreSQL job_batches/job_tasks for progress tracking
- **Conversational RAG**: `condense_plus_context` mode, `ChatMemoryBuffer` + `PostgresChatStore`
- **Document IDs**: `{doc_id}-chunk-{i}`

## Package Management

**Tool:** `uv` (not pip). All services use `pyproject.toml`.

```bash
cd services/rag_server
uv sync                    # install deps
uv sync --group eval       # install with eval group
uv add <package>           # add dependency
uv add --group dev <pkg>   # add dev dependency
uv run pytest              # run commands in venv
```

## Commands

**Task runner:** `just` (context7 id: `just_systems-man`)

```bash
just test-unit             # unit tests
just test-integration      # integration tests (requires docker services)
just test-eval             # eval tests (requires ANTHROPIC_API_KEY)
just test-eval-full        # full eval suite
just show-config           # show RAG configuration
just show-config-full      # show full config with all settings
```

**Evaluation:** DeepEval with Anthropic Claude. See `docs/DEEPEVAL_IMPLEMENTATION_SUMMARY.md`.
```bash
cd services/rag_server && uv sync --group eval
ANTHROPIC_API_KEY=sk-ant-... uv run python -m evaluation.cli eval --samples 5
```

**CI/CD:** Forgejo with `.forgejo/workflows/ci.yml`. Core tests run on every push. Eval tests triggered with `[eval]` in commit message or manual dispatch.

## Critical Implementation Details

### Hybrid Search & Contextual Retrieval
- `PgSearchBM25Retriever` + `PGVectorStore` combined via `HybridRRFRetriever` (k=60)
- BM25 index auto-refreshes on insert/update/delete
- Falls back to vector-only if `ENABLE_HYBRID_SEARCH=false`
- Contextual retrieval: `add_contextual_prefix_to_chunk()` in `pipelines/ingestion.py`

### Docling + LlamaIndex
- **CRITICAL**: Must use `DoclingReader(export_type=DoclingReader.ExportType.JSON)` — DoclingNodeParser requires JSON
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`, pre-initialized at startup
- Embeddings: `OllamaEmbedding` (`nomic-embed-text:latest`, 768 dimensions)

### Docker Build
- Dockerfile uses `--index-strategy unsafe-best-match` for PyTorch CPU index resolution
- Includes gcc, g++, make for pystemmer compilation (sentence-transformers dep)

### Database Schema
```sql
-- services/postgres/init.sql
documents, document_chunks, chat_sessions, chat_messages, job_batches, job_tasks
-- Extensions: pgvector, pg_search, pgmq
```

### Configuration
- `config.yml`: all model settings, prompt templates, provider config
- API keys stored as files under `secrets/` (filename = key name)
- Supported providers: ollama (local), openai/anthropic/google/deepseek (cloud, require API keys)
- Env vars (docker-compose.yml): `DATABASE_URL`, `MAX_UPLOAD_SIZE=80`, `LOG_LEVEL=WARNING`

## Key Files

All under `services/rag_server/`:
- `pipelines/ingestion.py` — document chunking, contextual retrieval, embedding
- `pipelines/inference.py` — RAG query, hybrid search, reranking, chat engine
- `infrastructure/search/` — `vector_store.py`, `bm25_retriever.py`, `hybrid_retriever.py`
- `infrastructure/database/postgres.py` — async connection pool (asyncpg + SQLAlchemy 2.0)
- `infrastructure/database/repositories/` — document, session, job repos
- `infrastructure/llm/factory.py` — multi-provider LLM client factory
- `infrastructure/tasks/` — `pgmq_queue.py`, `pgmq_worker.py`, `worker.py`
- `evaluation/` — DeepEval framework, `evals/data/golden_qa.json` (10 Q&A pairs)

## Common Issues

- **Docker build fails:** ensure `--index-strategy unsafe-best-match` in Dockerfile
- **Reranker slow on first query:** downloads model (~80MB), adds ~100-300ms
- **pgmq-worker issues:** `docker compose logs pgmq-worker` — auto-restarts, tasks timeout after 1 hour
- **Slow processing:** contextual retrieval takes ~85% of time (LLM calls per chunk)

## Testing

Tests in `services/rag_server/tests/`. Unit tests use `@patch` mocks. Integration tests require `--run-integration` flag. Eval tests require `--run-eval` flag + `ANTHROPIC_API_KEY`.
