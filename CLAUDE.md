# CLAUDE.md


# Global Software Coding Guidelines

## Communication Style
- Keep responses direct and concise without unnecessary affirmations or agreement phrases. Be blunt.
- Dont be a sycophant. Avoid phrases like "You're absolutely right" or "I agree" unless I explicitely ask "Suck up to me".

## Libraries
- When adding libraries or packages, search online and use the most current released version.
- Be progressive when selecting a library or package. Go for libraries and packages that have a high adoption rate unless it is not widely used (eg: less than 1K stars in github).

## Database
- Use SQL Schema for migrations (or migration tool output reviewed as SQL).
- Use **query builders** for most queries, not ORM.
- Optionally, lightweight mappings (dataclasses / pydantic models) without relying on ORM relationship loading for critical paths.
- Keep tricky SQL as explicit SQL (views, CTEs, window functions) and call it directly.

### git usage
- Use short one liners for git commit messages.


## Library and Tool documentation
Use the Svelte MCP server for any Svelte related coding, question or documentation.

Otherwise, use context7 when I need code generation, setup or configuration steps, or
library/API documentation. This means you should automatically use the Context7 MCP
tools to resolve library id when it is not known, and get library docs without me having to explicitly ask.

### Context7 IDs
For tailwind 4, use context7 with id: websites/tailwindcss
For DaisyUI doc, use context7 with id: websites/daisyui


## Language-Specific Guidelines

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

Local RAG system: FastAPI + Docling + LlamaIndex + PostgreSQL (pg_textsearch BM25) + ChromaDB + Ollama. Implements Hybrid Search (BM25 + vector + RRF) and Contextual Retrieval.

## Architecture

- `rag-server` (port 8001): RAG API — `public` network, exposed to host
- `task-worker`: Async document processing worker via SKIP LOCKED — `private` network
- `postgres`: PostgreSQL 17 with pg_textsearch (BM25) — `private` network
- `chromadb`: Vector database for embeddings — `private` network
- Ollama: runs on host at `http://host.docker.internal:11434`

### Document Processing

**Upload:** POST /upload → files saved to `/tmp/shared` → tasks created in job_tasks table → worker claims via SKIP LOCKED → DoclingReader → contextual prefix → DoclingNodeParser → embeddings → ChromaDB (vectors) + PostgreSQL (text + pg_textsearch BM25)

**Query:** hybrid retrieval (BM25 + ChromaDB vectors + RRF, top-k=10) → SentenceTransformerRerank (top 5-10) → LLM answer → chat history saved to PostgreSQL

### Key Patterns
- **Hybrid Search**: pg_textsearch BM25 + ChromaDB vectors with RRF fusion (k=60), auto-indexes
- **Contextual Retrieval**: LLM-generated context prepended to chunks before embedding (toggle: `ENABLE_CONTEXTUAL_RETRIEVAL`)
- **Async Processing**: SKIP LOCKED on job_tasks table for work queue + PostgreSQL job_batches for progress tracking
- **Async Concurrency**: Evals use `asyncio.gather()` + `Semaphore` for parallel RAG queries and LLM judge calls. RAG server offloads sync generators/LLM calls to executor threads to avoid blocking FastAPI's event loop.
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
- `pg_textsearch BM25Retriever` + `ChromaVectorStore` combined via `HybridRRFRetriever` (k=60)
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
-- Extensions: pg_textsearch
-- job_tasks serves as work queue via SKIP LOCKED (idx_tasks_claimable partial index)
-- Note: embedding vectors stored in ChromaDB, not PostgreSQL
```

### Configuration
- `config.yml`: all model settings, prompt templates, provider config
- API keys stored as files under `secrets/` (filename = key name)
- Supported providers: ollama (local), openai/anthropic/google/deepseek (cloud, require API keys)
- Env vars (docker-compose.yml): `DATABASE_HOST`, `DATABASE_PORT`, `CHROMADB_HOST`, `CHROMADB_PORT`, `MAX_UPLOAD_SIZE=80`, `LOG_LEVEL=WARNING`

## Key Files

All under `services/rag_server/`:
- `pipelines/ingestion.py` — document chunking, contextual retrieval, embedding
- `pipelines/inference.py` — RAG query, hybrid search, reranking, chat engine
- `infrastructure/search/` — `vector_store.py`, `bm25_retriever.py`, `hybrid_retriever.py`
- `infrastructure/database/postgres.py` — async connection pool (asyncpg + SQLAlchemy 2.0)
- `infrastructure/database/repositories/` — document, session, job repos
- `infrastructure/llm/factory.py` — multi-provider LLM client factory
- `infrastructure/tasks/` — `task_worker.py`, `worker.py`
- `evaluation/` — DeepEval framework, `evals/data/golden_qa.json` (10 Q&A pairs)

## Common Issues

- **Docker build fails:** ensure `--index-strategy unsafe-best-match` in Dockerfile
- **Reranker slow on first query:** downloads model (~80MB), adds ~100-300ms
- **task-worker issues:** `docker compose logs task-worker` — auto-restarts, stuck tasks reset after 1 hour
- **Slow processing:** contextual retrieval takes ~85% of time (LLM calls per chunk)

## Testing

Tests in `services/rag_server/tests/`. Unit tests use `@patch` mocks. Integration tests require `--run-integration` flag. Eval tests require `--run-eval` flag + `ANTHROPIC_API_KEY`.

### Integration Test Strategy
- **No separate test-runner service** — tests reuse the `rag-server` service definition to avoid config drift (env, secrets, volumes, networks)
- **Local/debug:** `docker compose exec -T rag-server .venv/bin/pytest tests/integration -v --run-integration` — fast, reuses running container
- **CI:** `docker compose run --rm rag-server .venv/bin/pytest tests/integration -v --run-integration` — fresh container, no state leakage
- `exec` uses `RAG_SERVER_URL=http://localhost:8001` (same container); `run` uses `http://rag-server:8001` (sibling container)
