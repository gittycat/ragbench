# CLAUDE.md

This file provides guidance to AI coding agent (eg: claude.ai/code) when working with code in this repository.

## Python

### Prefer Functions Over Classes
- Use module-level functions instead of classes for stateless operations
- Avoid singleton pattern (`_instance = None` + `get_instance()`) - just use functions
- Classes are appropriate for: stateful objects, resource lifecycle management, framework integration

### When Classes ARE Appropriate
- Object maintains state across method calls (e.g., accumulators, trackers)
- Resource management with setup/teardown (e.g., DB connections, HTTP clients)
- Framework requirements (e.g., `logging.Filter` subclasses)

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

```python
# BAD: Class with only @classmethod/@staticmethod
class Registry:
    _items = {}
    @classmethod
    def register(cls, name, item): cls._items[name] = item

# GOOD: Module-level state and functions
_items = {}
def register(name, item): _items[name] = item
```

### Documentation
- Skip docstrings on private helper functions (`_foo()`) - use inline comments if logic is non-obvious
- Type hints replace parameter/return documentation
- Keep public API docstrings to one line when possible


## Project Overview

Local RAG system with FastAPI REST API using Docling + LlamaIndex for document processing, PostgreSQL with pgvector for vector storage, pg_search for BM25 full-text search, and Ollama for LLM inference. Implements Hybrid Search (pg_search BM25 + pgvector + RRF) and Contextual Retrieval (Anthropic method) for improved accuracy.

## Architecture

### Service Design

- `rag-server` (port 8001): RAG API service - exposed to host for client integration
- `pgmq-worker`: Async document processing worker (polls PostgreSQL pgmq queue)
- `postgres`: PostgreSQL 18 with pgvector, pg_search (ParadeDB), and pgmq extensions

### Network Isolation

- `public` network: RAG server accessible from host on port 8001
- `private` network: Internal services (PostgreSQL, worker)
- Ollama runs on host machine at `http://host.docker.internal:11434`

### Document Processing Flow

**Upload (Async via pgmq):**
1. Documents uploaded → `rag-server` `/upload` endpoint
2. Files saved to `/tmp/shared` volume, pgmq tasks queued (one per file)
3. pgmq-worker polls queue, processes tasks: DoclingReader → contextual prefix → DoclingNodeParser → nodes
4. Embeddings generated per-chunk with progress tracking (via PostgreSQL job_tasks table)
5. Nodes stored in PostgreSQL via PGVectorStore (pgvector)
6. BM25 index auto-refreshes via pg_search

**Query (Synchronous):**
1. Query → hybrid retrieval (pg_search BM25 + pgvector + RRF with top-k=10)
2. Reranking → top-n selection (5-10 nodes)
3. LLM (gemma3:4b) generates answer with retrieved context
4. Chat history saved to PostgreSQL (session-based, persistent)

### Key Patterns

- **Hybrid Search**: pg_search BM25 (sparse) + pgvector (dense) with RRF fusion (k=60), auto-indexes
- **Contextual Retrieval**: LLM-generated document context prepended to chunks before embedding
- **Retrieval Pipeline**: Hybrid BM25+Vector → RRF fusion → reranking → dynamic top-n selection (5-10 nodes)
- **Async Processing**: pgmq + PostgreSQL for background uploads with real-time progress tracking
- **Conversational RAG**: Session-based memory with `condense_plus_context` mode, PostgreSQL persistence
- **Document Storage**: Nodes with `document_id` metadata, ID format: `{doc_id}-chunk-{i}`

## Async Upload Architecture

### Upload Flow

1. **Client**: Uploads files → `POST /upload`
2. **RAG Server**: Saves files to `/tmp/shared`, enqueues pgmq tasks, returns `batch_id`
3. **pgmq-worker**: Polls queue, processes tasks asynchronously, updates PostgreSQL progress
4. **Client**: Polls `GET /tasks/{batch_id}/status` for progress
5. **Completion**: All tasks complete, files indexed in PostgreSQL (pgvector + pg_search)

### Progress Tracking

- **Storage**: PostgreSQL tables (job_batches, job_tasks)
- **Structure**: `{batch_id, total, completed, tasks: {task_id: {filename, status, chunks...}}}`
- **Granularity**: Per-task status + per-chunk progress (for large documents)
- **Updates**: Client polls progress endpoint

### Shared Volume

- **Path**: `/tmp/shared` (Docker volume `docs_repo`)
- **Purpose**: File transfer between RAG server and pgmq-worker
- **Cleanup**: Worker deletes files after processing (success or error)

## Common Commands

### Development

```bash
# Start services (requires Ollama running on host)
docker compose up -d

# Build after dependency changes
docker compose build

# View logs
docker compose logs -f rag-server
docker compose logs -f pgmq-worker
docker compose logs -f postgres

# Stop services
docker compose down -v
```

### Configuration

```bash
# Show RAG configuration (compact)
just show-config

# Show full RAG configuration with all settings
just show-config-full
```

**Note:** Configuration banners are automatically displayed when running evaluation/benchmark CLI tools to help track which models are being used during testing.

### Testing

**Task Runner**: just. Use context7 with id "just_systems-man"

```bash
# Unit tests
just test-unit

# integration tests
just test-integration

# run eval tests
just test-eval

# Full eval tests
just test-eval-full
```

### Evaluation

**Framework:** DeepEval with Anthropic Claude

See [docs/DEEPEVAL_IMPLEMENTATION_SUMMARY.md](docs/DEEPEVAL_IMPLEMENTATION_SUMMARY.md) for complete guide.

```bash
cd services/rag_server
uv sync --group eval

# Quick evaluation (5 test cases)
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/python -m evaluation.cli eval --samples 5

# Full evaluation
.venv/bin/python -m evaluation.cli eval

# Show dataset stats
.venv/bin/python -m evaluation.cli stats

# Pytest integration
pytest tests/test_rag_eval.py --run-eval --eval-samples=5
```

### CI/CD

**Framework:** Forgejo (self-hosted GitHub alternative with integrated Actions)

See [docs/FORGEJO_CI_SETUP.md](docs/FORGEJO_CI_SETUP.md) for complete setup guide.

```bash
# Start Forgejo server and runner
docker compose -f docker-compose.ci.yml up -d

# Access Forgejo Web UI
open http://localhost:3000

# View CI logs
docker compose -f docker-compose.ci.yml logs -f

# Stop CI infrastructure
docker compose -f docker-compose.ci.yml down
```

**CI Pipeline** (`.forgejo/workflows/ci.yml`):
- **Core tests**: Run on every push (40 tests, ~30s)
- **Eval tests**: Optional, off by default (27 tests, ~2-5min) - trigger with `[eval]` in commit or manual dispatch
- **Docker build**: Verifies RAG server + webapp build

## Critical Implementation Details

### Hybrid Search & Contextual Retrieval

**Hybrid Search** (`infrastructure/search/`):
- pg_search BM25 via `PgSearchBM25Retriever` + pgvector via `PGVectorStore`
- Combined with `HybridRRFRetriever` using RRF fusion (k=60)
- BM25 index auto-refreshes on insert/update/delete (no manual refresh needed)
- Falls back to vector-only if `ENABLE_HYBRID_SEARCH=false`

**Contextual Retrieval** (`pipelines/ingestion.py`):
- LLM generates 1-2 sentence document context per chunk via `add_contextual_prefix_to_chunk()`
- Context prepended before embedding (zero query-time overhead)
- Toggle via `ENABLE_CONTEXTUAL_RETRIEVAL` (default: false for speed)

### Docling + LlamaIndex Integration

**Document Processing** (`pipelines/ingestion.py`):
- **CRITICAL**: Must use `DoclingReader(export_type=DoclingReader.ExportType.JSON)` - DoclingNodeParser requires JSON
- PostgreSQL JSONB handles nested metadata (no flattening needed)

**Vector Store** (`infrastructure/search/vector_store.py`):
- `PGVectorStore` from llama-index-vector-stores-postgres
- HNSW index for fast approximate nearest neighbor search
- Connection pooling via asyncpg + SQLAlchemy 2.0

**RAG Pipeline** (`pipelines/inference.py`):
- Hybrid: `create_hybrid_retriever()` → `CondensePlusContextChatEngine.from_defaults(retriever=...)`
- Vector-only fallback: `index.as_chat_engine(chat_mode="condense_plus_context")`
- Reranking: `SentenceTransformerRerank` (model: `cross-encoder/ms-marco-MiniLM-L-6-v2`, returns top 5-10 nodes)
- Session memory via `ChatMemoryBuffer` with `PostgresChatStore`
- Reranker pre-initializes at startup to avoid first-query timeout

**Embeddings** (`infrastructure/llm/embeddings.py`): `OllamaEmbedding` (default: `nomic-embed-text:latest`, 768 dimensions)

### Docker Build

**PyTorch CPU Index:** Dockerfile uses `--index-strategy unsafe-best-match` to resolve package version conflicts between PyTorch CPU index and PyPI.

**Build Tools:** Includes gcc, g++, make for pystemmer compilation (required by sentence-transformers)

### Database Schema

```sql
-- Core tables (see services/postgres/init.sql)
documents           -- Source files (id, file_name, file_type, file_hash, uploaded_at)
document_chunks     -- Chunks with embeddings (id, document_id, content, embedding vector(768))
chat_sessions       -- Chat sessions (id, title, llm_model, is_archived)
chat_messages       -- Chat history (id, session_id, role, content)
job_batches         -- Upload batches (id, total_tasks, completed_tasks, status)
job_tasks           -- Individual tasks (id, batch_id, filename, status, chunks)

-- Extensions
pgvector            -- Vector similarity search
pg_search           -- BM25 full-text search (ParadeDB)
pgmq                -- PostgreSQL message queue
```

### Configuration Files

**YAML-based Configuration (config.yml):**

The system uses YAML configuration for all model settings and prompt templates. Each model definition includes a `requires_api_key` field indicating whether it needs an API key.

**Supported providers:**
- **Local (no API keys):** ollama
- **Cloud (require API keys):** openai, anthropic, google, deepseek, moonshot

**API Keys (secrets/.env):**

Secrets are stored in `secrets/.env` (copy from `secrets/.env.example`). Use the naming convention `{PROVIDER}_API_KEY`:

```bash
OPENAI_API_KEY=      # For OpenAI models
ANTHROPIC_API_KEY=   # For Anthropic models and DeepEval
GOOGLE_API_KEY=      # For Google models
DEEPSEEK_API_KEY=    # For DeepSeek models
MOONSHOT_API_KEY=    # For Moonshot models
```

Models with `requires_api_key: true` in config.yml will fail validation if the corresponding API key is not set.

**Environment Variables (docker-compose.yml only):**

Minimal environment variables - most config moved to YAML:
- `DATABASE_URL`: PostgreSQL connection string (default: `postgresql+asyncpg://raguser:ragpass@postgres:5432/ragbench`)
- `MAX_UPLOAD_SIZE=80`: Max upload size in MB
- `LOG_LEVEL=WARNING`: Logging level (INFO or DEBUG for development)

API keys are loaded from `secrets/.env` via Docker Compose `env_file`.

**Note:** pgmq-worker shares all RAG Server configuration (config.yml and secrets/.env). Ollama settings (`base_url`, `keep_alive`) are now in `config.yml` per model.

## API Endpoints

**RAG Server** (port 8001):

*Info:* `GET /health`, `GET /config` (max_upload_size_mb), `GET /models/info`
*Chat:* `POST /query` (session_id optional), `GET /chat/history/{session_id}`, `POST /chat/clear`
*Documents:* `GET /documents`, `POST /upload` (async, returns batch_id), `GET /tasks/{batch_id}/status`, `DELETE /documents/{document_id}`

**Supported formats:** `.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.htm`, `.asciidoc`, `.adoc`

## Key Files

**Pipelines** (`services/rag_server/pipelines/`):
- `ingestion.py`: Document chunking, contextual retrieval, embedding
- `inference.py`: RAG query with hybrid search + reranking + chat engine

**Search Infrastructure** (`services/rag_server/infrastructure/search/`):
- `vector_store.py`: PGVectorStore wrapper for LlamaIndex
- `bm25_retriever.py`: pg_search BM25 retriever
- `hybrid_retriever.py`: RRF fusion combining BM25 + vector

**Database** (`services/rag_server/infrastructure/database/`):
- `postgres.py`: Async connection pool (asyncpg + SQLAlchemy 2.0)
- `models.py`: SQLAlchemy ORM models
- `repositories/`: Document, session, job repositories

**Tasks** (`services/rag_server/infrastructure/tasks/`):
- `pgmq_queue.py`: pgmq client wrapper
- `pgmq_worker.py`: Worker polling pgmq queue
- `worker.py`: Document processing job

**LLM Infrastructure** (`services/rag_server/infrastructure/llm/`):
- `factory.py`: Multi-provider LLM client factory
- `prompts.py`: System, context, and condense prompts
- `embeddings.py`: Ollama embedding configuration

**Services** (`services/rag_server/services/`):
- `session.py`: Chat session metadata management

**API:** `main.py` (FastAPI), `api/routes/`

**Evaluation:** `evaluation/` (DeepEval), `evals/data/golden_qa.json` (10 Q&A pairs)

## Common Issues

- **Ollama not accessible:** `docker compose exec rag-server curl http://host.docker.internal:11434/api/tags`
- **PostgreSQL connection fails:** Check `DATABASE_URL` environment variable, verify postgres service is healthy
- **Docker build fails:** Ensure `--index-strategy unsafe-best-match` in Dockerfile
- **Tests fail:** Use `.venv/bin/pytest` directly, not `uv run pytest`
- **Reranker performance:** First query downloads model (~80MB), adds ~100-300ms latency
- **Hybrid search not working:** Check `ENABLE_HYBRID_SEARCH=true` in config.yml
- **Contextual retrieval not working:** Check `ENABLE_CONTEXTUAL_RETRIEVAL=true`, verify OLLAMA_URL accessible
- **pgmq-worker issues:** `docker compose logs pgmq-worker`. Worker auto-restarts, tasks timeout after 1 hour
- **Slow processing:** Contextual retrieval takes ~85% of time (LLM calls per chunk). See [Performance Analysis](docs/PERFORMANCE_ANALYSIS.md)

## Detailed Documentation

- **[DEVELOPMENT.md](DEVELOPMENT.md)** - API docs, configuration, troubleshooting, roadmap
- **[Forgejo CI/CD Setup](docs/FORGEJO_CI_SETUP.md)** - Self-hosted CI/CD setup
- **[Conversational RAG](docs/CONVERSATIONAL_RAG.md)** - Session management, chat memory
- **[Performance Optimizations](docs/PERFORMANCE_OPTIMIZATIONS_SUMMARY.md)** - Recent optimizations (15x speedup)
- **[Performance Analysis](docs/PERFORMANCE_ANALYSIS.md)** - Bottlenecks, timing breakdown
- **[Ollama Optimization](docs/OLLAMA_OPTIMIZATION.md)** - Keep-alive, KV cache, prompt caching
- **[DeepEval Implementation](docs/DEEPEVAL_IMPLEMENTATION_SUMMARY.md)** - Metrics, workflow, best practices
- **[PostgreSQL Migration Plan](docs/POSTGRES_MIGRATION_PLAN.md)** - Migration from ChromaDB/Redis to PostgreSQL

## Testing Strategy

**Tests:** `services/rag_server/tests/`
- Unit tests (mocked dependencies)
- Integration tests (requires docker services, `--run-integration` flag)
- Evaluation tests (requires `--run-eval` flag and `ANTHROPIC_API_KEY`)

**Unit Tests:** Mock external dependencies via `@patch`. DoclingReader/DoclingNodeParser → Node objects, mock PostgreSQL via SQLAlchemy fixtures.

**Integration Tests:** Test real services (PostgreSQL, Ollama). Key tests:
- `test_pdf_full_pipeline`: PDF → Docling → PostgreSQL → queryable
- `test_pgmq_task_completes`: Async upload via pgmq
- `test_corrupted_pdf_handling`: Graceful error handling
