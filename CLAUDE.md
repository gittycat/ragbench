# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

Local RAG system with FastAPI REST API using Docling + LlamaIndex for document processing, ChromaDB for vector storage, and Ollama for LLM inference. Implements Hybrid Search (BM25 + Vector + RRF) and Contextual Retrieval (Anthropic method) for improved accuracy.

## Architecture

### Service Design

- `rag-server` (port 8001): RAG API service - exposed to host for client integration
- `celery-worker`: Async document processing worker
- `redis`: Message broker + result backend for Celery, chat memory persistence, progress tracking
- `chromadb`: Vector database (persistent storage)

### Network Isolation

- `public` network: RAG server accessible from host on port 8001
- `private` network: Internal services (ChromaDB, Redis, Celery)
- Ollama runs on host machine at `http://host.docker.internal:11434`

### Document Processing Flow

**Upload (Async via Celery):**
1. Documents uploaded → `rag-server` `/upload` endpoint
2. Files saved to `/tmp/shared` volume, Celery tasks queued (one per file)
3. Celery worker processes tasks: DoclingReader → contextual prefix → DoclingNodeParser → nodes
4. Embeddings generated per-chunk with progress tracking (via Redis)
5. Nodes stored in ChromaDB via VectorStoreIndex
6. BM25 index refreshed with new nodes

**Query (Synchronous):**
1. Query → hybrid retrieval (BM25 + Vector + RRF with top-k=10)
2. Reranking → top-n selection (5-10 nodes)
3. LLM (gemma3:4b) generates answer with retrieved context
4. Chat history saved to Redis (session-based, 1-hour TTL)

### Key Patterns

- **Hybrid Search**: BM25 (sparse) + Vector (dense) with RRF fusion (k=60), auto-refreshes after uploads/deletes
- **Contextual Retrieval**: LLM-generated document context prepended to chunks before embedding
- **Retrieval Pipeline**: Hybrid BM25+Vector → RRF fusion → reranking → dynamic top-n selection (5-10 nodes)
- **Async Processing**: Celery + Redis for background uploads with real-time progress tracking
- **Conversational RAG**: Session-based memory with `condense_plus_context` mode, Redis persistence (1-hour TTL)
- **Document Storage**: Nodes with `document_id` metadata, ID format: `{doc_id}-chunk-{i}`
- **Data Protection**: ChromaDB persistence verification, automated backup/restore scripts

## Async Upload Architecture

### Upload Flow

1. **Client**: Uploads files → `POST /upload`
2. **RAG Server**: Saves files to `/tmp/shared`, queues Celery tasks, returns `batch_id`
3. **Celery Worker**: Processes tasks asynchronously, updates Redis progress
4. **Client**: Polls `GET /tasks/{batch_id}/status` for progress
5. **Completion**: All tasks complete, files indexed in ChromaDB + BM25

### Progress Tracking

- **Storage**: Redis (key: `batch:{batch_id}`, TTL: 1 hour)
- **Structure**: `{batch_id, total, completed, total_chunks, completed_chunks, tasks: {task_id: {...}}}`
- **Granularity**: Per-task status + per-chunk progress (for large documents)
- **Updates**: Client polls progress endpoint

### Shared Volume

- **Path**: `/tmp/shared` (Docker volume `docs_repo`)
- **Purpose**: File transfer between RAG server and Celery worker
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
docker compose logs -f celery-worker
docker compose logs -f redis

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

**Framework:** DeepEval with Anthropic Claude (migrated from RAGAS on 2025-12-07)

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

**Hybrid Search** (`hybrid_retriever.py`):
- Combines BM25 + Vector with RRF fusion (k=60), auto-refreshes after uploads/deletes
- Passed to `CondensePlusContextChatEngine.from_defaults(retriever=...)` in rag_pipeline.py
- Falls back to vector-only if `ENABLE_HYBRID_SEARCH=false`

**Contextual Retrieval** (`document_processor.py`):
- LLM generates 1-2 sentence document context per chunk via `add_contextual_prefix()`
- Context prepended before embedding (zero query-time overhead)
- Toggle via `ENABLE_CONTEXTUAL_RETRIEVAL` (default: false for speed)

### Docling + LlamaIndex Integration

**Document Processing** (`document_processor.py`):
- **CRITICAL**: Must use `DoclingReader(export_type=DoclingReader.ExportType.JSON)` - DoclingNodeParser requires JSON
- ChromaDB metadata must be flat types (str, int, float, bool, None) - filtered via `clean_metadata_for_chroma()`

**Vector Store** (`chroma_manager.py`):
- `ChromaVectorStore` wraps ChromaDB collection, `VectorStoreIndex.from_vector_store()` creates index
- Direct ChromaDB access via `._vector_store._collection`, `get_all_nodes()` for BM25 indexing

**RAG Pipeline** (`rag_pipeline.py`):
- Hybrid: `create_hybrid_retriever()` → `CondensePlusContextChatEngine.from_defaults(retriever=...)`
- Vector-only fallback: `index.as_chat_engine(chat_mode="condense_plus_context")`
- Reranking: `SentenceTransformerRerank` (model: `cross-encoder/ms-marco-MiniLM-L-6-v2`, returns top 5-10 nodes)
- Session memory via `ChatMemoryBuffer` with `RedisChatStore` (1-hour TTL)
- Reranker pre-initializes at startup to avoid first-query timeout

**Embeddings** (`embeddings.py`): `OllamaEmbedding` (default: `nomic-embed-text:latest`)

### Docker Build

**PyTorch CPU Index:** Dockerfile uses `--index-strategy unsafe-best-match` to resolve package version conflicts between PyTorch CPU index and PyPI.

**Build Tools:** Includes gcc, g++, make for pystemmer compilation (required by BM25)

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
- `CHROMADB_URL`: ChromaDB endpoint (default: `http://chromadb:8000`)
- `REDIS_URL`: Redis endpoint (default: `redis://redis:6379/0`)
- `MAX_UPLOAD_SIZE=80`: Max upload size in MB
- `LOG_LEVEL=WARNING`: Logging level (INFO or DEBUG for development)

API keys are loaded from `secrets/.env` via Docker Compose `env_file`.

**Note:** Celery worker shares all RAG Server configuration (config.yml and secrets/.env). Ollama settings (`base_url`, `keep_alive`) are now in `config.yml` per model.

## API Endpoints

**RAG Server** (port 8001):

*Info:* `GET /health`, `GET /config` (max_upload_size_mb), `GET /models/info`
*Chat:* `POST /query` (session_id optional), `GET /chat/history/{session_id}`, `POST /chat/clear`
*Documents:* `GET /documents`, `POST /upload` (async, returns batch_id), `GET /tasks/{batch_id}/status`, `DELETE /documents/{document_id}`

**Supported formats:** `.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.htm`, `.asciidoc`, `.adoc`

## Key Files

**Services** (`services/rag_server/services/`):
- `rag.py`: RAG query with hybrid search + reranking + chat engine
- `hybrid_retriever.py`: BM25 + Vector + RRF implementation
- `document.py`: Document processing with Docling + contextual retrieval
- `chat.py`: Session-based ChatMemoryBuffer with RedisChatStore

**LLM Infrastructure** (`services/rag_server/infrastructure/llm/`):
- `factory.py`: Multi-provider LLM client factory
- `config.py`: LLMConfig dataclass + LLMProvider enum
- `providers.py`: Provider-specific client creators
- `prompts.py`: System, context, and condense prompts
- `embeddings.py`: Ollama embedding configuration

**Database** (`services/rag_server/infrastructure/database/`):
- `chroma.py`: VectorStoreIndex with ChromaDB

**API & Async:** `main.py` (FastAPI), `infrastructure/tasks/celery_app.py`, `tasks.py`

**Evaluation:** `evaluation/` (DeepEval), `evals/data/golden_qa.json` (10 Q&A pairs)

**Backup:** `scripts/backup_chromadb.sh`, `scripts/restore_chromadb.sh`

## Backup & Restore

```bash
# Manual backup (saves to ./backups/chromadb/)
./scripts/backup_chromadb.sh

# Restore from backup
./scripts/restore_chromadb.sh ./backups/chromadb/chromadb_backup_YYYYMMDD_HHMMSS.tar.gz

# Schedule daily at 2 AM (crontab)
0 2 * * * cd /path/to/ragbench && ./scripts/backup_chromadb.sh >> /var/log/chromadb_backup.log 2>&1
```

**Features:** Timestamped backups, 30-day retention, health verification, auto service stop/start

## Common Issues

- **Ollama not accessible:** `docker compose exec rag-server curl http://host.docker.internal:11434/api/tags`
- **ChromaDB connection fails:** RAG server must be on `private` network
- **Docker build fails:** Ensure `--index-strategy unsafe-best-match` in Dockerfile
- **Tests fail:** Use `.venv/bin/pytest` directly, not `uv run pytest`
- **Reranker performance:** First query downloads model (~80MB), adds ~100-300ms latency
- **BM25 not initializing:** Check `ENABLE_HYBRID_SEARCH=true`, requires documents at startup or initializes after first upload
- **Contextual retrieval not working:** Check `ENABLE_CONTEXTUAL_RETRIEVAL=true`, verify OLLAMA_URL accessible
- **Redis/Celery issues:** `docker compose logs redis|celery-worker`. Worker auto-restarts, tasks timeout after 1 hour
- **Slow processing:** Contextual retrieval takes ~85% of time (LLM calls per chunk). See [Performance Analysis](docs/PERFORMANCE_ANALYSIS.md)

## Detailed Documentation

- **[DEVELOPMENT.md](DEVELOPMENT.md)** - API docs, configuration, troubleshooting, roadmap
- **[Forgejo CI/CD Setup](docs/FORGEJO_CI_SETUP.md)** - Self-hosted CI/CD setup
- **[Conversational RAG](docs/CONVERSATIONAL_RAG.md)** - Session management, chat memory
- **[Performance Optimizations](docs/PERFORMANCE_OPTIMIZATIONS_SUMMARY.md)** - Recent optimizations (15x speedup)
- **[Performance Analysis](docs/PERFORMANCE_ANALYSIS.md)** - Bottlenecks, timing breakdown
- **[Ollama Optimization](docs/OLLAMA_OPTIMIZATION.md)** - Keep-alive, KV cache, prompt caching
- **[DeepEval Implementation](docs/DEEPEVAL_IMPLEMENTATION_SUMMARY.md)** - Metrics, workflow, best practices
- **[Accuracy Improvement Plan](docs/RAG_ACCURACY_IMPROVEMENT_PLAN_2025.md)** - Future optimizations
- **[Phase 1 Summary](docs/PHASE1_IMPLEMENTATION_SUMMARY.md)** - Redis chat store, backups, reranker
- **[Phase 2 Summary](docs/PHASE2_IMPLEMENTATION_SUMMARY.md)** - Hybrid search & contextual retrieval

## Testing Strategy

**Tests:** `services/rag_server/tests/`
- 32 unit tests (mocked dependencies)
- 25 integration tests (requires docker services, `--run-integration` flag)
- 27 evaluation tests (requires `--run-eval` flag and `ANTHROPIC_API_KEY`)

**Unit Tests:** Mock external dependencies via `@patch`. DoclingReader/DoclingNodeParser → Node objects, VectorStoreIndex with `._vector_store._collection` for ChromaDB.

**Integration Tests:** Test real services (ChromaDB, Redis, Ollama). Key tests:
- `test_pdf_full_pipeline`: PDF → Docling → ChromaDB → queryable
- `test_bm25_refresh_after_upload`: Index sync after document operations
- `test_celery_task_completes`: Async upload via Celery
- `test_corrupted_pdf_handling`: Graceful error handling
