# Development Guide

Technical documentation for the RAG system. Covers architecture, backend services, APIs, configuration, testing, and deployment.

For frontend/UI documentation, see [FRONT_END.md](FRONT_END.md).

## Table of Contents

- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Service Architecture](#service-architecture)
- [Backend Design](#backend-design)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Deployment](#deployment)
- [Evaluation Framework](#evaluation-framework)
- [Observability](#observability)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

## Architecture

### System Overview

```
     ┌──────────────┐              ┌────────────────────────────────────┐
     │   End User   │              │   Ollama                           │
     │   (browser)  │              │                                    │
     └──────┬───────┘              │   • LLM: gemma3:4b                 │
            │                      │   • Embeddings: nomic-embed-text   │
            │                      └──────────────────┬─────────────────┘
            │                                         │ :11434 (localhost)
            │         EXTERNAL NETWORK                │
   =========│=========================================│===================
            │         INTERNAL NETWORK                │
            │    (docker-compose priv network)        │
            │                                         │
            │ :8000                                   │
   ┌────────▼────────┐                                │
   │     WebApp      │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│ ─ ─ ─ ─ ─ ─ ─ ┐
   │   (SvelteKit)   │                                │               │
   └────────┬────────┘                                │               │
            │                                         │               │
            │ :8001                                   │               │
   ┌────────▼───────────────────────────────┐         │               │
   │            RAG Server (FastAPI)        │─────────┘               │
   │                                        │                         │
   │  ┌──────────────────────────────────┐  │                         │
   │  │  Docling + LlamaIndex            │  │                         │
   │  │  • Hybrid Search (BM25+Vector)   │  │                         │
   │  │  • Reranking                     │  │                         │
   │  │  • Contextual Retrieval          │  │                         │
   │  └──────────────────────────────────┘  │                         │
   └───────┬────────────────┬───────────────┘                         │
           │                │                                         │
           │                │ Celery task queue                       │
           │                │                                         │
   ┌───────▼──────┐  ┌──────▼──────┐  ┌─────────────────────┐         │
   │   ChromaDB   │  │    Redis    │  │    Celery Worker    │◄────────┘
   │  (Vectors)   │  │  (Broker +  │  │  (Async document    │
   │              │  │   Memory)   │  │   processing)       │
   └──────────────┘  └─────────────┘  └──────────┬──────────┘
                                                 │
   ┌─────────────────────────────────────────────▼────────────────┐
   │         Shared Volume: /tmp/shared (File Transfer)           │
   └──────────────────────────────────────────────────────────────┘
```

### Network Isolation

| Network | Services | Purpose |
|---------|----------|---------|
| `public` | webapp, rag-server | Accessible from host |
| `private` | chromadb, redis, celery-worker | Internal only |

Ollama runs on host machine, accessed via `host.docker.internal:11434`.

### Data Flow

**Document Upload (Async):**
1. Client uploads files → Webapp proxies to RAG Server
2. RAG Server saves to shared volume, queues Celery task
3. Celery Worker: Docling parsing → chunking → embeddings → ChromaDB
4. BM25 index refreshes automatically
5. Client polls progress via batch_id

**Query (Synchronous):**
1. Query → Hybrid retrieval (BM25 + Vector + RRF fusion)
2. Reranking → Top-N selection
3. LLM generates answer with context
4. Response with sources returned (streaming optional)

## Technology Stack

### Backend Services

| Component | Technology | Version |
|-----------|------------|---------|
| API Framework | FastAPI | 0.118+ |
| Python | Python | 3.13+ |
| Package Manager | uv | Latest |
| Vector Database | ChromaDB | 1.1+ |
| Cache/Broker | Redis | 8+ |
| Task Queue | Celery | 5.5+ |
| Document Parser | Docling | 2.53+ |
| RAG Framework | LlamaIndex | 0.14+ |
| Reranker | SentenceTransformers | 5.1+ |

### Frontend Services

| Component | Technology | Version |
|-----------|------------|---------|
| Framework | SvelteKit | 2.49+ |
| UI Library | DaisyUI | 5.5+ |
| CSS | Tailwind CSS | 4.1+ |
| Runtime | Node.js | 22+ |

### AI Models

| Purpose | Model | Provider | Size |
|---------|-------|----------|------|
| LLM | gemma3:4b | Ollama | 4B params |
| Embeddings | nomic-embed-text | Ollama | 137M params |
| Reranker | ms-marco-MiniLM-L-6-v2 | HuggingFace | 22M params |
| Evaluation | claude-sonnet-4-20250514 | Anthropic | Cloud |

### LLM Provider Support

The system supports multiple LLM providers via factory pattern:
- **Ollama** (default, local)
- **OpenAI**
- **Anthropic**
- **Google Gemini**
- **DeepSeek**
- **Moonshot**

Provider selection via `config/models.yml`.

## Service Architecture

### RAG Server (`services/rag_server/`)

FastAPI application handling document processing and RAG queries.

**Directory Structure:**
```
services/rag_server/
├── main.py                  # FastAPI app entry point
├── api/routes/              # REST endpoints
├── pipelines/               # Ingestion and inference logic
├── infrastructure/          # Database, LLM, task configuration
├── services/                # Business logic (sessions, metrics)
├── schemas/                 # Pydantic request/response models
├── core/                    # Global settings and logging
├── evaluation/              # DeepEval framework
└── tests/                   # Unit and integration tests
```

**Key Modules:**

| Directory | Purpose |
|-----------|---------|
| `pipelines/ingestion.py` | Document processing: parsing, chunking, embedding, indexing |
| `pipelines/inference.py` | Query processing: retrieval, reranking, generation |
| `infrastructure/llm/` | Multi-provider LLM client factory |
| `infrastructure/database/` | ChromaDB vector store management |
| `infrastructure/tasks/` | Celery configuration and workers |
| `infrastructure/config/` | YAML configuration loading |

### Celery Worker

Shares codebase with RAG Server (same Docker image, different entrypoint).

**Why Same Codebase:**
- Task signatures must match between producer and consumer
- Shared business logic (document processing, embeddings)
- Single dependency set prevents version mismatches
- Build once, deploy with different commands

**Task Configuration:**
- Concurrency: 1 worker (sequential processing)
- Timeout: 1 hour per task
- Auto-restart: Yes

### Docker Volumes

| Volume | Purpose |
|--------|---------|
| `chromadb_data` | Vector database persistence |
| `redis_data` | Cache and session persistence |
| `docs_repo` | Shared file upload staging |
| `huggingface_cache` | Reranker model cache |
| `documents_data` | Original document storage for downloads |

## Backend Design

### Ingestion Pipeline

**Supported File Formats:**
`.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.htm`, `.asciidoc`, `.adoc`

**Processing Steps:**

1. **Validation**: Check format, compute SHA-256 hash
2. **Parsing**:
   - Complex formats (PDF, DOCX, etc.): Docling with JSON export → DoclingNodeParser
   - Simple text: SimpleDirectoryReader → SentenceSplitter
3. **Chunking**: 500 tokens, 50 token overlap
4. **Contextual Enhancement** (optional): LLM generates 1-2 sentence context per chunk
5. **Embedding**: Via Ollama or cloud provider
6. **Indexing**: ChromaDB + BM25 index refresh
7. **Storage**: Original file saved for download functionality

**Critical Implementation Notes:**
- DoclingReader MUST use `export_type=JSON` (DoclingNodeParser requirement)
- ChromaDB metadata must be flat types only (str, int, float, bool, None)
- Contextual retrieval adds ~85% to processing time

### Inference Pipeline

**Hybrid Search (BM25 + Vector + RRF):**
- Retrieves top-K from each method (default: 10)
- Reciprocal Rank Fusion with k=60
- Reference: [Pinecone Hybrid Search](https://www.pinecone.io/learn/hybrid-search-intro/)
- Claims 48% improvement in retrieval quality

**Reranking:**
- Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Reranks combined results
- Returns top-N (default: 5)
- Pre-initialized at startup to avoid first-query latency

**Contextual Retrieval (Anthropic Method):**
- LLM generates document context per chunk before embedding
- Zero query-time overhead (context embedded at indexing)
- Reference: [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- Claims 49% reduction in retrieval failures

**Chat Memory:**
- Redis-backed via LlamaIndex RedisChatStore
- Session-based with no TTL (persistent)
- Chat mode: `condense_plus_context`

### Document Metadata Schema

| Field | Type | Description |
|-------|------|-------------|
| `document_id` | string | UUID identifying document |
| `file_name` | string | Original filename |
| `file_type` | string | Extension (.pdf, .txt, etc.) |
| `file_size_bytes` | int | File size |
| `file_hash` | string | SHA-256 for deduplication |
| `path` | string | File directory path |
| `chunk_index` | int | Position within document |
| `uploaded_at` | string | ISO 8601 timestamp |

### Deduplication

Client-side SHA-256 hash computation matches LlamaIndex hashing:
1. Browser computes hash before upload
2. Backend checks ChromaDB metadata for existing hash
3. Duplicates rejected with existing filename reference
4. Only new files processed

## Configuration

### YAML Configuration (`config/models.yml`)

Primary configuration file for models and retrieval settings.

**Structure:**
```yaml
llm:
  provider: ollama          # ollama, openai, anthropic, google, deepseek, moonshot
  model: gemma3:4b
  base_url: http://host.docker.internal:11434
  timeout: 120
  keep_alive: 10m           # Ollama only: -1=forever, 0=unload

embedding:
  provider: ollama
  model: nomic-embed-text:latest
  base_url: http://host.docker.internal:11434

eval:
  provider: anthropic
  model: claude-sonnet-4-20250514

reranker:
  enabled: true
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_n: 5

retrieval:
  top_k: 10
  enable_hybrid_search: true
  rrf_k: 60
  enable_contextual_retrieval: false
```

**Setup:** Copy `config/models.yml.example` to `config/models.yml`.

### Secrets (`secrets/.env`)

API keys and credentials (git-ignored).

```bash
LLM_API_KEY=              # For cloud providers
ANTHROPIC_API_KEY=        # For evaluations
```

**Setup:** Copy `secrets/.env.example` to `secrets/.env`.

### Ollama Configuration (`secrets/ollama_config.env`)

```bash
OLLAMA_URL=http://host.docker.internal:11434
OLLAMA_KEEP_ALIVE=10m
```

### Environment Variables (docker-compose.yml)

| Variable | Default | Purpose |
|----------|---------|---------|
| `CHROMADB_URL` | `http://chromadb:8000` | Vector database endpoint |
| `REDIS_URL` | `redis://redis:6379/0` | Cache and broker endpoint |
| `LOG_LEVEL` | `WARNING` | Logging verbosity |
| `MAX_UPLOAD_SIZE` | `80` | Max upload size in MB |
| `LLM_API_KEY` | - | Cloud LLM API key |
| `ANTHROPIC_API_KEY` | - | Evaluation API key |

## API Reference

Base URL: `http://localhost:8001`

### Health & Configuration

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/config` | System configuration (max_upload_size_mb) |
| GET | `/models/info` | Model names and settings |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/documents` | List documents (supports sorting) |
| POST | `/documents/check-duplicates` | Check file hashes for duplicates |
| POST | `/upload` | Upload files (returns batch_id) |
| GET | `/tasks/{batch_id}/status` | Upload progress |
| DELETE | `/documents/{document_id}` | Delete document |
| GET | `/documents/{document_id}/download` | Download original file |

**Sorting Parameters:** `sort_by` (name, chunks, uploaded_at), `sort_order` (asc, desc)

### Query & Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | RAG query (non-streaming) |
| POST | `/query/stream` | RAG query (SSE streaming) |
| GET | `/chat/history/{session_id}` | Get conversation history |
| POST | `/chat/clear` | Clear session history |

**Query Request:**
```json
{
  "query": "What is...",
  "session_id": "uuid-optional",
  "is_temporary": false
}
```

**Streaming Events:** `token`, `sources`, `done`, `error`

### Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/chat/sessions` | List sessions (paginated) |
| GET | `/chat/sessions/{session_id}` | Get session metadata |
| POST | `/chat/sessions/new` | Create new session |
| DELETE | `/chat/sessions/{session_id}` | Delete session |
| POST | `/chat/sessions/{session_id}/archive` | Archive session |

### Metrics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/metrics/system` | Complete system overview |
| GET | `/metrics/models` | Detailed model info |
| GET | `/metrics/retrieval` | Retrieval pipeline config |
| GET | `/metrics/evaluation/definitions` | Metric definitions |
| GET | `/metrics/evaluation/history` | Past evaluation runs |
| GET | `/metrics/evaluation/summary` | Latest evaluation with trends |
| GET | `/metrics/evaluation/{run_id}` | Get specific evaluation run |
| DELETE | `/metrics/evaluation/{run_id}` | Delete evaluation run |
| GET | `/metrics/baseline` | Get golden baseline |
| POST | `/metrics/baseline/{run_id}` | Set golden baseline |
| DELETE | `/metrics/baseline` | Clear golden baseline |
| GET | `/metrics/compare/{run_a}/{run_b}` | Compare two evaluation runs |
| GET | `/metrics/compare-to-baseline/{run_id}` | Compare run to baseline |
| POST | `/metrics/recommend` | Get config recommendation |

## Development Setup

### Prerequisites

1. **Python 3.13+** with [uv package manager](https://docs.astral.sh/uv/)
2. **Node.js 22+** with npm
3. **Docker** (Docker Desktop, OrbStack, or Podman)
4. **Ollama** running on host with required models:
   ```bash
   ollama pull gemma3:4b
   ollama pull nomic-embed-text
   ```

### Local Development

```bash
# Clone and setup
git clone <repo-url>
cd wt-dashboard

# Backend dependencies
cd services/rag_server
uv sync
uv sync --group dev    # Add test dependencies
uv sync --group eval   # Add evaluation dependencies

# Frontend dependencies
cd ../webapp
npm install

# Configuration
cp config/models.yml.example config/models.yml
cp secrets/.env.example secrets/.env
cp secrets/ollama_config.env.example secrets/ollama_config.env

# Start infrastructure
docker compose up chromadb redis -d

# Run RAG server (development)
cd services/rag_server
.venv/bin/uvicorn main:app --reload --port 8001

# Run frontend (development)
cd services/webapp
npm run dev
```

### Task Runner (just)

This project uses [just](https://just.systems/) for task automation.

```bash
# List all tasks
just

# Development
just setup              # Install dependencies
just docker-up          # Start all services
just docker-down        # Stop services
just docker-logs        # View logs

# Testing
just test-unit          # Unit tests only
just test-integration   # Integration tests (requires docker)
just test-eval          # Quick evaluation (5 samples)
just test-eval-full     # Full evaluation

# Deployment
just deploy local       # Deploy locally
just deploy cloud       # Deploy to cloud
just deploy-down local  # Stop deployment

# Version management
just show-version       # Show current version
just inject-version X.Y.Z  # Update version in manifests
just release X.Y.Z      # Full release workflow
```

## Testing

### Test Categories

| Category | Count | Requirements | Command |
|----------|-------|--------------|---------|
| Unit | 32 | None (mocked) | `just test-unit` |
| Integration | 25 | Docker services | `just test-integration` |
| Evaluation | 27 | ANTHROPIC_API_KEY | `just test-eval` |

### Test Structure

```
tests/
├── test_*.py                    # Unit tests (mocked dependencies)
├── integration/                 # Integration tests
│   ├── test_document_pipeline.py
│   ├── test_hybrid_search.py
│   ├── test_async_upload.py
│   └── test_error_recovery.py
└── evaluation/                  # Evaluation tests
    ├── test_dataset_loader.py
    ├── test_retrieval_eval.py
    └── test_reranking_eval.py
```

### Key Integration Tests

| Test | Validates |
|------|-----------|
| `test_pdf_full_pipeline` | PDF → Docling → ChromaDB → queryable |
| `test_bm25_refresh_after_upload` | Index sync after document operations |
| `test_celery_task_completes` | Async upload via Celery |
| `test_corrupted_pdf_handling` | Graceful error handling |

### Pytest Markers

```python
@pytest.mark.integration  # Requires --run-integration flag
@pytest.mark.slow         # Tests taking > 30s
@pytest.mark.eval         # Requires --run-eval and ANTHROPIC_API_KEY
```

## CI/CD

### Forgejo Setup

Self-hosted Git + CI/CD using Forgejo (GitHub Actions compatible).

**Infrastructure:**
```bash
# Start CI infrastructure
docker compose -f docker-compose.ci.yml up -d

# Access Web UI
open http://localhost:3000

# Register runner (get token from admin panel)
docker exec forgejo-runner forgejo-runner register \
  --instance http://forgejo:3000 \
  --token <TOKEN> \
  --name docker-runner \
  --labels docker:docker://node:20,docker:docker://python:3.13
```

### Pipeline (`.forgejo/workflows/ci.yml`)

**Triggers:**
- Every push to any branch
- Pull requests to main
- Manual workflow dispatch

**Jobs:**

| Job | Duration | Always Runs | Requirements |
|-----|----------|-------------|--------------|
| Core Tests | ~30s | Yes | None |
| Eval Tests | ~2-5min | Optional | ANTHROPIC_API_KEY |
| Docker Build | ~5-10min | Yes | None |

**Triggering Evaluation Tests:**
- Commit message containing `[eval]`
- Manual workflow dispatch with checkbox

### Secrets Configuration

Repository Settings → Secrets and Variables → Actions:
- `ANTHROPIC_API_KEY`: Required for evaluation tests

## Deployment

### Environment-Based Deployment

```bash
# Local (OrbStack/Docker Desktop)
just deploy local

# Cloud (configure docker-compose.cloud.yml first)
just deploy cloud

# Stop
just deploy-down local
```

### Compose File Structure

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Base configuration |
| `docker-compose.local.yml` | Local overrides (debug logging) |
| `docker-compose.cloud.yml` | Cloud overrides (registry images) |
| `docker-compose.ci.yml` | CI/CD infrastructure |

### Version Management

Version derived from git tags:

```bash
just show-version          # Display current version
just inject-version 0.2.0  # Update manifests
just release 0.2.0         # Tag, inject, commit, push
```

### Registry-Based Deployment (Future)

For production with container registry:
1. CI builds and pushes images on tag
2. Cloud server pulls new images
3. Compose restarts with new versions

## Evaluation Framework

### DeepEval Integration

Migrated from RAGAS (2025-12-07) for better CI/CD integration.

**Metrics:**

| Metric | Category | Threshold | Description |
|--------|----------|-----------|-------------|
| Contextual Precision | Retrieval | 0.7 | Are retrieved chunks relevant? |
| Contextual Recall | Retrieval | 0.7 | Did we retrieve all needed info? |
| Faithfulness | Generation | 0.7 | Is answer grounded in context? |
| Answer Relevancy | Generation | 0.7 | Does answer address question? |
| Hallucination | Safety | 0.5 | Unsupported information rate |

**LLM Judge:** Claude Sonnet 4 (Anthropic)

### Evaluation Commands

```bash
# CLI usage
cd services/rag_server
export ANTHROPIC_API_KEY=sk-ant-...

# Quick evaluation (5 samples)
.venv/bin/python -m evaluation.cli eval --samples 5

# Full evaluation
.venv/bin/python -m evaluation.cli eval

# Save results for metrics API
.venv/bin/python -m evaluation.cli eval --save

# Dataset statistics
.venv/bin/python -m evaluation.cli stats

# Generate synthetic Q&A pairs
.venv/bin/python -m evaluation.cli generate document.txt -n 10
```

### Golden Dataset

Location: `eval_data/golden_qa.json`
Current size: 10 Q&A pairs from Paul Graham essays
Target: 100+ pairs for comprehensive evaluation

### Evaluation Datasets

Five public datasets are supported for comprehensive RAG evaluation, each targeting specific aspects (retrieval, generation, citation, abstention). See [docs/RAG_EVALUATION_DATASETS.md](docs/RAG_EVALUATION_DATASETS.md) for dataset details, implementation plan, and integration guidance.

## Observability

### Metrics API

Comprehensive visibility into system configuration and performance.

**Core Endpoints:**
- `/metrics/system`: Complete overview
- `/metrics/models`: Model details with references
- `/metrics/retrieval`: Pipeline configuration
- `/metrics/evaluation/history`: Past evaluation runs
- `/metrics/evaluation/summary`: Trends analysis
- `/metrics/evaluation/{run_id}`: Get/delete specific run

**Baseline & Comparison:**
- `/metrics/baseline`: Get/set/clear golden baseline
- `/metrics/compare/{a}/{b}`: Compare two runs
- `/metrics/compare-to-baseline/{run_id}`: Compare to baseline
- `/metrics/recommend`: Get optimal config recommendation

### Health Monitoring

Component health status via `/metrics/system`:
- ChromaDB: Vector database connectivity
- Redis: Cache and broker connectivity
- Ollama: LLM availability

### Key Metrics to Track

| Category | Metrics |
|----------|---------|
| Retrieval | Contextual Precision, Contextual Recall, MRR, Hit Rate |
| Generation | Faithfulness, Answer Relevancy, Hallucination Rate |
| Operational | Latency (P50, P95), Tokens per query, Cost |

### Research References

- [Evidently AI - RAG Evaluation Guide](https://www.evidentlyai.com/llm-guide/rag-evaluation)
- [Braintrust - RAG Evaluation Tools 2025](https://www.braintrust.dev/articles/best-rag-evaluation-tools)
- [Patronus AI - RAG Best Practices](https://www.patronus.ai/llm-testing/rag-evaluation-metrics)

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Ollama not accessible | Check host binding: `curl http://localhost:11434/api/tags` |
| ChromaDB connection fails | Verify `private` network connectivity |
| Docker build fails | Ensure `--index-strategy unsafe-best-match` in Dockerfile |
| Tests fail | Use `.venv/bin/pytest` not `uv run pytest` |
| Reranker slow first query | Model downloads ~80MB on first use |
| BM25 not initializing | Requires documents or initializes after first upload |
| Contextual retrieval not working | Check `enable_contextual_retrieval: true` in config |

### Service Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f rag-server
docker compose logs -f celery-worker
docker compose logs -f redis
docker compose logs -f chromadb
```

### Database Reset

```bash
docker compose down
docker volume rm wt-dashboard_chromadb_data
docker volume rm wt-dashboard_redis_data
docker compose up -d
```

### Backup & Restore

```bash
# Manual backup
./scripts/backup_chromadb.sh

# Restore
./scripts/restore_chromadb.sh ./backups/chromadb_backup_*.tar.gz

# Scheduled (crontab)
0 2 * * * cd /path/to/project && ./scripts/backup_chromadb.sh
```

## Roadmap

### Completed

**Phase 1** (Oct 2025):
- Redis-backed chat memory
- ChromaDB backup/restore
- Reranker optimization
- Startup persistence verification

**Phase 2** (Oct 2025):
- Hybrid search (BM25 + Vector + RRF)
- Contextual retrieval (Anthropic method)
- Auto-refresh BM25 after uploads/deletes

**Evaluation Migration** (Dec 2025):
- DeepEval framework integration
- Anthropic Claude as LLM judge
- Pytest integration with custom markers
- Unified CLI for evaluation

**CI/CD Implementation** (Dec 2025):
- Forgejo self-hosted Git + CI/CD
- GitHub Actions-compatible workflows
- Automated testing on push/PR
- Docker build verification

**Metrics & Observability API** (Dec 2025):
- Comprehensive metrics endpoints
- Evaluation history and trends
- Component health monitoring

### Planned

- Webapp integration for metrics visualization
- Expand golden dataset to 100+ Q&A pairs
- Parent document retrieval (sentence window)
- Query fusion (multi-query generation)
- Multi-user support with authentication
- Additional file formats (CSV, JSON)

### Future / TBD

Most of these points would be needed for enterprise deployement. 
An Service Level Agreement document would define and quantify many of these.

- Authentication. Basic user authentication.
For enterprise: integration with Directory Service (eg: LDAP, IAM, SSO)
- Authorisation: Basic admin and user roles.
For enterprise: integration with Directory Service to get finer grain permisisons. Implement finer grain permissions in ragbench.
- Multi-modal support. Images, video, voice support in prompt.

- Initial data load: initial indexing of docs in the datastore.
This can take hours depending on the amount and type of documents and the embedding model used.
- Very large document or total storage. Current max size: TODO: find this out.
- Security: Create test suite targeted at finding vulnerability with the RAG (prompt injection, document content injection, etc.).
This requires research beyond the normal OWASP top 10 type issues.
- Monitoring: infrastructure and services. Alerting and escalation rules for support.
- Disaster Recovery. Includes backups.
- High Availability. What is the max downtime allowed at one time, and what is the 9's of availability.
- Data retention - prompts, answers, etc. Needed in corporate settings.
- High load: Max number of concurrent users allowed, maximum time allowed to answer under stress.


## Production Considerations

This project is not production-ready for enterprise deployment.

**Missing Capabilities:**

1. **Security**: Authentication, authorization, API keys, audit logging
2. **Observability**: Infrastructure monitoring, APM, distributed tracing
3. **High Availability**: Load balancing, failover, health-based routing
4. **Disaster Recovery**: Automated backups, cross-region replication, RTO/RPO SLAs

**Hardware Requirements:**
- Minimum: 4GB RAM, any CPU (CPU-only PyTorch)
- Recommended: GPU with 8GB+ VRAM for larger models
- For private deployment: In-house server with powerful GPU needed for full-size open models

## Documentation Index

| Document | Purpose |
|----------|---------|
| [FRONT_END.md](FRONT_END.md) | Frontend/UI development |
| [CLAUDE.md](CLAUDE.md) | Project instructions for Claude Code |
| [docs/FORGEJO_CI_SETUP.md](docs/FORGEJO_CI_SETUP.md) | CI/CD setup guide |
| [docs/DEEPEVAL_IMPLEMENTATION_SUMMARY.md](docs/DEEPEVAL_IMPLEMENTATION_SUMMARY.md) | Evaluation framework |
| [docs/CONVERSATIONAL_RAG.md](docs/CONVERSATIONAL_RAG.md) | Session management |
| [docs/PERFORMANCE_OPTIMIZATIONS_SUMMARY.md](docs/PERFORMANCE_OPTIMIZATIONS_SUMMARY.md) | Performance tuning |
| [docs/PHASE1_IMPLEMENTATION_SUMMARY.md](docs/PHASE1_IMPLEMENTATION_SUMMARY.md) | Phase 1 details |
| [docs/PHASE2_IMPLEMENTATION_SUMMARY.md](docs/PHASE2_IMPLEMENTATION_SUMMARY.md) | Phase 2 details |
| [docs/RAG_ACCURACY_IMPROVEMENT_PLAN_2025.md](docs/RAG_ACCURACY_IMPROVEMENT_PLAN_2025.md) | Future optimizations |
