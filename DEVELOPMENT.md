# Development Guide

This document covers the RAG server architecture, supporting services, APIs, evals, configuration, testing, and deployment. It targets both human devs and AI agents. It is brief and point like by design. An understanding of RAG and of the tech stacks used is assumed.

For frontend/UI documentation, see [FRONT_END.md](FRONT_END.md).

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Service Architecture](#service-architecture)
- [Configuration](#configuration)
- [Secrets](#secrets)
- [Evaluation Framework](#evaluation-framework)
- [API Reference](#api-reference)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Deployment](#deployment)
- [Observability](#observability)
- [Troubleshooting](#troubleshooting)
- [PII Masking](#pii-masking)
- [Roadmap](#roadmap)

## Overview

The project implements a Local RAG (Retrieval-Augmented Generation) system for querying document collections using a LLM.

The differentiating features of this RAG are:

### a) Data Privacy & Deployment Flexibility

**Fully On-Premises Option**:
- Complete open-source stack: Ollama (LLM + embeddings), PostgreSQL (pg_textsearch for BM25)
- No external network calls required - runs entirely within your infrastructure
- Ideal for sensitive documents requiring air-gapped deployment

**Cloud-Optimized Option**:
- Multi-provider LLM support: OpenAI, Anthropic, Google Gemini, DeepSeek, etc
- Leverage frontier models for maximum performance
- PII masking for cloud providers planned (see [PII Masking](#pii-masking) and [implementation plan](docs/PII_MASKING_IMPLEMENTATION_PLAN.md))

**Hybrid Deployment**:
- Mix and match: local embeddings with cloud LLM, or vice versa
- Configure per-component via `config.yml`

### b) Extensive Evaluation Framework

**Systematic RAG Evaluation**:
- DeepEval framework with LLM-as-judge (Claude Sonnet 4)
- Five metric categories: Contextual Precision, Contextual Recall, Faithfulness, Answer Relevancy, Hallucination
- Golden dataset testing with customizable Q&A pairs

**Admin Decision Support**:
- Evaluation history and trend tracking
- Baseline comparison (compare current config to golden baseline)
- Configuration recommendation API (optimize for accuracy vs cost vs latency)
- CLI and pytest integration for CI/CD workflows

**Metrics API**:
- Real-time system health monitoring
- Per-run detailed metrics
- Compare evaluation runs side-by-side
- Designed for frontend visualization (integration planned)

### c) Advanced RAG Capabilities

**Implemented**:
- **Hybrid Search**: Combines BM25 (sparse) + vector (dense) with Reciprocal Rank Fusion (~48% retrieval improvement)
- **Contextual Retrieval**: LLM-generated chunk context before embedding (~49% fewer retrieval failures)
- **Reranking**: Cross-encoder reranking for relevance optimization
- **Conversational Memory**: PostgreSQL-backed session management with chat history
- **Multi-Format Documents**: PDF, DOCX, PPTX, XLSX, Markdown, HTML, AsciiDoc via Docling parser
- **Async Processing**: Background document ingestion with progress tracking
- **Deduplication**: SHA-256 hash-based duplicate detection

**Planned** (see [Roadmap](#roadmap)):
- Parent document retrieval (sentence window method)
- Query fusion (multi-query generation)
- Multi-modal support (images, video, voice in prompts)
- Additional file formats (CSV, JSON)

## Architecture

The system is composed of multiple services running in a Docker Compose managed environment. Services communicate over a private Docker network, with select services exposed to the host for user access. Docker Compose handles service orchestration, dependency management, volume mounts, and network isolation.

### Services

**Backend Service**:
- **rag_server** (Python 3.13 + FastAPI): Main API service handling document processing and RAG queries
  - Docling for document parsing
  - LlamaIndex for RAG pipeline
  - Hybrid Search (BM25 + Vector with RRF fusion)
  - Reranking with cross-encoder
  - Contextual Retrieval (optional)

**Support Services**:
- **postgres** (PostgreSQL 17 + pg_textsearch): BM25 search and persistence
- **chromadb**: Vector database for embeddings
- **task-worker** (Python 3.13): Background task processor for async document ingestion via SKIP LOCKED
  - Shares codebase with rag_server (same Docker image, different entrypoint)

**Frontend Service**:
- **webapp** (Typescript + SvelteKit): User interface for document upload, chat, and session management
  - Proxies API requests to rag_server
  - Exposed on port 8000

### External Services

**Required**:
- **Ollama** (runs on host machine): Local LLM inference and embedding generation
  - Default models: gemma3:4b (LLM), nomic-embed-text (embeddings)
  - Accessed via `host.docker.internal:11434`
  - Can be replaced with cloud providers (OpenAI, Anthropic, Google, DeepSeek, Moonshot)

### Network Isolation

- **Public network**: webapp, rag-server (accessible from host)
- **Private network**: postgres, chromadb, task-worker (internal only)
- **Shared volumes**: Document staging (`/tmp/shared`), PostgreSQL data, model cache

### Data Flow

**Document Upload:**
1. Client uploads files → Webapp proxies to RAG Server
2. RAG Server saves to shared volume, creates task in job_tasks table
3. Task Worker claims via SKIP LOCKED: Docling parsing → chunking → embeddings → ChromaDB (vectors) + PostgreSQL (text + BM25)
4. BM25 index refreshes automatically
5. Client polls progress via batch_id

**Query (Inference):**
1. Query → Hybrid retrieval (BM25 + Vector + RRF fusion)
2. Reranking → Top-N selection
3. LLM generates answer with context
4. Response with sources returned (streaming optional)

### Connection Pooling

PostgreSQL connections are managed through multiple independent pools. All pool settings are configured in `config.yml` under the `database:` section and applied uniformly across pools on startup.

**Connection Paths:**

| # | Pool | Driver | Used By | Config Source |
|---|------|--------|---------|---------------|
| 1 | Main async engine (`postgres.py`) | asyncpg (SQLAlchemy) | All repository operations: documents, jobs, sessions, BM25 retriever, task queue | `config.yml` |

**Connection Budget (per service, with defaults):**

| Pool | pool_size | max_overflow | Max Connections |
|------|-----------|--------------|-----------------|
| Main engine | 10 | 20 | 30 |
| **Total per service** | | | **30** |

With 2 services (rag-server + task-worker), worst case = ~60 connections. PostgreSQL `max_connections` is set to 200 (via `docker-compose.yml` command) to accommodate this plus PostgreSQL's internal connections (autovacuum, WAL writer, etc.).

**Note:** Vector embeddings are now stored in ChromaDB, not PostgreSQL, which significantly reduces the connection count.

**Configuration (`config.yml`):**

```yaml
database:
  max_connections: 200   # PostgreSQL server-side limit
  pool_size: 10          # Persistent connections per pool
  max_overflow: 20       # Burst connections per pool
  pool_pre_ping: true    # Validate connections before use
  pool_recycle: 3600     # Recycle after 1 hour
```

**Key Files:**
- `infrastructure/database/postgres.py` — Main async engine pool
- `infrastructure/search/vector_store.py` — ChromaDB vector store (no PostgreSQL connections)
- `infrastructure/database/migrations/env.py` — Alembic (uses NullPool, short-lived)

### Database Access Pattern

The data access layer uses **module-level async functions with explicit session passing** instead of a full ORM repository pattern.

**Why not a full ORM (Repository/Unit-of-Work)?**
Repository classes with inheritance (e.g. `BaseRepository[T]` → `DocumentRepository`) add indirection that hurts readability. An AI agent tracing a database operation has to jump through class hierarchies, understand generic type parameters, and track implicit session state across methods. The abstraction saves a few lines of CRUD boilerplate but costs clarity at every call site.

**The pattern:**

```python
# infrastructure/database/documents.py — flat functions, explicit session
async def get_document(session: AsyncSession, document_id: UUID) -> Document | None:
    result = await session.execute(select(Document).where(Document.id == document_id))
    return result.scalar_one_or_none()

async def delete_document(session: AsyncSession, document_id: UUID) -> bool:
    result = await session.execute(delete(Document).where(Document.id == document_id))
    return result.rowcount > 0
```

```python
# Caller controls the transaction boundary — always visible
async with get_session() as session:
    doc = await db_docs.get_document(session, doc_id)
    await db_docs.delete_document(session, doc_id)
```

**Design rules:**
- Every DB function takes `session: AsyncSession` as its first parameter
- The caller owns the transaction scope via `async with get_session()`
- Multiple operations naturally share a session (no composite functions needed)
- ORM models (`models.py`) define schema and types but stay out of query logic
- `select()` for simple queries, `text()` for complex SQL (BM25 search, cross-table joins)
- Functions are directly testable by passing a mock session

**File layout:**
```
infrastructure/database/
├── postgres.py      # Connection pool, get_session(), async engine
├── models.py        # SQLAlchemy ORM models (schema + Alembic migrations)
├── documents.py     # Document and chunk DB operations
├── sessions.py      # Chat session and message DB operations + PostgresChatStore
├── jobs.py          # Job batch and task progress DB operations
```

## Technology Stack

### Backend Services

| Component | Technology | Version |
|-----------|------------|---------|
| API Framework | FastAPI | 0.118+ |
| Python | Python | 3.13+ |
| Package Manager | uv | Latest |
| Vector Database | ChromaDB | Latest |
| Full-text Search | pg_textsearch (Timescale) | 0.5+ |
| Database | PostgreSQL | 17+ |
| Task Queue | PostgreSQL SKIP LOCKED | — |
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

Provider selection via `config.yml`.

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
- `pipelines/ingestion.py`: Document processing (parsing, chunking, embedding, indexing)
- `pipelines/inference.py`: Query processing (retrieval, reranking, generation)
- `infrastructure/llm/`: Multi-provider LLM client factory
- `infrastructure/database/`: PostgreSQL + pg_textsearch (BM25)
- `infrastructure/search/`: ChromaDB vector store + BM25 retriever + hybrid RRF
- `infrastructure/tasks/`: Task worker (SKIP LOCKED)
- `infrastructure/config/`: YAML configuration loading

### PGMQ Worker

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

- `postgres_data`: PostgreSQL persistence (documents, embeddings, sessions, queues)
- `docs_repo`: Shared file upload staging
- `huggingface_cache`: Reranker model cache
- `documents_data`: Original document storage for downloads

## Core Concepts

### Document Processing

**Supported Formats**: `.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.htm`, `.asciidoc`, `.adoc`

**Pipeline**:
1. Validation and deduplication (SHA-256 hashing)
2. Parsing (Docling for complex formats, SimpleDirectoryReader for text)
3. Chunking (500 tokens, 50 overlap)
4. Optional contextual enhancement (LLM-generated chunk context)
5. Embedding generation
6. Indexing (ChromaDB vectors + PostgreSQL pg_textsearch BM25)

### Retrieval Strategy

**Hybrid Search**: Combines sparse (BM25) and dense (vector) retrieval using Reciprocal Rank Fusion (RRF). Improves retrieval quality by ~48% over single-method approaches.

**Reranking**: Cross-encoder (ms-marco-MiniLM-L-6-v2) reranks results for better relevance. Returns top 5 chunks by default.

**Contextual Retrieval**: Optional feature where LLM generates document context for each chunk before embedding. Reduces retrieval failures by ~49% with no query-time overhead. Adds ~85% to indexing time.

### Chat Features

**Session Management**: PostgreSQL-backed conversation history with persistent storage (no TTL).

**Chat Mode**: Uses `condense_plus_context` - condenses conversation history into standalone query, then retrieves context.

### Document Metadata

Each chunk stored with:
- `document_id`, `file_name`, `file_type`, `file_size_bytes`
- `file_hash` (SHA-256 for deduplication)
- `path`, `chunk_index`, `uploaded_at` (ISO 8601)

## Configuration

### YAML Configuration (`config.yml`)

Primary configuration file for models and retrieval settings.

**Setup:** Simply edit the existing file.

### Environment Variables (docker-compose.yml)

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_HOST` | `postgres` | PostgreSQL host |
| `DATABASE_PORT` | `5432` | PostgreSQL port |
| `DATABASE_NAME` | `ragbench` | PostgreSQL database |
| `LOG_LEVEL` | `WARNING` | Logging verbosity |
| `MAX_UPLOAD_SIZE` | `80` | Max upload size in MB |

## Secrets

Main points:
- API keys are provided via Docker Compose secrets mounted as files under `/run/secrets`.
- Each service reads secrets independently at startup (for example `rag-server`, `task-worker`, and `evals`).
- Secret files contain only the raw value (no `KEY=VALUE` format).
- Secrets are loaded via Pydantic Settings (file-based secrets) and kept in memory; do not log secret values.
- Avoid environment variables for API keys; use the mounted secret files instead.
- PostgreSQL credentials are provided via secrets: superuser (`POSTGRES_SUPERUSER`/`POSTGRES_SUPERPASSWORD`) and per-service client users (`RAG_SERVER_DB_USER`/`RAG_SERVER_DB_PASSWORD`).

References:
- `docker-compose.yml` (secrets definitions and mounts)
- `services/rag_server/app/settings.py`
- `services/evals/infrastructure/settings.py`

## Evaluation Framework

### Why Evaluate RAG Systems

RAG systems combine two failure-prone components: retrieval (finding relevant context) and generation (producing accurate answers). Evaluation ensures:
- **Retrieval Quality**: Are we finding the right chunks?
- **Answer Quality**: Is the generated response accurate and relevant?
- **Safety**: Are we hallucinating or producing unsupported claims?

Without systematic evaluation, configuration changes (chunk size, top-k, reranking) are guesswork.

### Evaluation Approach

**Test Dataset**: Golden Q&A pairs (question, expected answer, ground truth context)
- Current: 10 pairs from Paul Graham essays
- Target: 100+ pairs for production confidence
- Location: `evals/data/golden_qa.json`

**Evaluation Types**:
- **Retrieval Metrics**: Measure if correct chunks are retrieved
- **Generation Metrics**: Measure answer accuracy and relevance
- **Safety Metrics**: Detect hallucinations and unsupported claims

**Public Datasets**: Five additional datasets available for comprehensive evaluation (retrieval, generation, citation, abstention). See [docs/RAG_EVALUATION_DATASETS.md](docs/RAG_EVALUATION_DATASETS.md).

### Framework: DeepEval

**Why DeepEval**: Migrated from RAGAS (2025-12-07) for better CI/CD integration and pytest compatibility.

**LLM Judge**: Claude Sonnet 4 (Anthropic) - evaluates retrieval relevance, answer faithfulness, and hallucination detection.

**Integration**:
- Pytest integration with custom markers (`@pytest.mark.eval`)
- CLI tool for standalone evaluation
- CI/CD compatible (optional eval tests on demand)
- Results stored in `evals/data/runs/` for metrics API

### Metrics & Thresholds

**Retrieval Metrics**:
- **Contextual Precision** (threshold: 0.7): Are retrieved chunks relevant to the query?
- **Contextual Recall** (threshold: 0.7): Did we retrieve all information needed to answer?

**Generation Metrics**:
- **Faithfulness** (threshold: 0.7): Is the answer grounded in retrieved context?
- **Answer Relevancy** (threshold: 0.7): Does the answer address the question?

**Safety Metrics**:
- **Hallucination** (threshold: 0.5): Rate of claims not supported by context

Higher scores are better (except hallucination - lower is better).

### Running Evaluations

**Prerequisites**: `ANTHROPIC_API_KEY` environment variable

**CLI Usage**:
```bash
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

**Pytest Integration**:
```bash
# Run eval tests (requires --run-eval flag)
pytest tests/ --run-eval --eval-samples=5

# Just eval tests
just test-eval         # Quick (5 samples)
just test-eval-full    # Full dataset
```

**CI/CD**: Evaluation tests are optional (expensive, ~2-5min). Trigger via commit message containing `[eval]` or manual workflow dispatch.

### Evaluation API Endpoints

**Core Endpoints**:
- `GET /metrics/evaluation/definitions`: Metric descriptions and thresholds
- `GET /metrics/evaluation/history`: Past evaluation runs
- `GET /metrics/evaluation/summary`: Latest run with trend analysis
- `GET /metrics/evaluation/{run_id}`: Specific run details
- `DELETE /metrics/evaluation/{run_id}`: Delete evaluation run

**Baseline & Comparison**:
- `GET /metrics/baseline`: Get golden baseline run
- `POST /metrics/baseline/{run_id}`: Set run as golden baseline
- `DELETE /metrics/baseline`: Clear baseline
- `GET /metrics/compare/{run_a}/{run_b}`: Compare two runs
- `GET /metrics/compare-to-baseline/{run_id}`: Compare run to baseline

**Configuration Tuning**:
- `POST /metrics/recommend`: Get configuration recommendation based on evaluation history

### Research References

- [Evidently AI - RAG Evaluation Guide](https://www.evidentlyai.com/llm-guide/rag-evaluation)
- [Braintrust - RAG Evaluation Tools 2025](https://www.braintrust.dev/articles/best-rag-evaluation-tools)
- [Patronus AI - RAG Best Practices](https://www.patronus.ai/llm-testing/rag-evaluation-metrics)

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
cd ragbench

# Backend dependencies
cd services/rag_server
uv sync
uv sync --group dev    # Add test dependencies
uv sync --group eval   # Add evaluation dependencies

# Frontend dependencies
cd ../webapp
npm install

# Configuration
cp config.yml.example config.yml
cp secrets/.env.example secrets/.env

# Start infrastructure
docker compose up -d

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

| Category | Tests | Dependencies | Command |
|----------|-------|-------------|---------|
| **Unit** | ~32 | None (fully mocked) | `just test-unit` |
| **Integration** | 25 | Docker services (PostgreSQL, Ollama, RAG server) | `just test-integration` |
| **Integration (full)** | 25 + slow | Same + longer timeouts | `just test-integration-full` |
| **Evaluation** | 27 | ANTHROPIC_API_KEY + Docker services | `just test-eval` |

### Test Structure

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

### Integration Test Design

Integration tests verify **execution, connectivity, and structure** — not answer quality. Quality evaluation (relevance, faithfulness, hallucination) is the eval suite's job.

This separation follows industry consensus for RAG testing (Meta's [2024 RAG systems paper](https://arxiv.org/abs/2312.10997), Anthropic's [contextual retrieval guide](https://www.anthropic.com/news/contextual-retrieval), and Docker's [testing best practices](https://docs.docker.com/build/tests/)):

- **Embeddings**: "Did ChromaDB store a 768-dim vector?" — not "Are the embeddings semantically meaningful?"
- **Retrieval**: "Did a query return >0 chunks?" — not "Were they the right chunks?"
- **Reranking**: "Did the reranker produce output?" — not "Did it improve precision?"
- **LLM response**: "Did `/query` return a non-empty string?" — not "Was the answer faithful?"

One **canary test** bridges the gap: uploads a document with a unique random marker (`MARKER_<uuid>`), queries for it, and asserts the marker appears in a returned source chunk. This is a write-then-read test that catches content loss during ingestion → embedding → retrieval, without judging quality.

#### test_infrastructure.py — Service Connectivity & Database Schema (8 tests, fast)

Read-only checks that the system is wired correctly. No uploads or mutations.

**What it covers**: `/health` + `/config` + `/models/info` endpoints return expected fields. PostgreSQL has `pg_textsearch` extension. Required tables exist (`documents`, `document_chunks`, `chat_sessions`, `chat_messages`, `job_batches`, `job_tasks`). `idx_tasks_claimable` partial index exists for SKIP LOCKED. Ollama has required models (`gemma3`, `nomic-embed-text`).

**What it does NOT cover**: Whether these components work together under load, or whether the schema supports all query patterns correctly.

**Why these tests**: Infrastructure checks are the cheapest gate. If PostgreSQL extensions are missing or Ollama lacks models, every downstream test will fail with misleading errors. Running these first provides clear diagnostics.

#### test_pipeline.py — Upload → Index → Query → Delete (9 tests, slow)

Core RAG loop: upload a document, verify it's indexed, query it, delete it.

**What it covers**: Document appears in `/documents` after upload. Document has >0 chunks. `/query` returns non-empty answer with `session_id`. Query with `include_chunks=True` returns source list. Canary test: marker string survives full round-trip. PDF upload and query. Delete removes document from list. Query after delete returns 200 (not 500). Delete nonexistent returns 404.

**What it does NOT cover**: Large document handling, concurrent uploads, chunking quality, or retrieval ranking correctness.

**Why these tests**: The upload → index → query → delete cycle is the core user journey. Every component in the pipeline must work for these to pass: file upload, task creation, worker processing via SKIP LOCKED, Docling parsing, embedding generation, vector storage, BM25 indexing, hybrid retrieval, LLM generation, and document deletion with cascade cleanup.

#### test_chat_sessions.py — Session Lifecycle (5 tests, mixed speed)

Session creation, history growth, clearing, and temporary session behavior.

**What it covers**: Query creates a session with user + assistant messages in history. Second query on same session grows history. Nonexistent session returns empty (not 500). Clear empties history. Temporary sessions don't appear in session list.

**What it does NOT cover**: Session persistence across server restarts, concurrent session access, or session archiving.

**Why these tests**: Conversational RAG relies on session state. A broken session means context loss for multi-turn conversations.

#### test_async_upload.py — Async Processing & Progress (8 tests, slow)

Full async upload workflow via SKIP LOCKED task queue with progress tracking and edge cases.

**What it covers**: Upload via API → task-worker processes via SKIP LOCKED → status shows completed. Progress tracking accuracy for multi-chunk documents. Single and multiple file uploads appear in document list. Large markdown and PDF status progression. Invalid batch ID handling. Concurrent uploads without race conditions.

**What it does NOT cover**: Worker crash recovery, queue poisoning, or network partition scenarios.

**Why these tests**: Async processing is where most silent failures occur. Tasks can hang, progress can desync, and concurrent uploads can trigger race conditions in BM25 index refresh.

### Pytest Markers

```python
@pytest.mark.integration  # Requires --run-integration flag
@pytest.mark.slow         # Tests taking > 30s (skipped by default, use --run-slow)
@pytest.mark.eval         # Requires --run-eval and ANTHROPIC_API_KEY
```

### Future Testing Work

Additional testing improvements (smoke/full tiering, CI integration jobs, resilience tests) are tracked in [ROADMAP.md — Testing & CI Improvements](docs/ROADMAP.md#testing--ci-improvements).

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
- **Core Tests** (~30s): Always runs, no special requirements
- **Eval Tests** (~2-5min): Optional, requires ANTHROPIC_API_KEY
- **Docker Build** (~5-10min): Always runs, no special requirements

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

- `docker-compose.yml`: Base configuration
- `docker-compose.local.yml`: Local overrides (debug logging)
- `docker-compose.cloud.yml`: Cloud overrides (registry images)
- `docker-compose.ci.yml`: CI/CD infrastructure

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

## Observability

### Metrics API

Comprehensive visibility into system configuration and performance.

**System Endpoints**:
- `/metrics/system`: Complete system overview + health status
- `/metrics/models`: Model details with references
- `/metrics/retrieval`: Pipeline configuration

**Evaluation Endpoints**: See [Evaluation Framework](#evaluation-framework) section above for complete evaluation API documentation.

### Health Monitoring

Component health via `/metrics/system`:
- PostgreSQL: Vector store + BM25 + queue connectivity
- Ollama: LLM availability

### Key Metrics

**Retrieval**: Contextual Precision, Contextual Recall, MRR, Hit Rate

**Generation**: Faithfulness, Answer Relevancy, Hallucination Rate

**Operational**: Latency (P50, P95), Tokens per query, Cost

## Troubleshooting

### Common Issues

- **Ollama not accessible**: Check host binding with `curl http://localhost:11434/api/tags`
- **PostgreSQL connection fails**: Verify `DATABASE_URL` and `private` network connectivity
- **Docker build fails**: Ensure `--index-strategy unsafe-best-match` in Dockerfile
- **Tests fail**: Use `.venv/bin/pytest` not `uv run pytest`
- **Reranker slow first query**: Model downloads ~80MB on first use
- **BM25 not initializing**: Requires documents at startup or initializes after first upload
- **Contextual retrieval not working**: Check `enable_contextual_retrieval: true` in config

### Service Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f rag-server
docker compose logs -f task-worker
docker compose logs -f postgres
```

### Database Reset

```bash
docker compose down -v
docker compose up -d
```

### Backup & Restore

```bash
# Manual backup (PostgreSQL)
docker compose exec postgres pg_dump -U raguser ragbench > backups/ragbench.sql

# Restore
cat backups/ragbench.sql | docker compose exec -T postgres psql -U raguser -d ragbench
```

## PII Masking

Optional feature to anonymize sensitive data before sending to cloud LLM providers. Uses Microsoft Presidio for PII detection and reversible token-based masking.

**Status**: Planned (see [implementation plan](docs/PII_MASKING_IMPLEMENTATION_PLAN.md))

### How It Works

1. **Masking (outbound)**: PII detected via Microsoft Presidio (NER + regex), replaced with tokens like `[[[PERSON_0]]]`
2. **Token mapping**: Original values stored temporarily (session-scoped)
3. **Unmasking (inbound)**: Tokens in LLM response replaced with original values
4. **Validation**: Detects if LLM altered tokens, attempts fuzzy recovery
5. **Output guardrails**: Scans final response for accidentally leaked PII

### Configuration

Enable in `config.yml`:

```yaml
pii:
  enabled: true
  entities:
    - PERSON
    - EMAIL_ADDRESS
    - PHONE_NUMBER
    - CREDIT_CARD
    - US_SSN
  token_format: "[[[{entity_type}_{index}]]]"
  score_threshold: 0.5
  validation:
    enabled: true
    max_retries: 2
  output_guardrails:
    enabled: true
    block_on_detection: false
  audit:
    enabled: true
    log_level: INFO
```

### Supported Entity Types

`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, `US_SSN`, `IBAN_CODE`, `IP_ADDRESS`, `LOCATION`, `DATE_TIME`, `US_BANK_NUMBER`, `US_DRIVER_LICENSE`, `US_PASSPORT`, `MEDICAL_LICENSE`

### Audit Logging

When `audit.enabled: true`, all masking/unmasking operations are logged:

```json
{"operation": "MASK", "timestamp": "...", "context_id": "session_123", "entities_count": 3, "entity_types": ["PERSON", "EMAIL_ADDRESS"]}
{"operation": "UNMASK", "timestamp": "...", "context_id": "session_123", "tokens_found": 3, "tokens_replaced": 3, "validation_passed": true}
```

### Data Flow Points

| Path | Description |
|------|-------------|
| User queries | Query text, chat history, retrieved context sent to LLM |
| Contextual retrieval | Document chunks sent to LLM during ingestion |
| Session titles | First user message sent for title generation |
| Evaluation | Test data sent to evaluation LLM |

### Limitations

- **Token preservation**: LLMs may alter tokens (e.g., remove brackets). Validation detects this; fuzzy recovery attempts restoration
- **Performance**: Adds ~20-50ms per request for Presidio analysis
- **Not for embeddings**: Embeddings are generated from original text (stored locally in PostgreSQL)

### When to Enable

- Using cloud LLM providers (OpenAI, Anthropic, Google, etc.)
- Documents contain PII that shouldn't leave your infrastructure
- Compliance requirements (GDPR, HIPAA, etc.)

Not needed when using Ollama (local inference) as data never leaves your network.

## Roadmap

For detailed feature roadmap including implementation tasks and effort estimates, see [ROADMAP.md](docs/ROADMAP.md).

### Recently Completed

- **PostgreSQL-backed Chat Memory** (Oct 2025): Session-based conversation history with persistent storage
- **Hybrid Search** (Oct 2025): BM25 + Vector + RRF fusion with ~48% retrieval improvement
- **Contextual Retrieval** (Oct 2025): LLM-generated chunk context with ~49% fewer retrieval failures
- **DeepEval Framework** (Dec 2025): Anthropic Claude Sonnet 4 as LLM judge with pytest integration
- **Forgejo CI/CD** (Dec 2025): Self-hosted Git + CI/CD with GitHub Actions-compatible workflows
- **Metrics & Observability API** (Dec 2025): System health monitoring and evaluation history tracking

### In Planning

- **PII Masking**: Anonymize sensitive data for cloud LLM providers (see [implementation plan](docs/PII_MASKING_IMPLEMENTATION_PLAN.md))
- **Centralized Logging**: Grafana Loki + Promtail + structlog (see [implementation plan](docs/LOGGING_IMPLEMENTATION_PLAN.md))
- **Parent Document Retrieval**: Sentence window method for better context
- **Query Fusion**: Multi-query generation for improved recall
- **Metrics Visualization**: Frontend dashboards for evaluation and system health


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
| [docs/ROADMAP.md](docs/ROADMAP.md) | Feature roadmap with tasks and effort estimates |
| [docs/FORGEJO_CI_SETUP.md](docs/FORGEJO_CI_SETUP.md) | CI/CD setup guide |
| [docs/DEEPEVAL_IMPLEMENTATION_SUMMARY.md](docs/DEEPEVAL_IMPLEMENTATION_SUMMARY.md) | Evaluation framework |
| [docs/CONVERSATIONAL_RAG.md](docs/CONVERSATIONAL_RAG.md) | Session management |
| [docs/PERFORMANCE_OPTIMIZATIONS_SUMMARY.md](docs/PERFORMANCE_OPTIMIZATIONS_SUMMARY.md) | Performance tuning |
| [docs/PHASE1_IMPLEMENTATION_SUMMARY.md](docs/PHASE1_IMPLEMENTATION_SUMMARY.md) | Phase 1 details |
| [docs/PHASE2_IMPLEMENTATION_SUMMARY.md](docs/PHASE2_IMPLEMENTATION_SUMMARY.md) | Phase 2 details |
| [docs/RAG_ACCURACY_IMPROVEMENT_PLAN_2025.md](docs/RAG_ACCURACY_IMPROVEMENT_PLAN_2025.md) | Future optimizations |
| [docs/PII_MASKING_IMPLEMENTATION_PLAN.md](docs/PII_MASKING_IMPLEMENTATION_PLAN.md) | PII masking for cloud LLMs |
| [docs/LOGGING_IMPLEMENTATION_PLAN.md](docs/LOGGING_IMPLEMENTATION_PLAN.md) | Centralized logging with Grafana Loki |
