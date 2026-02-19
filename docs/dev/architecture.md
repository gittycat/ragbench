# Architecture

The system is composed of multiple services running in a Docker Compose managed environment. Services communicate over a private Docker network, with select services exposed to the host for user access. Docker Compose handles service orchestration, dependency management, volume mounts, and network isolation.

## Services

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

**Evaluation Service**:
- **evals** (Python 3.13 + FastAPI): Evaluation API and CLI for RAG quality assessment
  - Runs on port 8002, always-on (no profile gating)
  - Triggers eval runs, tracks progress, serves results
  - Heavy deps (deepeval, datasets, HuggingFace) isolated from rag-server
  - CLI still accessible via `docker compose exec evals .venv/bin/python -m evals.cli ...`

**Frontend Service**:
- **webapp** (Typescript + SvelteKit): User interface for document upload, chat, and session management
  - Proxies `/api/eval/*` to evals service, all other `/api/*` to rag-server
  - Exposed on port 8000

## External Services

**Required**:
- **Ollama** (runs on host machine): Local LLM inference and embedding generation
  - Default models: gemma3:4b (LLM), nomic-embed-text (embeddings)
  - Accessed via `host.docker.internal:11434`
  - Can be replaced with cloud providers (OpenAI, Anthropic, Google, DeepSeek, Moonshot)

## Network Isolation

- **Public network**: webapp, rag-server (accessible from host)
- **Private network**: postgres, chromadb, task-worker (internal only)
- **Shared volumes**: Document staging (`/tmp/shared`), PostgreSQL data, model cache

## Data Flow

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

## RAG Server (`services/rag_server/`)

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

## Task Worker

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

## Docker Volumes

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
