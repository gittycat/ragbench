# PostgreSQL as Unified Data Layer for RAG

## Summary

Migrate RAG system from ChromaDB + Redis to PostgreSQL 18.1 with:
- **pgvector** - Vector storage and similarity search (replaces ChromaDB)
- **pg_search (ParadeDB)** - True BM25 full-text search (replaces in-memory BM25)
- **pgmq** - Message queue (replaces RQ + Redis)
- Native tables for chat history, sessions, progress tracking (replaces Redis)

**Docker Image**: ParadeDB (includes pgvector + pg_search, pgmq installed separately)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PostgreSQL 18.1                           │
│  ┌───────────┬───────────┬───────────┬──────────┬────────┐ │
│  │ pgvector  │ pg_search │  tables   │   pgmq   │ JSONB  │ │
│  │ (vectors) │  (BM25)   │ (chunks)  │ (queue)  │(meta)  │ │
│  └───────────┴───────────┴───────────┴──────────┴────────┘ │
└─────────────────────────────────────────────────────────────┘
         ▲                              ▲
         │                              │
    ┌────┴────┐                    ┌────┴────┐
    │rag-server│                   │ worker  │
    │(queries) │                   │(ingest) │
    └──────────┘                   └─────────┘
```

**Services Eliminated**: ChromaDB, Redis, RQ

---

## Module Structure

```
services/rag_server/
├── infrastructure/
│   ├── database/
│   │   ├── postgres.py          # Connection pool (asyncpg + SQLAlchemy 2.0)
│   │   ├── models.py            # SQLAlchemy ORM models
│   │   ├── migrations/          # Alembic migrations
│   │   └── repositories/
│   │       ├── base.py          # Abstract repository
│   │       ├── documents.py     # Document + chunk operations
│   │       ├── sessions.py      # Chat sessions + messages
│   │       └── jobs.py          # Progress tracking
│   ├── search/
│   │   ├── vector_store.py      # LlamaIndex PGVectorStore wrapper
│   │   ├── bm25_retriever.py    # Custom retriever using pg_search SQL
│   │   └── hybrid_retriever.py  # RRF fusion (both retrievers)
│   └── tasks/
│       ├── pgmq_queue.py        # pgmq client wrapper
│       └── worker.py            # Worker (polling pgmq)
```

---

## Database Schema

### Core Tables

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;
CREATE EXTENSION IF NOT EXISTS pgmq;

-- Documents (source files)
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_path TEXT,
    file_size_bytes BIGINT,
    file_hash VARCHAR(64) UNIQUE,
    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Document chunks with vectors
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_with_context TEXT,  -- Contextual prefix + content
    embedding vector(768),       -- nomic-embed-text
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);

-- HNSW index for vector search
CREATE INDEX idx_chunks_embedding ON document_chunks
USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- BM25 index via pg_search
CALL paradedb.create_bm25(
    index_name => 'idx_chunks_bm25',
    table_name => 'document_chunks',
    key_field => 'id',
    text_fields => paradedb.field('content', tokenizer => paradedb.tokenizer('en_stem'))
);

-- Chat sessions
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY,
    title VARCHAR(255) DEFAULT 'New Chat',
    llm_model VARCHAR(100),
    search_type VARCHAR(20),
    is_archived BOOLEAN DEFAULT FALSE,
    is_temporary BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chat messages
CREATE TABLE chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Job batches (progress tracking)
CREATE TABLE job_batches (
    id UUID PRIMARY KEY,
    total_tasks INTEGER NOT NULL,
    completed_tasks INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE job_tasks (
    id UUID PRIMARY KEY,
    batch_id UUID NOT NULL REFERENCES job_batches(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    total_chunks INTEGER DEFAULT 0,
    completed_chunks INTEGER DEFAULT 0,
    error_message TEXT
);

-- pgmq queue
SELECT pgmq.create('documents');
```

---

## LlamaIndex Integration Strategy

### Vector Store
Use LlamaIndex's `PGVectorStore` with `hybrid_search=False` (we handle BM25 separately via pg_search):

```python
from llama_index.vector_stores.postgres import PGVectorStore

vector_store = PGVectorStore.from_params(
    database="ragbench",
    host="postgres",
    embed_dim=768,
    hybrid_search=False,  # We use pg_search for BM25
    hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 100}
)
```

### BM25 Retriever (Custom)
Build custom LlamaIndex retriever wrapping pg_search SQL:

```python
class PgSearchBM25Retriever(BaseRetriever):
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        sql = """
            SELECT id, content, paradedb.score(id) as score
            FROM document_chunks
            WHERE content @@@ :query
            ORDER BY score DESC LIMIT :limit
        """
        # Execute and convert to NodeWithScore
```

### Hybrid RRF Fusion
Combine both retrievers with Reciprocal Rank Fusion:

```python
class HybridRRFRetriever(BaseRetriever):
    def __init__(self, bm25_retriever, vector_retriever, rrf_k=60):
        # RRF: score = sum(1 / (k + rank))
```

### Chat Store (Custom)
Replace `RedisChatStore` with PostgreSQL-backed implementation:

```python
class PostgresChatStore:
    async def get_messages(self, session_id: str) -> List[ChatMessage]:
        # SELECT from chat_messages

    async def add_message(self, session_id: str, message: ChatMessage):
        # INSERT into chat_messages
```

---

## Docker Compose Changes

```yaml
services:
  postgres:
    image: paradedb/paradedb:latest
    environment:
      POSTGRES_USER: raguser
      POSTGRES_PASSWORD: ragpass
      POSTGRES_DB: ragbench
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./services/postgres/init.sql:/docker-entrypoint-initdb.d/01-init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U raguser -d ragbench"]
    networks:
      - private

  pgmq-worker:
    build: ./services/rag_server
    command: [".venv/bin/python", "-m", "infrastructure.tasks.pgmq_worker"]
    environment:
      - DATABASE_URL=postgresql+asyncpg://raguser:ragpass@postgres:5432/ragbench
    depends_on:
      postgres: {condition: service_healthy}

  rag-server:
    environment:
      - DATABASE_URL=postgresql+asyncpg://raguser:ragpass@postgres:5432/ragbench
    depends_on:
      postgres: {condition: service_healthy}

# REMOVE: chromadb, redis services
# REMOVE: chromadb_data, redis_data volumes
```

---

## Dependencies

Add to `pyproject.toml`:
```toml
asyncpg = "^0.30.0"
sqlalchemy = {extras = ["asyncio"], version = "^2.0.36"}
pgvector = "^0.3.6"
alembic = "^1.14.0"
tembo-pgmq-python = "^1.2.0"
llama-index-vector-stores-postgres = "^0.3.4"
```

Remove:
```toml
chromadb
redis
rq
llama-index-vector-stores-chroma
llama-index-storage-chat-store-redis
```

---

## Implementation Phases

### Phase 1: Foundation (Infrastructure)
| File | Action |
|------|--------|
| `infrastructure/database/postgres.py` | Create async connection pool |
| `infrastructure/database/models.py` | Define SQLAlchemy models |
| `infrastructure/database/migrations/` | Set up Alembic |
| `docker-compose.yml` | Add PostgreSQL, keep old services temporarily |
| `pyproject.toml` | Add new dependencies |

### Phase 2: Document Storage
| File | Action |
|------|--------|
| `infrastructure/database/repositories/documents.py` | Create document repository |
| `infrastructure/search/vector_store.py` | Create PGVectorStore wrapper |
| `infrastructure/database/chroma.py` | Delete (after migration) |
| `pipelines/ingestion.py` | Update to use PostgreSQL |

### Phase 3: Hybrid Search
| File | Action |
|------|--------|
| `infrastructure/search/bm25_retriever.py` | Create pg_search retriever |
| `infrastructure/search/hybrid_retriever.py` | Create RRF fusion retriever |
| `pipelines/inference.py` | Replace BM25Retriever, remove ChromaDB imports |

### Phase 4: Chat & Sessions
| File | Action |
|------|--------|
| `infrastructure/database/repositories/sessions.py` | Create session repository + chat store |
| `services/session.py` | Replace Redis with PostgreSQL |
| `pipelines/inference.py` | Replace RedisChatStore |

### Phase 5: Job Queue
| File | Action |
|------|--------|
| `infrastructure/tasks/pgmq_queue.py` | Create pgmq wrapper |
| `infrastructure/database/repositories/jobs.py` | Create job/batch repository |
| `infrastructure/tasks/worker.py` | Update for pgmq polling |
| `infrastructure/tasks/rq_queue.py` | Delete |
| `infrastructure/tasks/progress.py` | Delete |

### Phase 6: Cleanup
| Action |
|--------|
| Remove chromadb, redis services from docker-compose.yml |
| Remove old dependencies from pyproject.toml |
| Update tests for PostgreSQL |
| Update CLAUDE.md documentation |

---

## Files to Modify/Delete

### Delete
- `infrastructure/database/chroma.py`
- `infrastructure/tasks/rq_queue.py`
- `infrastructure/tasks/progress.py`

### Major Changes
- `pipelines/inference.py` - Replace BM25Retriever, RedisChatStore
- `pipelines/ingestion.py` - Use PostgreSQL repository
- `services/session.py` - Replace Redis with PostgreSQL
- `infrastructure/tasks/worker.py` - Switch from RQ to pgmq
- `docker-compose.yml` - Remove ChromaDB/Redis, add PostgreSQL
- `pyproject.toml` - Update dependencies

### New Files
- `infrastructure/database/postgres.py`
- `infrastructure/database/models.py`
- `infrastructure/database/repositories/base.py`
- `infrastructure/database/repositories/documents.py`
- `infrastructure/database/repositories/sessions.py`
- `infrastructure/database/repositories/jobs.py`
- `infrastructure/search/vector_store.py`
- `infrastructure/search/bm25_retriever.py`
- `infrastructure/search/hybrid_retriever.py`
- `infrastructure/tasks/pgmq_queue.py`
- `services/postgres/init.sql`

---

## Verification

### 1. Database Setup
```bash
docker compose up postgres -d
docker compose exec postgres psql -U raguser -d ragbench -c "\dx"
# Verify: vector, pg_search, pgmq extensions installed
```

### 2. Document Ingestion
```bash
just test-integration  # After updating tests
# Or manually:
curl -X POST http://localhost:8001/upload -F "files=@test.pdf"
curl http://localhost:8001/documents  # Verify document appears
```

### 3. Hybrid Search
```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "session_id": "test-session"}'
# Verify: sources returned, hybrid search logs show BM25 + vector
```

### 4. Chat Persistence
```bash
# Query twice with same session_id
curl -X POST http://localhost:8001/query -d '{"query": "q1", "session_id": "sess1"}'
curl -X POST http://localhost:8001/query -d '{"query": "q2", "session_id": "sess1"}'
# Restart rag-server
docker compose restart rag-server
# Query again - chat history should persist
curl -X GET http://localhost:8001/chat/history/sess1
```

### 5. Async Upload
```bash
curl -X POST http://localhost:8001/upload -F "files=@large.pdf"
# Returns batch_id
curl http://localhost:8001/tasks/{batch_id}/status
# Poll until completed
```

### 6. Run Tests
```bash
just test-unit
just test-integration --run-integration
```

---

## Research Sources

- [pgvector GitHub](https://github.com/pgvector/pgvector) - ~14K stars
- [ParadeDB pg_search](https://www.paradedb.com/blog/introducing-search) - True BM25 in PostgreSQL
- [ParadeDB Hybrid Search Manual](https://www.paradedb.com/blog/hybrid-search-in-postgresql-the-missing-manual) - RRF implementation
- [pgmq Documentation](https://legacy.tembo.io/blog/pgmq-with-python/) - PostgreSQL message queue
- [LlamaIndex PGVectorStore](https://docs.llamaindex.ai/en/stable/examples/vector_stores/postgres/) - Native integration
- [pgvector vs ChromaDB benchmarks](https://www.myscale.com/blog/pgvector-vs-chroma-performance-analysis-vector-databases/) - Performance comparison
