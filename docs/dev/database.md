# Database & Connection Pooling

## Connection Pooling

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

## Database Access Pattern

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
