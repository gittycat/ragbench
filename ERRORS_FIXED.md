# Error Fixes - 2026-02-04

## Critical Errors Fixed

### 1. pgmq Message Deserialization Error
**Error:** `'datetime.datetime' object has no attribute 'get'`

**Root Cause:** PostgreSQL pgmq extension v1.9.0 returns 7 columns (`msg_id`, `read_ct`, `enqueued_at`, `last_read_at`, `vt`, `message`, `headers`), but the Python pgmq library v1.0.3 expects only 5 columns.

**Fix:** Updated `infrastructure/tasks/pgmq_queue.py` `read_message()` function to correctly map columns:
```python
# Handle 7-column structure instead of 5
return Message(
    msg_id=row[0],
    read_ct=row[1],
    enqueued_at=row[2],
    vt=row[4],  # Skip last_read_at at index 3
    message=row[5]  # Message is at index 5, not 4
)
```

### 2. Session Creation AsyncIO Event Loop Error
**Error:** `Task got Future attached to a different loop`

**Root Cause:** FastAPI async endpoint `create_new_session()` was calling sync `create_session_metadata()` which internally used `_run_async()` with ThreadPoolExecutor to run `asyncio.run()`, creating nested event loops.

**Fix:** Updated `api/routes/sessions.py` to call `create_session_async()` directly instead of using the sync wrapper:
```python
async def create_new_session(request: CreateSessionRequest):
    # Call async function directly instead of sync wrapper
    await create_session_async(session_id, title, llm_model, search_type)
```

### 3. Database Schema - Status Column Too Short
**Error:** `value too long for type character varying(20)` for status 'completed_with_errors'

**Fix:** Increased `job_batches.status` column length from VARCHAR(20) to VARCHAR(30):
```sql
ALTER TABLE job_batches ALTER COLUMN status TYPE VARCHAR(30);
```

**Permanent Fix Needed:** Update `services/postgres/init.sql` to use VARCHAR(30)

### 4. Worker AsyncIO Event Loop Conflicts
**Error:** `RuntimeError: Task got Future attached to a different loop` in pgmq worker

**Root Cause:** Complex mixing of:
- pgmq library (sync, uses psycopg2)
- LlamaIndex PGVectorStore (async, uses asyncpg via SQLAlchemy)
- Job tracking (async, uses asyncpg via SQLAlchemy)

**Partial Fix:** Made entire worker async:
- Updated `infrastructure/tasks/pgmq_worker.py` to use `async def run_worker_async()`
- Updated `infrastructure/tasks/worker.py` to use `async def process_document_async()`

**Remaining Issue:** LlamaIndex's PGVectorStore singleton creates async connections that persist across event loop boundaries. This causes intermittent "attached to a different loop" errors.

**Workaround Options:**
1. Reset vector store singleton between messages (clears connection pool)
2. Switch to sync psycopg2 instead of asyncpg for LlamaIndex (breaks async patterns)
3. Run worker in separate process with dedicated event loop (deployment complexity)

## Files Modified

- `services/rag_server/infrastructure/tasks/pgmq_queue.py` - Fixed message column mapping
- `services/rag_server/api/routes/sessions.py` - Direct async function calls
- `services/rag_server/services/session.py` - Improved `_run_async()` error handling
- `services/rag_server/infrastructure/tasks/worker.py` - Made fully async
- `services/rag_server/infrastructure/tasks/pgmq_worker.py` - Made fully async
- `services/rag_server/infrastructure/database/postgres.py` - Added `close_all_connections()`

## Testing

After fixes, test with:
```bash
docker compose down -v
docker compose up -d --build
# Upload documents and verify processing
```
