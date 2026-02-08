# Post-Mortem: Async Event Loop Contamination

**Date:** 2026-02-08
**Severity:** High (intermittent streaming query failures, connection pool corruption)
**Symptom:** `RuntimeError: Task got Future attached to a different loop`

## Table of Contents

- [Introduction: Async in This Codebase](#introduction-async-in-this-codebase)
- [The Problem](#the-problem)
- [Original Design and Rationale](#original-design-and-rationale)
- [Root Cause Analysis](#root-cause-analysis)
- [The Fix](#the-fix)
- [Rules for Future Development](#rules-for-future-development)
- [References](#references)

---

## Introduction: Async in This Codebase

This project runs on FastAPI (async) with PostgreSQL via SQLAlchemy 2.0's async engine (`asyncpg` driver). The codebase has three layers with different async characteristics, and the friction between them is what caused this incident.

### Layer 1: FastAPI (async-native)

FastAPI route handlers are `async def` functions running on uvicorn's main event loop. Database operations, session management, and HTTP responses all happen on this single loop. This is the "happy path" — everything awaits on the same loop, no thread boundaries crossed.

```
Client Request → uvicorn main loop → async route handler → await db_query() → response
```

### Layer 2: LlamaIndex (sync-only)

LlamaIndex's core chat engine, retriever, and chat store interfaces are synchronous. `CondensePlusContextChatEngine.chat()` and `.stream_chat()` are blocking calls. `BaseChatStore.get_messages()` and `.set_messages()` are sync methods that subclasses must implement. This is the framework constraint that created the bridging problem.

```
chat_engine.chat(query)          # sync, blocks the calling thread
chat_store.get_messages(key)     # sync interface, must return List[ChatMessage]
retriever._retrieve(query)       # sync interface, must return List[NodeWithScore]
```

### Layer 3: PostgreSQL via asyncpg (async-only)

All database access goes through SQLAlchemy's async engine backed by `asyncpg`. The connection pool (`AsyncAdaptedQueuePool`) manages persistent connections. These connections are **bound to the event loop that created them** — this is an asyncpg invariant, not a SQLAlchemy choice.

```
async with get_session() as session:
    result = await session.execute(query)    # must be awaited on the creating loop
```

### The Bridging Problem

The conflict: LlamaIndex calls sync methods → those methods need to do async database I/O → but you can't `await` from sync code.

Three components faced this exact problem:

| Component | Sync Caller | Async Operation |
|-----------|-------------|-----------------|
| `PostgresChatStore` | LlamaIndex `BaseChatStore` interface | Read/write chat messages to PostgreSQL |
| `PgSearchBM25Retriever` | LlamaIndex `BaseRetriever._retrieve()` | Execute BM25 search queries via pg_search |
| `session.py` service | `inference.py` sync pipeline functions | Read/write session metadata to PostgreSQL |

Each of these implemented its own sync-to-async bridge. All three got it wrong in the same way.

---

## The Problem

### Error Messages

```
sqlalchemy.pool.impl.AsyncAdaptedQueuePool - ERROR -
  Exception terminating connection <AdaptedConnection <asyncpg.connection.Connection>>
RuntimeError: Task <Task pending> got Future <Future pending> attached to a different loop
```

```
pipelines.inference - ERROR - [QUERY_STREAM] Error during streaming:
  Task <Task pending coro=<PostgresChatStore._async_get_messages()>>
  got Future <Future pending> attached to a different loop
```

### Impact

- Streaming queries (`POST /query/stream`) failed intermittently during SSE token generation.
- The connection pool became corrupted: connections created on temporary event loops were returned to the shared pool, poisoning it for all subsequent operations on the main loop.
- Connection termination errors cascaded as the pool tried to close connections bound to now-dead loops.

---

## Original Design and Rationale

### What Was Done

Three independent sync-to-async bridges were implemented, all following the same pattern: **create a temporary event loop in a thread pool worker**.

**`PostgresChatStore` in `sessions.py`:**

```python
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="chat_store")

def _run_async_in_thread(async_func, *args, **kwargs):
    def _run_in_new_loop():
        loop = asyncio.new_event_loop()         # NEW LOOP per call
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_func(*args, **kwargs))
        finally:
            loop.close()

    try:
        asyncio.get_running_loop()
        future = _executor.submit(_run_in_new_loop)    # thread + new loop
        return future.result()
    except RuntimeError:
        return asyncio.run(async_func(*args, **kwargs)) # also creates a new loop
```

**`PgSearchBM25Retriever` in `bm25_retriever.py`:**

```python
def _retrieve(self, query_bundle):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, self._aretrieve(query_bundle))
            return future.result()                      # thread + new loop
    else:
        return asyncio.run(self._aretrieve(query_bundle))
```

**`session.py` service:**

```python
def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)     # thread + new loop
            return future.result(timeout=30)
    except RuntimeError:
        return asyncio.run(coro)
```

### Why It Was Done This Way

The reasoning was sound on the surface:

1. **You can't `await` from sync code.** LlamaIndex's `BaseChatStore` and `BaseRetriever` interfaces are sync-only. The implementor has no choice but to bridge.

2. **You can't call `asyncio.run()` inside a running loop.** FastAPI's route handlers run on uvicorn's event loop. Calling `asyncio.run()` from sync code invoked by an async handler raises `RuntimeError: This event loop is already running`. The thread pool approach sidesteps this by running the new loop in a separate thread where no loop exists.

3. **`nest_asyncio` was avoided.** Monkey-patching the event loop with `nest_asyncio` is a common workaround but is widely discouraged for production use — it breaks asyncio's reentrancy guarantees and can mask deadlocks.

4. **The SQLAlchemy connection pool appeared thread-safe.** SQLAlchemy's `QueuePool` is indeed thread-safe for connection checkout/checkin. The code comment even stated: *"The database engine connection pool is thread-safe and works across event loops since SQLAlchemy manages this properly."* This is true for the pool's locking mechanism — but false for the connections themselves.

### The Incorrect Assumption

The critical mistake was treating "thread-safe pool" as equivalent to "loop-portable connections". SQLAlchemy's pool correctly handles concurrent access from multiple threads. But the connections inside the pool (asyncpg protocol objects) carry internal state — pending futures, protocol buffers, waiter callbacks — that are bound to the event loop that created them. This is not a SQLAlchemy limitation; it is an asyncpg (and by extension, Python `asyncio`) invariant.

---

## Root Cause Analysis

### The Contamination Cycle

```
                    Main Event Loop (uvicorn)
                    ┌─────────────────────────────┐
                    │  Connection Pool             │
                    │  ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐  │
                    │  │C1│ │C2│ │C3│ │C4│ │C5│  │
                    │  └──┘ └──┘ └──┘ └──┘ └──┘  │
                    └─────────────────────────────┘
                              ▲          │
                    checkout  │          │ checkin
                              │          ▼
Thread Pool Worker ──────────────────────────────────
    loop_B = asyncio.new_event_loop()
    loop_B.run_until_complete(db_query())
        → pool checks out C2 (created by loop_A)
        → asyncpg tries to use C2 on loop_B
        → RuntimeError: Future attached to a different loop
```

Step by step:

1. **Startup**: The main event loop (loop A) initializes the connection pool via `init_db()`. The pool creates connections bound to loop A.

2. **Request arrives**: FastAPI handler calls `query_rag()` (sync). LlamaIndex calls `chat_store.get_messages()` (sync). The bridge detects a running loop, submits to `ThreadPoolExecutor`.

3. **Thread creates loop B**: `asyncio.new_event_loop()` creates a fresh loop in the thread. The bridge runs `loop_B.run_until_complete(async_db_operation())`.

4. **Pool checkout on wrong loop**: The async DB operation calls `get_session()`, which checks out a connection from the shared pool. If the connection was created by loop A, asyncpg's internal protocol tries to attach futures to loop B — and raises `RuntimeError`.

5. **Or: new connection on wrong loop**: If the pool creates a new connection, it is bound to loop B. After the operation completes, this connection is returned to the shared pool. Later, when loop A (or another temporary loop) checks out this connection, the same error occurs.

6. **Pool corruption spreads**: Each request has a chance of creating connections on a different loop. Over time, the pool becomes a mix of connections bound to various dead loops. Connection termination errors cascade during pool maintenance.

### Why Streaming Was Hit Harder

The non-streaming path (`POST /query`) blocks the main event loop thread directly. While blocking, no other coroutines run on the main loop, reducing the chance of cross-loop checkout. The temporary loop creates connections, uses them, and returns them — but the main loop doesn't try to use them simultaneously.

The streaming path (`POST /query/stream`) runs the sync generator in Starlette's thread pool (standard behavior for sync generators in `StreamingResponse`). The main loop continues serving other requests concurrently. When the generator's thread creates a temporary loop for DB operations, the main loop may simultaneously be using connections from the same pool — maximizing the chance of cross-loop collision.

---

## The Fix

### Principle

**All asyncpg operations must execute on the same event loop.** Instead of creating temporary loops, sync-to-async bridges schedule coroutines on the main event loop using `asyncio.run_coroutine_threadsafe()`.

### Changes

#### 1. Central bridge function (`postgres.py`)

```python
_main_loop: asyncio.AbstractEventLoop | None = None

def set_main_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _main_loop
    _main_loop = loop

def run_async_safely(coro):
    """
    Run async coroutine from sync context on the main event loop.
    Must be called from a thread OTHER than the main loop's thread.
    """
    if _main_loop is not None and _main_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, _main_loop)
        return future.result(timeout=60)
    return asyncio.run(coro)   # fallback for testing / startup
```

`run_coroutine_threadsafe` submits the coroutine to the main loop's task queue from any thread. The calling thread blocks on `future.result()` until the main loop executes the coroutine and returns the result. All connections are created and used on the main loop — no cross-loop contamination.

#### 2. Store the main loop at startup (`main.py`)

```python
async def startup():
    set_main_event_loop(asyncio.get_running_loop())
    # ... rest of startup
```

#### 3. Replace all three bridges

| File | Before | After |
|------|--------|-------|
| `sessions.py` | `_run_async_in_thread()` with `ThreadPoolExecutor` + `new_event_loop()` | `run_async_safely(coro)` |
| `session.py` | `_run_async()` with `ThreadPoolExecutor` + `asyncio.run()` | `run_async_safely(coro)` |
| `bm25_retriever.py` | `ThreadPoolExecutor` + `asyncio.run()` | `run_async_safely(coro)` |

#### 4. Prevent main loop blocking in route handlers

`run_async_safely` requires the main loop to be **free** to execute the scheduled coroutine. If a sync call from an `async def` handler blocks the main loop thread, scheduling work on that same loop deadlocks.

Fix: wrap blocking sync calls in `run_in_executor` so they run in a thread, keeping the main loop free.

```python
# query.py — before (blocks main loop)
result = query_rag(request.query, ...)

# query.py — after (main loop stays free)
loop = asyncio.get_running_loop()
result = await loop.run_in_executor(None, partial(query_rag, request.query, ...))
```

Applied to all sync calls from async handlers:
- `query.py`: `query_rag()`
- `chat.py`: `get_chat_history()`, `clear_session_memory()`
- `sessions.py`: `delete_session()`

Note: `query_rag_stream()` (sync generator) already runs in a thread via Starlette's `iterate_in_threadpool` inside `StreamingResponse` — no change needed.

### Call Flow After Fix

```
Async Handler (main loop thread)
    │
    ├─ await run_in_executor(query_rag)
    │       │
    │       ▼
    │   Thread Pool Worker
    │       │
    │       ├─ LlamaIndex chat_engine.chat()     [sync, runs in thread]
    │       │   ├─ chat_store.get_messages()      [sync]
    │       │   │   └─ run_async_safely(coro)
    │       │   │       └─ run_coroutine_threadsafe(coro, main_loop)
    │       │   │           │
    │       │   │           ▼
    │       │   │       Main Loop executes DB query  ← same loop, same connections
    │       │   │           │
    │       │   │       ◄───┘ future.result() returns
    │       │   │
    │       │   ├─ retriever._retrieve()          [sync]
    │       │   │   └─ run_async_safely(BM25 query)
    │       │   │       └─ (same pattern)
    │       │   │
    │       │   └─ LLM call to Ollama             [sync HTTP, no DB]
    │       │
    │       └─ return result
    │
    ├─ await touch_session_async()                [direct async, main loop]
    └─ return response
```

---

## Rules for Future Development

### 1. Never create temporary event loops for database operations

SQLAlchemy's async documentation states:

> *An application that makes use of multiple event loops should not share the same AsyncEngine with different event loops when using the default pool implementation. If the same engine must be shared between different loops, it should be configured to disable pooling using NullPool.*

Since `NullPool` creates a new connection per operation (expensive and prone to exhaustion under load), the correct approach is to keep all operations on one loop.

### 2. Use `run_async_safely()` for all sync-to-async bridges

Any new component that implements a sync interface but needs async DB access should use `run_async_safely()` from `postgres.py`. Do not create your own bridge.

### 3. Never call blocking sync code directly from `async def` handlers

From the FastAPI documentation:

> *When you declare a path operation function with normal `def` instead of `async def`, it is run in an external threadpool that is then awaited, so as not to block the server.*

If a route handler is `async def` and calls a blocking sync function directly, it blocks the main event loop. Either:
- Use `await loop.run_in_executor(None, sync_function)` to run it in a thread, or
- Define the handler as `def` (not `async def`) so FastAPI auto-offloads it

The `run_in_executor` approach is preferred when the handler also needs `await` for other async operations (as our query handlers do).

### 4. `run_coroutine_threadsafe` requires the target loop to be free

This is the deadlock trap: if you call `run_async_safely()` from the main loop's own thread, the `future.result()` call blocks that thread, but the main loop needs that thread to execute the scheduled coroutine. Result: deadlock. Always ensure sync-to-async bridges are called from a worker thread, not the main loop thread.

### 5. Understand which thread you're on

| Context | Thread | Main loop free? | `run_async_safely` works? |
|---------|--------|-----------------|---------------------------|
| `async def` handler body | Main loop thread | No (you're on it) | **No — deadlock** |
| `def` handler body | FastAPI threadpool | Yes | Yes |
| `run_in_executor` callback | Default threadpool | Yes | Yes |
| `StreamingResponse` sync generator | Starlette threadpool | Yes | Yes |
| LlamaIndex sync interface called from any thread | Worker thread | Yes | Yes |

---

## References

- [SQLAlchemy: Using multiple asyncio event loops](https://docs.sqlalchemy.org/en/21/orm/extensions/asyncio.html) — Documents the `NullPool` workaround and the `AsyncEngine.dispose()` requirement when sharing engines across loops.
- [Python: `asyncio.run_coroutine_threadsafe`](https://docs.python.org/3/library/asyncio-task.html#asyncio.run_coroutine_threadsafe) — Thread-safe coroutine scheduling. Returns `concurrent.futures.Future` for cross-thread result retrieval.
- [Python: Concurrency and Multithreading (asyncio)](https://docs.python.org/3/library/asyncio-dev.html#concurrency-and-multithreading) — Official guidance on mixing asyncio with threads.
- [FastAPI: Async](https://fastapi.tiangolo.com/async/) — Explains `async def` vs `def` handler behavior and threadpool offloading.
