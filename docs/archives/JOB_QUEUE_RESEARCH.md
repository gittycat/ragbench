# Job Queue Research: Replacing PGMQ

## Current State

The project uses **PGMQ** (PostgreSQL Message Queue extension) with the `tembo-pgmq-python` sync library. The queue handles document processing: upload → enqueue → worker picks up → chunk/embed/index → done.

### Problems with PGMQ

| Problem | Impact |
|---------|--------|
| Sync Python library in async codebase | Event loop mixing, `asyncio.run()` wrappers, `run_async_safely()` bridge code |
| pgmq v1.9.0 column mismatch | Custom SQL workaround calling internal `_execute_query_with_result()` — fragile |
| Separate queue state from job tracking | Two systems: pgmq messages + job_tasks table. Redundant and harder to reason about |
| `tembo-pgmq-python` is niche | Small library, low adoption, risky dependency for long-term maintenance |
| Requires pgmq PostgreSQL extension | Extra extension to manage in Docker image and init scripts |

---

## Desired Features

| Feature | Why it matters for this codebase |
|---------|--------------------------------|
| **Async-native (Python async/await)** | The entire codebase is async: asyncpg, SQLAlchemy async, FastAPI, LlamaIndex async. A sync queue library forces event loop mixing — the exact problem we have now. |
| **Lightweight, fast startup** | The worker runs as a Docker container. Heavy frameworks (Celery) add startup time and memory overhead for a system that processes a few documents at a time. |
| **Simple, readable API** | Code must be maintainable by AI agents (Sonnet/GPT). Minimal configuration surface, no implicit magic, straightforward enqueue/dequeue/retry pattern. |
| **Built-in retry with backoff** | Document processing can fail (Ollama down, file parse errors). Need automatic retry with configurable delays. |
| **Progress tracking** | Users poll for upload progress. Either the queue supports job status, or we keep using the existing `job_tasks` table alongside it. |
| **Docker-deployable** | Must run in Docker Compose alongside existing services (postgres, chromadb, rag-server). |
| **Widely adopted (2K+ GitHub stars)** | Ensures long-term maintenance, community support, documentation quality, and AI training data coverage. |
| **No RabbitMQ** | Adds too much operational complexity for this scale. Redis is acceptable as a broker. |

---

## Candidates Evaluated

### Python-Native Libraries

#### Celery — 28K stars
- **Broker:** Redis or RabbitMQ
- **Async:** No (sync, with async wrappers)
- **Pros:** Industry standard, massive ecosystem, every feature imaginable
- **Cons:** Enormous configuration surface, slow startup, heavy memory footprint, notoriously hard to debug. AI agents struggle with implicit behavior and multi-file configuration. Way too heavy for processing a few documents.
- **Verdict:** Excluded — overkill for this project

#### RQ (Redis Queue) — 10.5K stars
- **Broker:** Redis
- **Async:** No (sync only)
- **Pros:** Simplest API in the field (`q.enqueue(func, args)`), built-in web dashboard, built-in retry
- **Cons:** Sync-only — would need `asyncio.run()` wrappers (same problem as PGMQ). Slowest in benchmarks (51s for 20K jobs). Adds Redis container.
- **Verdict:** Simple but reintroduces sync/async mixing

#### Huey — 5.9K stars
- **Broker:** Redis or SQLite
- **Async:** No (sync, threaded mode available)
- **Pros:** Very lightweight ("a little task queue"), battle-tested, simple decorator API, SQLite mode avoids extra container, 0 open issues
- **Cons:** Sync-only — needs `asyncio.run()` wrappers. Single maintainer (bus factor). Adds Redis container (or SQLite for single-worker).
- **Verdict:** Right-sized for the project scope, but sync-only is a real drawback

#### Dramatiq — 5.1K stars
- **Broker:** Redis or RabbitMQ
- **Async:** No (threaded)
- **Pros:** Clean API, good defaults (acks_late, proper prefetch), built-in retries with middleware, better-designed Celery alternative
- **Cons:** Sync/threaded only. Needs Redis. Similar complexity class to Celery, just cleaner.
- **Verdict:** Good middle ground but still sync

#### ARQ — 2.8K stars
- **Broker:** Redis
- **Async:** Yes (native asyncio)
- **Pros:** Created by Samuel Colvin (Pydantic author). Async-native — `async def` tasks work directly. Clean `WorkerSettings` config. Built-in retry, job status tracking.
- **Cons:** **Maintenance-only mode** (confirmed in issue #510). Modernization plans (#437) fell through. Effectively abandoned for active development.
- **Verdict:** Would be the best technical fit, but dead end due to maintenance status

#### Taskiq — 1.9K stars
- **Broker:** Redis, NATS, RabbitMQ, Kafka
- **Async:** Yes (native asyncio)
- **Pros:** Fastest benchmarks (2s for 20K jobs). Native FastAPI integration. Full type hints. Multiple broker backends. Hot reload.
- **Cons:** Below 2K stars threshold. Smaller community, less battle-tested. Thinner documentation.
- **Verdict:** Best async option if stars threshold is relaxed

#### Procrastinate — 1.2K stars
- **Broker:** PostgreSQL (LISTEN/NOTIFY + SKIP LOCKED)
- **Async:** Yes (native asyncio)
- **Pros:** Uses existing PostgreSQL — zero extra infrastructure. Async-native. Retries, task locks, periodic tasks. Jobs are queryable SQL rows. Transactional safety.
- **Cons:** Well below 2K stars. Slower benchmarks (27s for 20K jobs, PostgreSQL overhead). Smaller community.
- **Verdict:** Perfect technical fit for this stack, but small community

#### SAQ — 725 stars
- **Broker:** Redis or PostgreSQL
- **Async:** Yes (native asyncio)
- **Pros:** Lower latency than ARQ (<5ms). Built-in web UI. Inspired by ARQ with improvements.
- **Cons:** Well below 2K stars. Small community.
- **Verdict:** Promising but too small

### Server-Based (Go/Rust with Python clients)

#### NATS + JetStream — 19.1K stars (server)
- **Language:** Go (CNCF project)
- **Python client:** nats.py (1.2K stars, official, async-native)
- **Pros:** Extremely lightweight (~20MB RAM, starts instantly). Battle-tested at massive scale. Async Python client with full JetStream support. Pub/sub + persistence + work queues. Runs on a Raspberry Pi.
- **Cons:** It's a messaging system, not a purpose-built job queue. Need to build queue semantics on top (consumer groups, retry logic, dead letter handling) or use taskiq-nats (part of sub-2K Taskiq). Adds a container.
- **Verdict:** Powerful infrastructure, but requires building job queue abstractions

#### Faktory — 6.1K stars (server)
- **Language:** Go (by Mike Perham, creator of Sidekiq)
- **Python clients:** pyfaktory (16 stars), faktory_worker_python (71 stars)
- **Pros:** Standalone binary, no Redis needed (uses RocksDB). Built-in web UI. Designed as language-agnostic Sidekiq. Retries, scheduling, batches.
- **Cons:** Both Python clients are tiny, sync-only, and barely maintained. Last release of main Python client: April 2022. Reintroduces sync/async mixing.
- **Verdict:** Great server, poor Python story

#### Asynq — 12.9K stars (server)
- **Language:** Go
- **Pros:** Simple, reliable, efficient. Web UI (Asynqmon). Priority queues, task dedup, scheduling.
- **Cons:** Go-only — no Python client exists at all. Would need to interact via Redis protocol directly.
- **Verdict:** Not viable without writing a Python client from scratch

### PostgreSQL SKIP LOCKED (no library)

- **Broker:** Existing PostgreSQL instance
- **Async:** Yes (via SQLAlchemy async, already in use)
- **Pros:** Zero extra containers or dependencies. Fully async. The `job_tasks` table already exists and tracks task state — just add `SELECT FOR UPDATE SKIP LOCKED` polling. ~50 lines of code. Validated pattern used by 37signals (Solid Queue), pg-boss (Node, 2.3K stars), Oban (Elixir, 3.5K stars). HN/Reddit consensus: "Use PostgreSQL until you can't."
- **Cons:** Not a library — must implement retry, backoff, and polling yourself. No web UI. ~10x slower than Redis for raw throughput (irrelevant at this scale). Table bloat under high UPDATE churn (irrelevant at this scale).
- **Verdict:** Simplest path. Eliminates a dependency instead of adding one.

---

## Benchmark Reference (20K jobs, 10 workers)

| Library | Backend | Time |
|---------|---------|------|
| Taskiq | Redis Streams | 2.0s |
| Huey | Redis | 3.6s |
| Dramatiq | Redis | 4.1s |
| Celery (threads) | Redis | 11.7s |
| Procrastinate | PostgreSQL | 27.5s |
| ARQ | Redis | 35.4s |
| RQ | Redis | 51.1s |

Source: [Steven Yue's benchmark (2025)](https://stevenyue.com/blogs/exploring-python-task-queue-libraries-with-load-test)

At this project's scale (single-digit documents per batch), all options are effectively instant.

---

## Shortlist

### 1. PostgreSQL SKIP LOCKED (via existing job_tasks table)
- No new dependency, no new container
- Fully async with SQLAlchemy (already in codebase)
- ~50 lines of queue code, merged into existing job tracking
- Trade-off: DIY retry/backoff (~20 extra lines)

### 2. NATS JetStream (Go server + nats.py async client)
- Adds lightweight container (~20MB), async Python client
- Massive adoption (19.1K stars server, CNCF)
- Trade-off: messaging system, not a job queue — need to build queue semantics or use taskiq-nats

### 3. Huey (Python, Redis or SQLite backend)
- 5.9K stars, battle-tested, simplest decorator API
- Trade-off: sync-only, needs `asyncio.run()` wrappers (same problem as PGMQ)

---

## Sources

- [ARQ maintenance-only mode (issue #510)](https://github.com/python-arq/arq)
- [ARQ future plans (issue #437)](https://github.com/python-arq/arq/issues/437)
- [streaq discussion: "ARQ is officially dead"](https://github.com/tastyware/streaq/discussions/108)
- [Python Task Queue Benchmark with Load Tests](https://stevenyue.com/blogs/exploring-python-task-queue-libraries-with-load-test)
- [Choosing The Right Python Task Queue](https://judoscale.com/blog/choose-python-task-queue)
- [PGQueuer HN Discussion (SKIP LOCKED pattern)](https://news.ycombinator.com/item?id=42348864)
- [Procrastinate HN Discussion](https://news.ycombinator.com/item?id=30126152)
- [The Unreasonable Effectiveness of SKIP LOCKED](https://www.inferable.ai/blog/posts/postgres-skip-locked)
- [Faktory — language-agnostic job server](https://github.com/contribsys/faktory)
- [NATS Server](https://github.com/nats-io/nats-server)
- [nats.py async client](https://github.com/nats-io/nats.py)
- [Asynq — Go task queue](https://github.com/hibiken/asynq)
- [Taskiq — async Python task queue](https://github.com/taskiq-python/taskiq)
