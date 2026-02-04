# Centralized Logging Infrastructure Implementation Plan
Created Jan 24 2026

## Overview

Implement a production-ready centralized logging stack using **Grafana Loki + Promtail + Grafana** with **structlog** for enhanced structured logging. This solution provides log aggregation, visualization, and cross-service request tracing while preserving existing logging patterns.

**Recommendation:** Grafana Loki stack (industry-standard for Docker/K8s, lightweight, no Elasticsearch complexity)

## Current State

- **Logging foundation:** `core/logging.py` with environment-based LOG_LEVEL, custom filters (URLShortener, HealthCheck)
- **312 logging calls** across 34 files using contextual prefixes: `[STARTUP]`, `[QUERY]`, `[TASK {id}]`, `[PGMQ]`
- **5 Docker services:** rag-server, pgmq-worker, evals, postgres, webapp
- **Pain point:** Logs scattered across containers, no centralization, lost on restart

## Implementation Phases

### Phase 1: Add Structured Logging (structlog)

**Goal:** Enhance existing logs with JSON formatting and correlation IDs (backward compatible, zero breaking changes)

**1.1 Add structlog dependency**
- File: `services/rag_server/pyproject.toml`
- Add: `structlog = "^24.4.0"`
- Run: `uv sync` to install

**1.2 Create structlog configuration module**
- File: `services/rag_server/core/structlog_config.py` (new)
- Features:
  - JSON processor for Loki ingestion
  - Context processor for correlation IDs (`request_id`, `session_id`, `task_id`, `eval_run_id`)
  - Integration with existing `logging.Logger` instances
  - Preserves LOG_LEVEL environment variable behavior
  - Automatic exception tracking with `exc_info`

**1.3 Update existing logging config**
- File: `services/rag_server/core/logging.py`
- Changes:
  - Add `configure_structlog()` function
  - Call from `configure_logging()` to initialize both systems
  - Make structlog optional (graceful degradation if import fails)
  - Preserve all existing filters (URLShortener, HealthCheck)

**1.4 Add correlation ID middleware**
- File: `services/rag_server/middleware/correlation.py` (new)
- Features:
  - Generate `request_id` (UUID) per API request
  - Bind to structlog context: `structlog.contextvars.bind_contextvars(request_id=...)`
  - Pass `request_id` to PGMQ tasks via message metadata
  - Return `X-Request-ID` header in responses for client tracing

**1.5 Update FastAPI application**
- File: `services/rag_server/main.py`
- Changes:
  - Import and apply `CorrelationMiddleware`
  - Call `configure_structlog()` at startup (alongside existing `configure_logging()`)

**Impact:** ~200 lines of new code, 10 lines modified. No changes to existing 312 logging calls.

---

### Phase 2: Docker Compose Logging Stack

**Goal:** Deploy Loki, Promtail, and Grafana as Docker services

**2.1 Create logging stack compose file**
- File: `docker-compose.logging.yml` (new)
- Services:
  - **loki:** `grafana/loki:3.3.2` (port 3100, persistent volume: `loki_data`)
  - **promtail:** `grafana/promtail:3.3.2` (mounts `/var/run/docker.sock` for container discovery)
  - **grafana:** `grafana/grafana:11.5.0` (port 3001, persistent volume: `grafana_data`)
- Networks:
  - New `logging` network (bridge mode)
  - Promtail also connects to `private` network for health checks

**2.2 Update main docker-compose.yml**
- File: `docker-compose.yml`
- Changes for all services (rag-server, pgmq-worker, evals, postgres, webapp):
  - Add `logging` network
  - Add logging driver config:
    ```yaml
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
    ```
  - Add labels for Promtail discovery:
    ```yaml
    labels:
      - "logging=promtail"
      - "service=<service-name>"
    ```

**2.3 Create Loki configuration**
- File: `config/loki/loki-config.yml` (new)
- Settings:
  - Filesystem storage: `/loki`
  - 30-day retention policy
  - Label-based indexing (service, level, job)
  - Optimized chunk size for Docker logs

**2.4 Create Promtail configuration**
- File: `config/promtail/promtail-config.yml` (new)
- Features:
  - Auto-discover containers via Docker socket
  - Extract labels from container metadata (service, environment)
  - Parse JSON logs from structlog
  - Relabel with contextual fields (request_id, session_id, task_id)
  - Filter out health check spam

**2.5 Add Grafana admin password to secrets**
- File: `secrets/.env.example`
- Add: `GRAFANA_ADMIN_PASSWORD=admin`
- User copies to `secrets/.env` (gitignored)

**Impact:** 1 new compose file, 3 config files, minimal changes to existing compose file

---

### Phase 3: Grafana Dashboards

**Goal:** Provide pre-built dashboards for monitoring

**3.1 Loki datasource provisioning**
- File: `config/grafana/datasources/loki.yml` (new)
- Connects Grafana to Loki at `http://loki:3100`

**3.2 Dashboard provisioning config**
- File: `config/grafana/dashboards/dashboard.yml` (new)
- Auto-loads dashboards from `/etc/grafana/provisioning/dashboards/`

**3.3 Application Errors Dashboard**
- File: `config/grafana/dashboards/application-errors.json` (new)
- Panels:
  - Error rate over time (by service)
  - Exception types breakdown
  - Recent error logs (table with request_id, session_id)
  - Top error sources (by logger name)
  - Error correlation (task_id → error)
- Example LogQL: `{service=~"rag-server|pgmq-worker"} |= "ERROR" | json`

**3.4 PGMQ Task Tracking Dashboard**
- File: `config/grafana/dashboards/pgmq-tasks.json` (new)
- Panels:
  - Task processing rate (tasks/min)
  - Task duration histogram
  - Failed tasks (by filename)
  - Task progress tracking (chunk embeddings)
- Example LogQL: `{service="pgmq-worker"} |~ "TASK" | json | task_id != ""`

**3.5 Query Performance Dashboard**
- File: `config/grafana/dashboards/query-performance.json` (new)
- Panels:
  - Query latency (p50, p95, p99)
  - Token usage over time
  - Retrieval timing (hybrid search)
  - Slow queries (>5s)
- Example LogQL: `{service="rag-server"} |= "[QUERY]" | json`

**3.6 Infrastructure Health Dashboard**
- File: `config/grafana/dashboards/infrastructure.json` (new)
- Panels:
  - Container status (up/down)
  - PostgreSQL document count
  - PostgreSQL connection errors
  - Ollama model load status
  - BM25 index refresh events
- Example LogQL: `{service="rag-server"} |~ "postgres" |= "ERROR"`

**Impact:** 6 new JSON files for dashboards + datasource provisioning

---

### Phase 4: Justfile Integration

**Goal:** Add commands for easy logging stack management

**4.1 Update justfile**
- File: `justfile`
- Add commands:
  ```just
  # Start logging stack
  logging-up:
      docker compose -f docker-compose.yml -f docker-compose.logging.yml up -d

  # Stop logging stack
  logging-down:
      docker compose -f docker-compose.logging.yml down

  # View Loki logs
  logging-logs:
      docker compose -f docker-compose.logging.yml logs -f loki

  # Restart Promtail (after label changes)
  logging-restart-promtail:
      docker compose -f docker-compose.logging.yml restart promtail

  # Open Grafana
  logging-open:
      @echo "Opening Grafana at http://localhost:3001"
      @echo "Default credentials: admin / admin"
      @open http://localhost:3001 || xdg-open http://localhost:3001
  ```

**Impact:** ~15 lines added to justfile

---

### Phase 5: Documentation

**Goal:** Document the logging infrastructure for team use

**5.1 Create logging infrastructure guide**
- File: `docs/LOGGING_INFRASTRUCTURE.md` (new)
- Sections:
  - Architecture overview (diagram showing log flow)
  - Quick start: `just logging-up`
  - Accessing Grafana (http://localhost:3001, admin/admin)
  - Using dashboards (screenshots, common queries)
  - LogQL cheat sheet for developers
  - Troubleshooting (Loki not receiving logs, Promtail errors)
  - Correlation ID examples (tracing requests across services)

**5.2 Update CLAUDE.md**
- File: `CLAUDE.md`
- Add section:
  ```markdown
  ## Centralized Logging

  **Stack:** Grafana Loki + Promtail + Grafana

  **Quick Start:**
  ```bash
  just logging-up
  just logging-open  # Opens Grafana at http://localhost:3001
  ```

  **Dashboards:**
  - Application Errors: Track exceptions across services
  - PGMQ Tasks: Monitor async processing
  - Query Performance: Latency, token usage
  - Infrastructure: Container health, PostgreSQL status

  **Correlation IDs:**
  - `request_id`: Full API request trace
  - `session_id`: Conversation history
  - `task_id`: PGMQ task execution

  **Common Queries:**
  ```logql
  # Find all logs for a specific request
  {service="rag-server"} | json | request_id = "abc123"

  # Track PGMQ task from queue to completion
  {service="pgmq-worker"} | json | task_id = "xyz789"

  # All errors in last hour
  {service=~"rag-server|pgmq-worker"} |= "ERROR" | json
  ```
  ```

**5.3 Create config README**
- File: `config/README.md` (new)
- Document configuration files:
  - `loki/loki-config.yml`: Retention, storage, limits
  - `promtail/promtail-config.yml`: Log discovery, parsing
  - `grafana/datasources/`: Loki connection
  - `grafana/dashboards/`: Pre-built dashboards

**Impact:** 3 documentation files

---

## Critical Files to Create/Modify

**New Files (13):**
1. `services/rag_server/core/structlog_config.py` - structlog configuration
2. `services/rag_server/middleware/correlation.py` - Correlation ID middleware
3. `docker-compose.logging.yml` - Logging stack services
4. `config/loki/loki-config.yml` - Loki configuration
5. `config/promtail/promtail-config.yml` - Promtail configuration
6. `config/grafana/datasources/loki.yml` - Grafana datasource
7. `config/grafana/dashboards/dashboard.yml` - Dashboard provisioning
8. `config/grafana/dashboards/application-errors.json` - Errors dashboard
9. `config/grafana/dashboards/pgmq-tasks.json` - Tasks dashboard
10. `config/grafana/dashboards/query-performance.json` - Performance dashboard
11. `config/grafana/dashboards/infrastructure.json` - Infrastructure dashboard
12. `docs/LOGGING_INFRASTRUCTURE.md` - User documentation
13. `config/README.md` - Config documentation

**Modified Files (7):**
1. `services/rag_server/pyproject.toml` - Add structlog dependency
2. `services/rag_server/core/logging.py` - Integrate structlog initialization
3. `services/rag_server/main.py` - Add correlation middleware
4. `docker-compose.yml` - Add logging network and labels to all services
5. `justfile` - Add logging commands
6. `CLAUDE.md` - Add logging section
7. `secrets/.env.example` - Add GRAFANA_ADMIN_PASSWORD

## Verification Plan

**1. Unit Tests**
- No changes needed (structlog doesn't affect test behavior)
- Existing tests continue to work

**2. Integration Tests**
- Add test: Verify `X-Request-ID` header in response
- Add test: Verify structlog JSON output in logs

**3. Manual Verification**
```bash
# 1. Start logging stack
just logging-up

# 2. Verify services are running
docker compose -f docker-compose.logging.yml ps

# 3. Upload a document
curl -X POST http://localhost:8001/upload -F "file=@test.pdf"

# 4. Check Grafana
just logging-open
# Navigate to PGMQ Tasks dashboard
# Verify task logs appear with task_id

# 5. Run a query
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "session_id": "test123"}'

# 6. Verify request_id in response header
# Check Application Errors dashboard for any errors
# Search logs by request_id in Explore view

# 7. Verify correlation
# In Grafana Explore: {service="rag-server"} | json | request_id = "<id>"
# Should show full request trace
```

**4. Performance Check**
- Log volume estimation: ~100MB/day (5 services × 20MB/day)
- Loki storage: 3GB for 30-day retention (compressed)
- Memory overhead: ~300-400MB total (Loki + Promtail + Grafana)

## Migration Strategy

**Zero Downtime:**
1. Phase 1 (structlog) is backward compatible - logs still go to stdout
2. Phase 2 (Docker Compose) adds logging stack alongside existing services
3. Promtail reads existing logs retroactively (from container start)
4. Team can use `docker compose logs` CLI or Grafana (both work)

**Rollback:**
- Remove `logging` network from docker-compose.yml
- Stop logging stack: `just logging-down`
- structlog gracefully degrades to standard logging if disabled

## Open Questions

1. **Grafana admin password**: Use default `admin/admin` for local dev, or require users to set in secrets/.env?
   - **Recommendation:** Add to `secrets/.env.example` as `GRAFANA_ADMIN_PASSWORD=admin`, users can override

2. **Log retention**: 30 days sufficient, or need longer for compliance?
   - **Recommendation:** Start with 30 days, add S3 archival if needed later

3. **Docker Compose integration**: Merge logging into main compose file or keep separate?
   - **Recommendation:** Keep separate for cleaner separation, users opt-in with `just logging-up`

## Performance Impact

- **structlog:** ~5-10% CPU overhead per log call (negligible)
- **Promtail:** ~50MB RAM, minimal CPU
- **Loki:** ~100-200MB RAM for 30-day retention
- **Grafana:** ~150MB RAM
- **Total overhead:** ~300-400MB RAM, <5% CPU

## Security Considerations

- Logs may contain query text, filenames, session IDs (no passwords/API keys)
- Grafana behind authentication (admin user required)
- Grafana port 3001 exposed to localhost only
- Consider: Add sensitive data scrubbing in Promtail config (future enhancement)

## Next Steps After Approval

1. Create structlog module and correlation middleware
2. Update logging config and main.py
3. Create docker-compose.logging.yml
4. Create Loki/Promtail/Grafana configs
5. Build 4 Grafana dashboards
6. Update justfile and documentation
7. Test end-to-end flow
8. Update CI pipeline (optional health check)
