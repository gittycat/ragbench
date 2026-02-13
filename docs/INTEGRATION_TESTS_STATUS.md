# Integration Tests — TODO

## Docker Strategy

No dedicated `test-runner` service. Tests reuse the `rag-server` service definition to eliminate config drift (duplicated env vars, secrets, volumes, networks).

### Local / Debug

```bash
docker compose exec -T rag-server \
  .venv/bin/pytest tests/integration -v --run-integration
```

- Attaches to the running rag-server container — fast, no startup cost
- `RAG_SERVER_URL=http://localhost:8001` (tests and server share the container)
- Shared filesystem state with the running server (acceptable for local iteration)

### CI / Reproducible

```bash
docker compose run --rm \
  -e RAG_SERVER_URL=http://rag-server:8001 \
  rag-server .venv/bin/pytest tests/integration -v --run-integration
```

- Fresh container from the same service definition — clean state, no leakage between runs
- `RAG_SERVER_URL=http://rag-server:8001` (sibling container, resolved via Docker DNS)
- `--rm` cleans up the container after the run

### Why not a separate test-runner service

The previous `test-runner` service in `docker-compose.yml` duplicated rag-server's entire config (same Dockerfile, env, secrets, volumes, networks). Every change to rag-server required a matching change to test-runner, causing failures from config drift (wrong DB host, missing secrets, missing volume mounts).

Both `exec` and `run` against the existing `rag-server` service inherit all config automatically.

## Remaining Work

- [ ] Remove `test-runner` service from `docker-compose.yml`
- [ ] Update `justfile` test-integration recipes to use `exec` / `run`
- [ ] Fix `conftest.py` to detect exec vs run and set `RAG_SERVER_URL` default accordingly
- [ ] Fix remaining test failures (file processing, timing, cleanup)
- [ ] Delete this document once tests are green
