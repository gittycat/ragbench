# Configuration & Secrets

## YAML Configuration (`config.yml`)

Primary configuration file for models and retrieval settings.

**Setup:** Simply edit the existing file.

## Environment Variables (docker-compose.yml)

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_HOST` | `postgres` | PostgreSQL host |
| `DATABASE_PORT` | `5432` | PostgreSQL port |
| `DATABASE_NAME` | `ragbench` | PostgreSQL database |
| `LOG_LEVEL` | `WARNING` | Logging verbosity |
| `MAX_UPLOAD_SIZE` | `80` | Max upload size in MB |
| `RAG_SERVER_URL` | `http://rag-server:8001` | RAG server URL (used by evals + webapp) |
| `EVALS_SERVICE_URL` | `http://evals:8002` | Eval service URL (used by webapp) |

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
