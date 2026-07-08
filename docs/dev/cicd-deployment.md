# CI/CD & Deployment

## Forgejo Setup

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

## Pipeline (`.forgejo/workflows/ci.yml`)

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

## Secrets Configuration

Repository Settings → Secrets and Variables → Actions:
- `ANTHROPIC_API_KEY`: Required for evaluation tests

## Deployment

### Environment-Based Deployment

```bash
# Local (OrbStack/Docker Desktop) — base compose + override
just up

# Server tier (Caddy TLS + bearer auth)
just deploy server

# Cloud (configure docker-compose.cloud.yml first)
just deploy cloud

# Stop
just deploy-down server
```

### Compose File Structure

- `docker-compose.yml`: Base configuration
- `docker-compose.override.yml`: Local dev overrides (auto-loaded)
- `docker-compose.cloud.yml`: Cloud overrides (registry images)
- `docker-compose.ci.yml`: CI/CD infrastructure
- `docker-compose.server.yml`: Confidential-compute VM / thin-client tier overlay (see below)

### Server Tier (Caddy TLS + Bearer Auth)

For the confidential-compute VM tier, thin clients connect over a network
instead of localhost, so transport security and auth are needed:

```bash
docker compose -f docker-compose.yml -f docker-compose.server.yml up
# or: just deploy server
```

- Puts [Caddy](https://caddyserver.com) in front of the webapp with automatic
  HTTPS (`services/caddy/Caddyfile`). Set `SERVER_DOMAIN` for a real domain +
  Let's Encrypt, otherwise Caddy's internal CA self-signs for `localhost`/LAN
  IPs. Only Caddy publishes ports in this overlay — webapp/rag-server/evals
  stop publishing to the host.
- Enables bearer-token auth on `rag-server` (`infrastructure/auth.py`): mount
  a token at `secrets/RAG_SERVER_AUTH_TOKEN` and every route except `/health`
  requires `Authorization: Bearer <token>`. The webapp reads the same secret
  and forwards it on proxied `/api/*` requests. Unset in the base/local
  compose file, so local-tier behavior is unchanged.

### Version Management

Version derived from git tags:

```bash
just release 0.2.0  # Tag v0.2.0, bump manifests, commit, push
```
