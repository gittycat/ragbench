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
# Local (OrbStack/Docker Desktop)
just deploy local

# Cloud (configure docker-compose.cloud.yml first)
just deploy cloud

# Stop
just deploy-down local
```

### Compose File Structure

- `docker-compose.yml`: Base configuration
- `docker-compose.local.yml`: Local overrides (debug logging)
- `docker-compose.cloud.yml`: Cloud overrides (registry images)
- `docker-compose.ci.yml`: CI/CD infrastructure

### Version Management

Version derived from git tags:

```bash
just show-version          # Display current version
just inject-version 0.2.0  # Update manifests
just release 0.2.0         # Tag, inject, commit, push
```
