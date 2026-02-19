# Development Setup

## Prerequisites

1. **Python 3.13+** with [uv package manager](https://docs.astral.sh/uv/)
2. **Node.js 22+** with npm
3. **Docker** (Docker Desktop, OrbStack, or Podman)
4. **Ollama** running on host with required models:
   ```bash
   ollama pull gemma3:4b
   ollama pull nomic-embed-text
   ```

## Local Development

```bash
# Clone and setup
git clone <repo-url>
cd ragbench

# Backend dependencies
cd services/rag_server
uv sync
uv sync --group dev    # Add test dependencies
uv sync --group eval   # Add evaluation dependencies

# Frontend dependencies
cd ../webapp
npm install

# Configuration
cp config.yml.example config.yml
cp secrets/.env.example secrets/.env

# Start infrastructure
docker compose up -d

# Run RAG server (development)
cd services/rag_server
.venv/bin/uvicorn main:app --reload --port 8001

# Run frontend (development)
cd services/webapp
npm run dev
```

## Task Runner (just)

This project uses [just](https://just.systems/) for task automation.

```bash
# List all tasks
just

# Development
just setup              # Install dependencies
just docker-up          # Start all services
just docker-down        # Stop services
just docker-logs        # View logs

# Testing
just test-unit          # Unit tests only
just test-integration   # Integration tests (requires docker)
just test-eval          # Quick evaluation (5 samples)
just test-eval-full     # Full evaluation

# Deployment
just deploy local       # Deploy locally
just deploy cloud       # Deploy to cloud
just deploy-down local  # Stop deployment

# Version management
just show-version       # Show current version
just inject-version X.Y.Z  # Update version in manifests
just release X.Y.Z      # Full release workflow
```

## Production Considerations

This project is not production-ready for enterprise deployment.

**Missing Capabilities:**

1. **Security**: Authentication, authorization, API keys, audit logging
2. **Observability**: Infrastructure monitoring, APM, distributed tracing
3. **High Availability**: Load balancing, failover, health-based routing
4. **Disaster Recovery**: Automated backups, cross-region replication, RTO/RPO SLAs

**Hardware Requirements:**
- Minimum: 4GB RAM, any CPU (CPU-only PyTorch)
- Recommended: GPU with 8GB+ VRAM for larger models
- For private deployment: In-house server with powerful GPU needed for full-size open models
