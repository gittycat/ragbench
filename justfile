# Load the secrets/.env file
set dotenv-path := "./secrets/.env"

# Suppress command echoing globally
set quiet

alias test := test-unit

default:
    just --list --list-heading "Usage: just <recipe>"

# ============================================================================
# Core — build, start, stop
# ============================================================================

# Build all docker images
[group('core')]
build:
    docker compose build

# Check host dependencies (Docker daemon, Ollama) and fail early with a clear message
[group('core')]
preflight:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! docker info > /dev/null 2>&1; then
        echo "ERROR: Docker daemon is not running. Start OrbStack or Docker Desktop." >&2
        exit 1
    fi
    # Ollama is only required when an active model in config.yml uses the ollama provider
    needs_ollama=false
    for name in $(awk '/^active:/{f=1;next} f&&/^[^ ]/{exit} f{print $2}' config.yml); do
        provider=$(awk -v m="$name:" '$1==m{f=1;next} f&&$1=="provider:"{print $2; exit}' config.yml)
        [ "$provider" = "ollama" ] && needs_ollama=true
    done
    if $needs_ollama && ! curl -sf --max-time 3 http://localhost:11434/api/version > /dev/null; then
        echo "ERROR: Ollama is not running on localhost:11434, but an active model in config.yml uses the ollama provider." >&2
        echo "Start it by opening the Ollama app or running 'ollama serve'." >&2
        exit 1
    fi
    echo "Preflight OK: Docker daemon running$($needs_ollama && echo ", Ollama reachable" || true)"

# Start all services (rag-server, task-worker, webapp, evals, postgres, chromadb)
[group('core')]
up: preflight
    docker compose up -d

# Stop all services
[group('core')]
down:
    docker compose down

# Tail logs from all services
[group('core')]
logs:
    docker compose logs -f

# ============================================================================
# Setup
# ============================================================================

# Install rag_server dev dependencies into a local venv (uv)
[group('setup')]
setup:
    cd services/rag_server && \
    uv sync --group dev --python 3.13

# Pre-download the reranker model into .cache/huggingface (bind-mounted)
[group('setup')]
init MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2":
    mkdir -p .cache/huggingface .cache/datasets
    docker compose run --rm --no-deps --build rag-server \
      .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('{{MODEL}}')"

# Remove __pycache__, .pytest_cache and *.pyc
[group('setup')]
clean:
    fd -t d -H __pycache__ ./services/rag_server -X rm -rf
    fd -t d -H '\.pytest_cache' ./services/rag_server -X rm -rf
    fd -t f -H -e pyc . ./services/rag_server -X rm

# ============================================================================
# Tests
# ============================================================================

# Unit tests (local venv, no docker needed)
[group('test')]
test-unit: setup
    cd services/rag_server && \
    .venv/bin/pytest tests/ --ignore=tests/integration -v

# Integration tests (fresh container, clean state)
[group('test')]
test-integration: up
    docker compose run --rm -e RAG_SERVER_URL=http://rag-server:8001 rag-server \
      .venv/bin/pytest tests/integration -v --run-integration

# Integration tests including slow ones
[group('test')]
test-integration-full: up
    docker compose run --rm -e RAG_SERVER_URL=http://rag-server:8001 rag-server \
      .venv/bin/pytest tests/integration -v --run-integration --run-slow

# ============================================================================
# Evals
# ============================================================================

# Quick eval smoke test (ragbench end-to-end, 5 samples)
[group('eval')]
test-eval: show-config up
    docker compose exec evals .venv/bin/python -m evals.cli eval --tier end_to_end --datasets ragbench --samples 5

# Full eval suite (all end-to-end datasets, all samples)
[group('eval')]
test-eval-full: show-config up
    docker compose exec evals .venv/bin/python -m evals.cli eval --tier end_to_end --datasets ragbench,qasper,hotpotqa,msmarco

# Custom eval run, e.g. `just eval --tier generation --datasets squad_v2 --samples 5`
[group('eval')]
eval +ARGS: show-config up
    docker compose exec evals .venv/bin/python -m evals.cli eval {{ARGS}}

# List available eval datasets
[group('eval')]
eval-datasets: up
    docker compose exec evals .venv/bin/python -m evals.cli datasets

# Calibrate the LLM judge against RAGBench TRACe ground-truth annotations
[group('eval')]
eval-calibrate SAMPLES="20": up
    docker compose exec evals .venv/bin/python -m evals.cli calibrate --samples {{SAMPLES}}

# Compare evaluation runs, e.g. `just eval-compare <run_id> <run_id>`
[group('eval')]
eval-compare +ARGS: up
    docker compose exec evals .venv/bin/python -m evals.cli compare {{ARGS}}

# ============================================================================
# Config
# ============================================================================

# Show RAG configuration (compact)
[group('config')]
show-config:
    cd services/rag_server && \
    .venv/bin/python -c "from infrastructure.config.display import print_config_banner; print_config_banner(compact=True)"

# Show full RAG configuration
[group('config')]
show-config-full:
    cd services/rag_server && \
    .venv/bin/python -c "from infrastructure.config.display import print_config_banner; print_config_banner(compact=False)"

# ============================================================================
# Deploy & Release
# ============================================================================

# Deploy with a compose overlay: `just deploy server` or `just deploy cloud`
[group('deploy')]
deploy ENV="server": preflight
    docker compose -f docker-compose.yml -f docker-compose.{{ENV}}.yml up -d --build

# Stop a deployed overlay
[group('deploy')]
deploy-down ENV="server":
    docker compose -f docker-compose.yml -f docker-compose.{{ENV}}.yml down

# Release: tag v{VERSION}, bump service manifests, commit, push
[group('deploy')]
release VERSION:
    git tag -a v{{VERSION}} -m "Release {{VERSION}}"
    sed -i '' 's/^version = .*/version = "{{VERSION}}"/' services/rag_server/pyproject.toml
    cd services/webapp && npm version {{VERSION}} --no-git-tag-version --allow-same-version
    git add services/rag_server/pyproject.toml services/webapp/package.json services/webapp/package-lock.json
    git commit -m "Bump version to {{VERSION}}"
    git push origin main --tags
