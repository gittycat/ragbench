# UV_CACHE_DIR := invocation_directory() + "/.cache/uv"

alias test := test-unit

# Get version from latest git tag (strips 'v' prefix), falls back to 0.0.0-dev
version := `git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || echo "0.0.0-dev"`

default:
    @just --list --list-heading "Usage: list <recipe>"

# ============================================================================
# Version Management
# ============================================================================

# Show current version from git tags
show-version:
    @echo "{{version}}"

# Inject version into all service manifest files
inject-version VERSION=version:
    @echo "Injecting version {{VERSION}}..."
    sed -i '' 's/^version = .*/version = "{{VERSION}}"/' services/rag_server/pyproject.toml
    cd services/webapp && npm version {{VERSION}} --no-git-tag-version --allow-same-version
    @echo "Version {{VERSION}} injected into all services"

# Create a release: tag, inject version, commit, and push
release VERSION:
    @echo "Creating release v{{VERSION}}..."
    git tag -a v{{VERSION}} -m "Release {{VERSION}}"
    just inject-version {{VERSION}}
    git add services/rag_server/pyproject.toml services/webapp/package.json services/webapp/package-lock.json
    git commit -m "Bump version to {{VERSION}}"
    git push origin main --tags
    @echo "Release v{{VERSION}} created and pushed"

# ============================================================================
# Deployment
# ============================================================================

# Deploy to specified environment (local or cloud)
deploy ENV="local":
    docker compose -f docker-compose.yml -f docker-compose.{{ENV}}.yml up -d --build

# Deploy to local OrbStack
deploy-local:
    just deploy local

# Deploy to cloud (placeholder - configure docker-compose.cloud.yml first)
deploy-cloud:
    just deploy cloud

# Stop deployment for specified environment
deploy-down ENV="local":
    docker compose -f docker-compose.yml -f docker-compose.{{ENV}}.yml down 

setup:
    cd services/rag_server && \
    uv sync --group dev

test-unit: setup
    cd services/rag_server && \
    .venv/bin/pytest tests/ --ignore=tests/integration --ignore=tests/evaluation --ignore=tests/test_rag_eval.py -v

test-integration: setup docker-up
    cd services/rag_server && \
    .venv/bin/pytest tests/integration -v --run-integration

test-eval:
    cd services/rag_server && \
    uv sync --group dev --group eval && \
    .venv/bin/pytest tests/test_rag_eval.py --run-eval --eval-samples=5 -v

test-eval-full:
    cd services/rag_server && \
    uv sync --group dev --group eval && \
    .venv/bin/pytest tests/test_rag_eval.py --run-eval -v

docker-up:
    docker compose up -d

docker-down:
    docker compose down

docker-logs: docker-up
    docker compose logs -f

migrate-sessions: docker-up
    docker compose exec rag-server python scripts/migrate_sessions.py

@clean:
    # These fd commands run in parallel
    # -H includes hidden dirs (.pytest_cache)
    # -e pyc filters by extension directly
    # -X batches results into single rm call
    @fd -t d -H -X rm -rf __pycache__ ./services/rag_server
    @fd -t d -H -X rm -rf '\.pytest_cache' ./services/rag_server
    @fd -t f -H -e pyc -X rm . ./services/rag_server

# ============================================================================
# Benchmarking
# ============================================================================

# Setup benchmark dependencies
bench-setup:
    cd services/rag_server && \
    uv sync --group dev --group eval --group bench

# Start ephemeral benchmark environment
bench-up:
    docker compose -f docker-compose.bench.yml up -d --build
    @echo "Waiting for services to be healthy..."
    @sleep 10
    @echo "Benchmark environment ready at http://localhost:8003"

# Stop benchmark environment (ephemeral data is lost)
bench-down:
    docker compose -f docker-compose.bench.yml down -v

# View benchmark environment logs
bench-logs:
    docker compose -f docker-compose.bench.yml logs -f

# Load a dataset (downloads and caches)
bench-load DATASET:
    cd services/rag_server && \
    .venv/bin/python -c "from evaluation.datasets import get_dataset; d = get_dataset('{{DATASET}}'); d.load(); print(f'Loaded {d.info.name}: {d.info.num_documents} docs, {d.info.num_test_cases} tests')"

# Run benchmark on a dataset
bench-run DATASET SAMPLES="": bench-up bench-setup
    cd services/rag_server && \
    .venv/bin/python -m evaluation.benchmark_cli run {{DATASET}} {{ if SAMPLES != "" { "--samples " + SAMPLES } else { "" } }}

# Run quick benchmark (100 samples)
bench-quick DATASET="squad": (bench-run DATASET "100")

# List available datasets
bench-datasets:
    cd services/rag_server && \
    .venv/bin/python -m evaluation.benchmark_cli datasets

# Show benchmark history
bench-history DATASET="":
    cd services/rag_server && \
    .venv/bin/python -m evaluation.benchmark_cli history {{DATASET}}

# Show benchmark statistics
bench-stats DATASET="":
    cd services/rag_server && \
    .venv/bin/python -m evaluation.benchmark_cli stats {{DATASET}}

# Launch benchmark dashboard
bench-dashboard: bench-setup
    cd services/rag_server && \
    .venv/bin/python -m evaluation.benchmark_cli dashboard
    