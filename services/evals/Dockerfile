FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Create non-root user with home directory
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -m -d /home/appuser appuser && \
    chown -R appuser:appuser /app

# Copy project files
COPY --chown=appuser:appuser services/evals/pyproject.toml services/evals/uv.lock ./
COPY --chown=appuser:appuser services/evals/evals ./evals
COPY --chown=appuser:appuser services/evals/infrastructure ./infrastructure

# Install dependencies
RUN uv sync --frozen

# Switch to non-root user
USER appuser

# Set entrypoint
ENTRYPOINT [".venv/bin/python", "-m", "evals.cli"]
