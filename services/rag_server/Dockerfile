# syntax=docker/dockerfile:1
# Use Python 3.13 slim image
FROM python:3.13-slim

# Install system dependencies for docling (OCR support) and build tools (for pystemmer/BM25)
# Graphics libraries needed for OpenCV (used by Docling for image processing)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    gcc \
    g++ \
    make \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:0.9.0 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Create non-root user early
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /tmp/shared && \
    chown -R appuser:appuser /tmp/shared && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser/.cache && \
    mkdir -p /data/documents && \
    chown -R appuser:appuser /data/documents

# Switch to non-root user for dependency installation
USER appuser

# Copy dependency file first for layer caching
COPY --chown=appuser:appuser services/rag_server/pyproject.toml ./

# Install dependencies using uv with CPU-only PyTorch for smaller image
# Use unsafe-best-match to allow finding best versions across all indexes
# Cache mount persists downloaded packages across builds
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=1000,gid=1000 \
    uv sync --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

# Copy application code (exclude pyproject.toml to avoid overwriting .venv)
COPY --chown=appuser:appuser services/rag_server/api ./api
COPY --chown=appuser:appuser services/rag_server/schemas ./schemas
COPY --chown=appuser:appuser services/rag_server/services ./services
COPY --chown=appuser:appuser services/rag_server/pipelines ./pipelines
COPY --chown=appuser:appuser services/rag_server/infrastructure ./infrastructure
COPY --chown=appuser:appuser services/rag_server/core ./core
COPY --chown=appuser:appuser services/rag_server/app ./app
COPY --chown=appuser:appuser services/rag_server/tests ./tests
COPY --chown=appuser:appuser services/rag_server/scripts ./scripts
COPY --chown=appuser:appuser services/rag_server/main.py ./

# Expose port
EXPOSE 8001

# Run using the pre-built virtual environment directly
# Use shell form to interpolate LOG_LEVEL env var (converted to lowercase for uvicorn)
CMD .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001 --log-level $(echo ${LOG_LEVEL:-warning} | tr '[:upper:]' '[:lower:]')
