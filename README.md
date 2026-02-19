## About

This projects implements a RAG AI assistant that searches your organisation’s trusted content and answers questions using it. It delivers more accurate, relevant responses grounded in your data than what is obtained by simple LLM apps like ChatGPT.

RAGs have become the most common application of AI in enterprise environments.
This specific RAG focuses on two core features: data privacy and observability.

## Data Privacy

- Option to run fully on-prem (locally). No calls outside the intranet needed.
For decent performance, this requires a dedicated server spec'd for large open source models.
- Frontier cloud models can also be used. Privacy is then ensured by performing data anynimization on any request to the cloud and inserting back the redacted data on responses.

## Observability

This covers both measures of the **quality** of the data (accuracy, completeness, groundedness / hallucination rate, relevance) and **Operational metrics** values like cost, latency and speed.

The system includes an **Evaluation Service** that runs automated quality assessments against multiple datasets. Results are distilled into 5 dashboard metrics:

- **Retrieval Relevance** — Are we finding the right content?
- **Faithfulness** — Is the answer grounded in retrieved context?
- **Answer Completeness** — Does the answer cover all key points?
- **Answer Relevance** — Does the answer address the question asked?
- **Response Latency** — Is the system fast enough?

These metrics allow admins to determine the best combinations of LLM models and settings for their data and organisation constraints.

## Tech Stack

- **Backend**: Python, FastAPI, PostgreSQL (pg_textsearch for BM25)
- **RAG Pipeline**: Docling, LlamaIndex
- **Vector DB**: ChromaDB
- **Search**: Hybrid (BM25 + Vector + RRF)
- **LLM**: Ollama (local) or cloud providers (OpenAI, Anthropic, etc.)
- **Infrastructure**: Docker compose

## Requirements

- **Docker** - Docker Desktop, OrbStack, or Podman
- **Ollama** - For local AI models (optional if using cloud only)
- **4GB RAM** - For local development with slow inference.
- **2GB disk** - For models and data for development.

## Status

This is a development/research project, not production-ready software. It lacks authentication, enterprise security, monitoring, high availability features to name some main ones.

## AI Development

This project is developed using **Claude Code** (Anthropic) as the primary coding assitant. OpenAI GPT and Google Gemini models are also used to explore alternative implementations.

All code is reviewed, tested (TDD), and validated for correctness and security.

## Quick Start

### 1. Install Prerequisites

**macOS:**
```bash
# Install Ollama
brew install ollama

# Download AI models
ollama pull gemma3:4b
ollama pull nomic-embed-text

# Install Docker
brew install orbstack            # or Docker Desktop if you prefer
```

**Linux:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download AI models
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

### 2. Download ragbench source

```bash
# Clone the repository
git clone https://github.com/gittycat/ragbench.git
cd ragbench
```

### 3. Configure

**Select the LLM Models to use** (`config.yml`):

The `config.yml` file defines available models and RAG settings. The `active` section controls which models are used:

```yaml
active:
  inference: gemma3-4b    # LLM for answering questions
  embedding: nomic-embed  # Model for document embeddings
  eval: claude-sonnet     # Model for evaluation metrics
  reranker: minilm-l6     # Model for result reranking
```

To switch models, change the active model name to any model defined in the `models` section. Local models (Ollama) work out of the box. Cloud models require API keys.

**Secrets**:

API keys are provided via Docker Compose secrets mounted as files under `/run/secrets` and loaded at startup via Pydantic Settings (no environment variables). This follows OWASP best practices for secrets handling and storage guidance:

```
https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html?utm_source=chatgpt.com
https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html
```

Database access also uses secrets per service. Create the required files under `secrets/` before starting containers.

### 4. Start the Application

```bash
# Start Ollama (if not already running)
ollama serve &

# Pre-fetch the re-ranking model
# This significantly speeds up the rag-server container startup.
just init

# Start RAG Bench
docker compose up -d
```

Open **http://localhost:8000** in your browser. The eval service API is available at **http://localhost:8002**.

### 5. Stop the Application

```bash
docker compose down
```

### Delete all persistent stores (database, document storage)

```bash
docker compose down -v
```

## Development

For development setup, testing, and technical documentation, see [DEVELOPMENT.md](DEVELOPMENT.md).
This tech doc is more of a dumping ground for notes as I hit them during development.
It is focused on giving an AI the background knowledge on the architecture and implementation details on a per need basis.

## License

Built on the shoulder of a multitude of great open source software.
[MIT License](./LICENSE.md)
