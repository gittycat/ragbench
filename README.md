## About

This projects implements a RAG AI assistant that searches your organisation’s trusted content and answers questions using it. It delivers more accurate, relevant responses grounded in your data than what is obtained by simple LLM apps like ChatGPT.

RAGs have become the most common application of AI in enterprise environments.
This specific RAG focuses on two core features: data privacy and observability.

## Data Privacy

- Option to run fully on-prem (locally). No calls outside the intranet needed.
For decent performance, this requires a dedicated server spec'd for large open source models.
- Frontier cloud models can also be used. Privacy is then ensured by performing data anynimization on any request to the cloud and inserting back the redacted data on responses.

## Observability

This covers both measures of the **quality** of the data (accuracy, completeness, groundedness / hallucination rate, relevance) and **Operational metrics** values like cost, latency and speed. This project provides dashboards that should allow the admin to determine the best combinations of LLM models used and settings for their data and organisation constraints. This is an ever improving area of RAGs, including this project.

## Tech Stack

- **Backend**: Python, FastAPI, PostgreSQL (pgvector + pg_search + pgmq)
- **RAG Pipeline**: Docling, LlamaIndex
- **Vector DB**: PostgreSQL (pgvector)
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

### 2. Download and Configure

```bash
# Clone the repository
git clone https://github.com/gittycat/ragbench.git
cd ragbench
```

### 3. Configure

**Model Selection** (`config.yml`):

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

### 4. Start the Application

```bash
# Start Ollama (if not already running)
ollama serve &

# Start RAG Lab
docker compose up -d
```

Open **http://localhost:8000** in your browser.

### 5. Stop the Application

```bash
docker compose down
```

## Usage

### Upload Documents

1. Go to **Documents** page
2. Click **Upload** and select files
3. Wait for processing (progress bar shows status)

**Supported formats**: PDF, DOCX, PPTX, XLSX, TXT, Markdown, HTML, AsciiDoc

### Ask Questions

1. Go to **Chat** page
2. Type your question
3. Get AI-powered answers with source citations

### Manage Documents

- View all uploaded documents in **Documents** page
- Delete documents you no longer need
- Start new chat sessions anytime

### Reset Everything

```bash
docker compose down -v
docker compose up -d
```

## Development

For development setup, testing, and technical documentation, see [DEVELOPMENT.md](DEVELOPMENT.md).

## License

Built on the shoulder of a multitude of great open source software.
[MIT License](./LICENSE.md)
