# RAG Bench

A locally hosted RAG (Retrieval-Augmented Generation) system for private document research. Upload your documents, ask questions in natural language, and get AI-powered answers with source citations.

RAG Lab lets you chat with your documents using AI. It runs on your machine, keeping your data private. The system supports multiple document formats and provides source citations for every answer.

**Status**: This is a development/research project, not production-ready software. It lacks authentication, enterprise security, and high availability features.

**Goal**: Create a RAG system running locally with multiple LLM models that includes the quality evals needed to measure accuracy and speed.

## Model Options

RAG Lab supports both local and cloud AI models through a plugin architecture:

**Local (Ollama)**:  Full privacy, good with small models, Free, 4GB+ RAM
**Cloud (OpenAI, Anthropic, etc.)**: Data sent to provider, Best accuracy, Pay per use through API key

**Note**: Local models run well for basic use. To match frontier cloud models, more expensive servers with powerful GPUs will be needed to run full-weight non quantised open models.

## Requirements

- **Docker** - Docker Desktop, OrbStack, or Podman
- **Ollama** - For local AI models (optional if using cloud only)
- **4GB RAM** - For AI models
- **2GB disk** - For models and data

## Quick Start

### 1. Install Prerequisites

**macOS:**
```bash
# Install Ollama
brew install ollama

# Download AI models
ollama pull gemma3:4b
ollama pull nomic-embed-text

# Install Docker (choose one)
brew install --cask docker       # Docker Desktop
# OR
brew install orbstack            # OrbStack (faster on macOS)
```

**Linux:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download AI models
ollama pull gemma3:4b
ollama pull nomic-embed-text

# Install Docker
# Follow: https://docs.docker.com/engine/install/
```

### 2. Download and Configure

```bash
# Clone the repository
git clone https://github.com/gittycat/ragbench.git
cd ragbench

# Create configuration files from templates
cp config.yml.example config.yml
cp secrets/.env.example secrets/.env
```

### 3. Start the Application

```bash
# Start Ollama (if not already running)
ollama serve &

# Start RAG Lab
docker compose up -d
```

Open **http://localhost:8000** in your browser.

### 4. Stop the Application

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

## Configuration

### Default Setup (Local with Ollama)

The example configuration works out-of-the-box with Ollama. No changes needed for basic use.

### Using Cloud Providers

Edit `config.yml` to use cloud models:

```yaml
llm:
  provider: anthropic          # or: openai, google, deepseek, moonshot
  model: claude-sonnet-4-20250514
```

Add your API key to `secrets/.env`:
```bash
LLM_API_KEY=your-api-key-here
```

### Configuration Files

| File | Purpose |
|------|---------|
| `config.yml` | AI models, Ollama settings, and retrieval configuration |
| `secrets/.env` | API keys for cloud providers |

See `config/README.md` for detailed options.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Cannot connect to Ollama" | Run `ollama serve` or check if Ollama is running |
| Slow first response | Normal - models load on first use |
| Out of memory | Use smaller model or increase Docker memory |
| Upload fails | Check file format is supported |

### View Logs

```bash
docker compose logs -f
```

### Reset Everything

```bash
docker compose down -v
docker compose up -d
```

## Development

For development setup, testing, and technical documentation, see [DEVELOPMENT.md](DEVELOPMENT.md).

## License

[MIT License](./LICENSE.md)
