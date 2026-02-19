# Technology Stack

## Backend Services

| Component | Technology | Version |
|-----------|------------|---------|
| API Framework | FastAPI | 0.118+ |
| Python | Python | 3.13+ |
| Package Manager | uv | Latest |
| Vector Database | ChromaDB | Latest |
| Full-text Search | pg_textsearch (Timescale) | 0.5+ |
| Database | PostgreSQL | 17+ |
| Task Queue | PostgreSQL SKIP LOCKED | — |
| Document Parser | Docling | 2.53+ |
| RAG Framework | LlamaIndex | 0.14+ |
| Reranker | SentenceTransformers | 5.1+ |

## Frontend Services

| Component | Technology | Version |
|-----------|------------|---------|
| Framework | SvelteKit | 2.49+ |
| UI Library | DaisyUI | 5.5+ |
| CSS | Tailwind CSS | 4.1+ |
| Runtime | Node.js | 22+ |

## AI Models

| Purpose | Model | Provider | Size |
|---------|-------|----------|------|
| LLM | gemma3:4b | Ollama | 4B params |
| Embeddings | nomic-embed-text | Ollama | 137M params |
| Reranker | ms-marco-MiniLM-L-6-v2 | HuggingFace | 22M params |
| Evaluation | claude-sonnet-4-20250514 | Anthropic | Cloud |

## LLM Provider Support

The system supports multiple LLM providers via factory pattern:
- **Ollama** (default, local)
- **OpenAI**
- **Anthropic**
- **Google Gemini**
- **DeepSeek**
- **Moonshot**

Provider selection via `config.yml`.
