# RAGBench

**A private, self-hostable RAG assistant that answers questions from your own trusted content — with the observability to prove the answers are good.**

RAGBench turns your organisation's documents into an AI assistant that gives accurate, grounded answers instead of the confident guesses you get from a generic chatbot. It runs entirely on your infrastructure, keeps sensitive data private even when using frontier cloud models, and ships with a built-in evaluation service so you can measure answer quality rather than hope for it.

---

## Why RAGBench

Retrieval-Augmented Generation (RAG) has become the most common way enterprises put AI to work: instead of relying on a model's training data, answers are grounded in *your* content. But most RAG deployments have two blind spots — **privacy** (your documents leave the building) and **quality** (nobody actually measures whether the answers are correct). RAGBench is built around closing both gaps.

- **Grounded answers, not hallucinations.** Every response is retrieved from and cited against your own corpus.
- **Privacy by design.** Run 100% on-premises, or use cloud models with automatic PII redaction on every outbound request.
- **Measured, not guessed.** A dedicated evaluation service scores retrieval and answer quality against public and custom benchmarks.
- **Own your stack.** Docker Compose, open-source components, no per-seat SaaS lock-in.

---

## Key Features

### 🔒 Data Privacy
- **Fully on-premises option** — run every component locally with open-source models; no request ever leaves your network.
- **Safe cloud-model usage** — when you opt for frontier models (OpenAI, Anthropic, Google, DeepSeek, Moonshot), sensitive data is anonymised before it leaves the perimeter and restored in the response.
- **Reversible PII masking** — Microsoft Presidio + spaCy detect and token-mask names, emails, and other identifiers across the query, retrieved context, chat history, and session titles, with a corpus-local guardrail and audit logging.
- **Secrets handled correctly** — API keys and DB credentials via Docker secrets mounted as files, following OWASP guidance (no secrets in environment variables or logs).

### 🎯 Retrieval Quality
- **Hybrid search** — combines sparse keyword search (BM25) and dense vector search, fused with Reciprocal Rank Fusion (RRF) for ~48% better retrieval than either method alone.
- **Contextual retrieval** — an LLM prepends document-level context to each chunk before embedding, cutting retrieval failures by ~49% with zero added query-time latency.
- **Cross-encoder reranking** — a second-stage reranker (ms-marco-MiniLM) reorders candidates so the most relevant passages reach the model.
- **Broad document support** — PDF, DOCX, PPTX, XLSX, HTML, Markdown, AsciiDoc, and plain text, parsed with Docling.

### 📊 Observability & Evaluation
- **Built-in evaluation service** — a standalone API that runs automated quality assessments against multiple datasets (RAGBench, SQuAD 2.0, QASPER, HotpotQA, MS MARCO) plus your own golden Q&A.
- **Five headline metrics** — Retrieval Relevance, Faithfulness, Answer Completeness, Answer Relevance, and Response Latency, distilled for at-a-glance dashboards.
- **LLM-as-judge scoring** — Anthropic Claude evaluates faithfulness, correctness, citation accuracy, and abstention handling, with configurable weighted scoring.
- **Run comparison & trends** — compare configurations side-by-side to find the best model/setting mix for your data and cost constraints.

### 💬 Conversational RAG
- **Persistent, session-based chat** — PostgreSQL-backed conversation history so context survives restarts.
- **Context-aware follow-ups** — `condense_plus_context` mode rewrites follow-up questions into standalone queries before retrieval.

### ⚙️ Operations
- **Async document processing** — an isolated worker claims ingestion jobs via PostgreSQL `SKIP LOCKED`, with live progress tracking.
- **Multi-provider flexibility** — swap LLM, embedding, reranker, and eval models through a single `config.yml`, no code changes.
- **Network-isolated services** — internal services run on a private Docker network; only the web app and API are exposed.

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **API & Backend** | Python 3.13, FastAPI |
| **RAG Pipeline** | LlamaIndex, Docling (document parsing) |
| **Vector Search** | ChromaDB (dense embeddings) |
| **Keyword Search** | PostgreSQL 17 + pg_textsearch (BM25) |
| **Fusion & Rerank** | Reciprocal Rank Fusion (RRF) + SentenceTransformers cross-encoder |
| **Async Processing** | PostgreSQL `SKIP LOCKED` work queue |
| **Chat & Persistence** | PostgreSQL (sessions, history, metadata) |
| **Privacy** | Microsoft Presidio + spaCy (PII detection & masking) |
| **Evaluation** | DeepEval, LLM-as-judge (Anthropic Claude) |
| **Frontend** | SvelteKit, Tailwind CSS, DaisyUI |
| **LLM Inference** | Ollama (local) or OpenAI / Anthropic / Google / DeepSeek / Moonshot (cloud) |
| **Deployment** | Docker Compose |

### Default local models
| Purpose | Model | Runs on |
|---------|-------|---------|
| Answer generation | gemma3:4b | Ollama (local) |
| Embeddings | nomic-embed-text | Ollama (local) |
| Reranking | ms-marco-MiniLM-L-6-v2 | HuggingFace (local) |
| Evaluation judge | Claude Sonnet | Anthropic (cloud) |

All defaults are swappable in `config.yml`.

---

## Deployment at a Glance

RAGBench ships as a set of Docker Compose services:

- **Web app** (SvelteKit) — upload, chat, and dashboards
- **RAG server** (FastAPI) — retrieval and answer generation
- **Task worker** — async document ingestion
- **Evaluation service** (FastAPI) — automated quality scoring
- **PostgreSQL 17** — BM25 search, chat, queue, and metadata
- **ChromaDB** — vector store
- **Ollama** (on the host) — local model inference

Minimum footprint for local development: Docker, Ollama, ~4 GB RAM, ~2 GB disk. Production on-prem deployments benefit from a GPU server sized for larger open-source models.

---

## Project Status

RAGBench is an actively developed research/reference implementation focused on **privacy** and **observability** in RAG. It is not yet hardened for production — authentication, multi-tenancy, and high-availability features are on the roadmap. It is ideal for teams evaluating self-hosted RAG architectures, benchmarking model/retrieval trade-offs, or building a privacy-first internal knowledge assistant.

*Built on the shoulders of great open-source software. Licensed under the MIT License.*
