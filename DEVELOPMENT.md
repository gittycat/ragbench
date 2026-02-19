# Development Guide

This document covers the RAG server architecture, supporting services, APIs, evals, configuration, testing, and deployment. It targets both human devs and AI agents. It is brief and point like by design. An understanding of RAG and of the tech stacks used is assumed.

For frontend/UI documentation, see [FRONT_END.md](FRONT_END.md).

## Documentation

Development docs are split by topic under [`docs/dev/`](docs/dev/INDEX.md). Use `/dev-docs <topic>` in Claude Code to load relevant sections into context.

| Topic | File |
|-------|------|
| Architecture & services | [docs/dev/architecture.md](docs/dev/architecture.md) |
| Database & connection pooling | [docs/dev/database.md](docs/dev/database.md) |
| Tech stack & versions | [docs/dev/tech-stack.md](docs/dev/tech-stack.md) |
| Configuration & secrets | [docs/dev/configuration.md](docs/dev/configuration.md) |
| Eval framework & API | [docs/dev/eval-framework.md](docs/dev/eval-framework.md) |
| RAG server API reference | [docs/dev/rag-api.md](docs/dev/rag-api.md) |
| Testing | [docs/dev/testing.md](docs/dev/testing.md) |
| CI/CD & deployment | [docs/dev/cicd-deployment.md](docs/dev/cicd-deployment.md) |
| Observability & troubleshooting | [docs/dev/observability.md](docs/dev/observability.md) |
| PII masking | [docs/dev/pii-masking.md](docs/dev/pii-masking.md) |
| Development setup | [docs/dev/setup.md](docs/dev/setup.md) |

## Roadmap

### Recently Completed

- **Eval Service API** (Feb 2026): Standalone FastAPI service (port 8002) for triggering evals, tracking progress, and serving results with 5 dashboard metrics. Webapp proxy routing for `/api/eval/*`
- **PostgreSQL-backed Chat Memory** (Oct 2025): Session-based conversation history with persistent storage
- **Hybrid Search** (Oct 2025): BM25 + Vector + RRF fusion with ~48% retrieval improvement
- **Contextual Retrieval** (Oct 2025): LLM-generated chunk context with ~49% fewer retrieval failures
- **DeepEval Framework** (Dec 2025): Anthropic Claude Sonnet 4 as LLM judge with pytest integration
- **Forgejo CI/CD** (Dec 2025): Self-hosted Git + CI/CD with GitHub Actions-compatible workflows
- **Metrics & Observability API** (Dec 2025): System health monitoring and evaluation history tracking

### In Planning

- **Eval Dashboard UI**: Frontend pages for triggering evals, viewing results, and comparing runs
- **PII Masking**: Anonymize sensitive data for cloud LLM providers (see [implementation plan](docs/PII_MASKING_IMPLEMENTATION_PLAN.md))
- **Centralized Logging**: Grafana Loki + Promtail + structlog (see [implementation plan](docs/LOGGING_IMPLEMENTATION_PLAN.md))
- **Parent Document Retrieval**: Sentence window method for better context
- **Query Fusion**: Multi-query generation for improved recall

For detailed feature roadmap including implementation tasks and effort estimates, see [ROADMAP.md](docs/ROADMAP.md).

## Documentation Index

| Document | Purpose |
|----------|---------|
| [FRONT_END.md](FRONT_END.md) | Frontend/UI development |
| [CLAUDE.md](CLAUDE.md) | Project instructions for Claude Code |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Feature roadmap with tasks and effort estimates |
| [docs/FORGEJO_CI_SETUP.md](docs/FORGEJO_CI_SETUP.md) | CI/CD setup guide |
| [docs/DEEPEVAL_IMPLEMENTATION_SUMMARY.md](docs/DEEPEVAL_IMPLEMENTATION_SUMMARY.md) | Evaluation framework |
| [docs/CONVERSATIONAL_RAG.md](docs/CONVERSATIONAL_RAG.md) | Session management |
| [docs/PERFORMANCE_OPTIMIZATIONS_SUMMARY.md](docs/PERFORMANCE_OPTIMIZATIONS_SUMMARY.md) | Performance tuning |
| [docs/PHASE1_IMPLEMENTATION_SUMMARY.md](docs/PHASE1_IMPLEMENTATION_SUMMARY.md) | Phase 1 details |
| [docs/PHASE2_IMPLEMENTATION_SUMMARY.md](docs/PHASE2_IMPLEMENTATION_SUMMARY.md) | Phase 2 details |
| [docs/RAG_ACCURACY_IMPROVEMENT_PLAN_2025.md](docs/RAG_ACCURACY_IMPROVEMENT_PLAN_2025.md) | Future optimizations |
| [docs/PII_MASKING_IMPLEMENTATION_PLAN.md](docs/PII_MASKING_IMPLEMENTATION_PLAN.md) | PII masking for cloud LLMs |
| [docs/LOGGING_IMPLEMENTATION_PLAN.md](docs/LOGGING_IMPLEMENTATION_PLAN.md) | Centralized logging with Grafana Loki |
