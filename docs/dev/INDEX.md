# Development Documentation Index

| File | Topic | Description |
|------|-------|-------------|
| [architecture.md](architecture.md) | architecture, services, data flow, docker, rag pipeline, retrieval, chat | Service architecture, network isolation, data flow, document processing, retrieval strategy, chat features |
| [database.md](database.md) | database, postgres, connection pool, sqlalchemy, queries | Connection pooling config, DB access patterns, session management, file layout |
| [tech-stack.md](tech-stack.md) | tech stack, versions, models, providers, llm | Backend/frontend technology versions, AI models, LLM provider support |
| [configuration.md](configuration.md) | config, secrets, env vars, yaml, api keys | YAML config, environment variables, Docker secrets management |
| [eval-framework.md](eval-framework.md) | eval, evaluation, metrics, deepeval, dashboard, api | Eval framework, metrics, DeepEval, eval service API (port 8002), dashboard metrics, running evals |
| [rag-api.md](rag-api.md) | api, rest, endpoints, query, documents, sessions, upload | RAG server REST API reference (port 8001) — all endpoints for documents, queries, sessions, metrics |
| [testing.md](testing.md) | testing, pytest, integration, unit, markers | Test categories, structure, integration test design, pytest markers |
| [cicd-deployment.md](cicd-deployment.md) | ci, cd, forgejo, deployment, docker compose, versioning | Forgejo CI/CD setup, pipeline config, deployment environments, version management |
| [observability.md](observability.md) | observability, metrics, health, troubleshooting, logs | Metrics API, health monitoring, common issues, service logs, database reset |
| [pii-masking.md](pii-masking.md) | pii, masking, privacy, presidio, gdpr | PII detection and masking for cloud LLM providers, configuration, audit logging |
| [setup.md](setup.md) | setup, development, prerequisites, just, local | Development prerequisites, local setup, just task runner, production considerations |
