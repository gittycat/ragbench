# Roadmap

This document outlines planned features and enhancements for the RAG system, organized by feature area rather than timeline. Each feature includes effort estimates and is broken down into tasks suitable for single development sessions.

## Table of Contents

- [Completed Features](#completed-features)
- [Planned Features](#planned-features)
  - [Centralized Logging Infrastructure](#centralized-logging-infrastructure)
  - [Metrics Visualization](#metrics-visualization)
  - [Model Leaderboard & Recommendations](#model-leaderboard--recommendations)
  - [PII Masking for Cloud LLM Providers](#pii-masking-for-cloud-llm-providers)
  - [GraphRAG](#graph-rag)
  - [Parent Document Retrieval](#parent-document-retrieval)
  - [Query Fusion](#query-fusion)
  - [Evaluation Dataset Expansion](#evaluation-dataset-expansion)
  - [Multi-user Support & Authentication](#multi-user-support--authentication)
- [Future Considerations](#future-considerations)

---

## Completed Features

### PostgreSQL-backed Chat Memory (Oct 2025)
- Session-based conversation history
- Persistent storage with no TTL
- Progress tracking for async uploads

### PostgreSQL Migration (Jan 2026)
- Consolidated vector storage, full-text search, and queueing into PostgreSQL
- pgvector for embeddings, pg_search for BM25, pgmq for async tasks

### Reranker Optimization (Oct 2025)
- Cross-encoder reranking with ms-marco-MiniLM-L-6-v2
- Pre-initialization to avoid first-query timeout
- Dynamic top-n selection (5-10 nodes)

### Hybrid Search (Oct 2025)
- BM25 + Vector retrieval with RRF fusion
- Auto-refresh after uploads/deletes
- ~48% retrieval improvement over single-method

### Contextual Retrieval (Oct 2025)
- LLM-generated chunk context before embedding
- ~49% reduction in retrieval failures
- Zero query-time overhead (adds 85% to indexing)

### DeepEval Framework Integration (Dec 2025)
- Anthropic Claude Sonnet 4 as LLM judge
- Five metric categories: Contextual Precision, Contextual Recall, Faithfulness, Answer Relevancy, Hallucination
- Pytest integration with custom markers
- Unified CLI for evaluation

### Forgejo CI/CD (Dec 2025)
- Self-hosted Git + CI/CD
- GitHub Actions-compatible workflows
- Automated testing on push/PR
- Docker build verification

### Metrics & Observability API (Dec 2025)
- System health monitoring endpoints
- Evaluation history and trend tracking
- Baseline comparison
- Configuration recommendation API

---

## Planned Features

### Centralized Logging Infrastructure

**Description:** Implement centralized logging using Grafana Loki + Promtail + Grafana with structlog for structured logs. Replaces current print-based logging with searchable, filterable log aggregation.

**Why Important:** Current logging is scattered and difficult to debug in production. Centralized logging enables troubleshooting, performance analysis, and operational visibility.

**Effort Estimate:** Medium (3-4 sessions)

**Implementation Plan:** See [LOGGING_IMPLEMENTATION_PLAN.md](LOGGING_IMPLEMENTATION_PLAN.md)

#### Tasks

1. **Planning & Architecture**
   - Review implementation plan
   - Design log schema (structured fields)
   - Plan Loki/Promtail deployment strategy
   - Define retention policies

2. **Structlog Integration**
   - Replace current logging with structlog
   - Add context processors (session_id, document_id, user_id)
   - Implement log levels and filtering
   - Add performance metrics (timing, token counts)

3. **Loki + Promtail Setup**
   - Add Loki and Promtail to docker-compose
   - Configure Promtail to scrape container logs
   - Set up Loki data source in Grafana
   - Configure retention and storage limits

4. **Grafana Dashboards**
   - Create system health dashboard
   - Create query performance dashboard
   - Create error tracking dashboard
   - Add alerting for critical errors

### Metrics Visualization

**Description:** Integrate evaluation metrics and system health monitoring into the webapp frontend. Provides visual dashboards for performance tracking and configuration tuning.

**Why Important:** Current metrics API is backend-only. Frontend visualization makes metrics accessible to non-technical users and enables data-driven configuration decisions.

**Effort Estimate:** Medium (3-4 sessions)

#### Tasks

1. **Planning & Design**
   - Review existing metrics API endpoints
   - Design dashboard layouts (system health, evaluation trends, comparison)
   - Choose charting library (Chart.js, D3, or Recharts)
   - Plan real-time update strategy (polling vs. SSE)

2. **System Health Dashboard**
   - Create component health status display
   - Add model configuration summary
   - Add retrieval pipeline settings
   - Implement auto-refresh

3. **Evaluation Metrics Dashboard**
   - Display evaluation history timeline
   - Show metric trends over time
   - Add baseline comparison view
   - Implement run-to-run comparison

4. **Configuration Tuning Interface**
   - Display recommendation API results
   - Add interactive configuration editor
   - Implement what-if scenario analysis
   - Add export/import for configurations

### Model Leaderboard & Recommendations

**Description:** Implement an intelligent model recommendation system that provides curated, up-to-date recommendations for the top-performing models across all RAG pipeline components (embeddings, reranking, inference, retrieval). Recommendations are sourced from authoritative leaderboards (MTEB, BEIR, LMSys Chatbot Arena) and continuously updated to reflect the latest releases and benchmarks.

**Why Important:** The RAG model landscape evolves rapidly, with new high-performance models released monthly. Manually tracking leaderboards across multiple categories is time-consuming and error-prone. An automated recommendation system enables users to discover and adopt state-of-the-art models quickly, ensuring optimal pipeline performance without extensive manual research.

**Effort Estimate:** Medium (3-4 sessions)

#### Tasks

1. **Planning & Data Sources**
   - Identify authoritative leaderboard sources (MTEB for embeddings, BEIR for retrieval, LMSys for LLM inference)
   - Define API integration strategy (web scraping vs. API endpoints)
   - Design recommendation schema (model name, provider, performance metrics, use case fit)
   - Plan caching strategy for leaderboard data (PostgreSQL table with daily refresh)

2. **Leaderboard Data Aggregation**
   - Implement leaderboard scrapers/API clients for each data source
   - Parse and normalize performance metrics across sources
   - Categorize models by RAG pipeline component (embeddings, reranking, inference, contextual retrieval)
   - Distinguish between open-source (self-hosted) and cloud-based (API) models
   - Extract model metadata (parameter count, weights availability, license)

3. **Recommendation Engine**
   - Build recommendation logic: rank by performance, filter by deployment type (self-hosted/cloud)
   - Return top 5 models per category with justification (accuracy, latency, cost, license)
   - Add filtering options (open-source only, cloud only, by provider, by model size)
   - Include model weights information for self-hosted models (HuggingFace repo links, size, quantization options)
   - Create API endpoint: `GET /models/recommendations?category=embeddings&deployment=open-source`

4. **Frontend Integration & Alerts**
   - Display recommendations in webapp (filterable table with performance metrics)
   - Add "Compare Current vs. Recommended" view showing potential improvements
   - Implement notification system for significant new model releases (e.g., 10%+ performance gain)
   - Add one-click configuration export for recommended models

### PII Masking for Cloud LLM Providers

PII (Personally Identifiable Information)

**Description:** Anonymize sensitive data (names, emails, SSNs, etc.) before sending to cloud LLM providers using Microsoft Presidio. Implements reversible token-based masking with validation and output guardrails.

**Why Important:** Enables safe use of cloud LLMs with sensitive documents. Critical for GDPR, HIPAA, and enterprise compliance requirements.

**Effort Estimate:** Medium (3-4 sessions)

**Implementation Plan:** See [PII_MASKING_IMPLEMENTATION_PLAN.md](PII_MASKING_IMPLEMENTATION_PLAN.md)

#### Tasks

1. **Planning & Architecture Review**
   - Review implementation plan
   - Identify integration points (query pipeline, contextual retrieval, session titles, evaluation)
   - Define configuration schema
   - Set up test fixtures

2. **Presidio Integration & Core Masking**
   - Install Microsoft Presidio dependencies
   - Implement `PIIMasker` class with entity detection
   - Create token-based masking/unmasking logic
   - Add session-scoped token mapping storage
   - Unit tests for masking operations

3. **Pipeline Integration**
   - Integrate into query pipeline (user queries, chat history, retrieved context)
   - Integrate into contextual retrieval (document chunks sent to LLM)
   - Integrate into session title generation
   - Integration tests for all data flow points

4. **Validation & Guardrails**
   - Implement token validation (detect LLM-altered tokens)
   - Add fuzzy recovery for corrupted tokens
   - Add output guardrails (scan final response for leaked PII)
   - Audit logging for masking/unmasking operations
   - End-to-end tests with real LLM calls

### Graph RAG

Using Graph db instead of or in addition to a vector db.
TODO: expand


### Parent Document Retrieval

**Description:** Implement sentence window retrieval method where embeddings are generated for sentence-level chunks but larger surrounding context is returned to the LLM. Improves answer quality by providing more coherent context.

**Why Important:** Current fixed-size chunking (500 tokens) can split semantic units. Parent retrieval retrieves precise matches but provides broader context, improving answer quality by 10-20%.

**Effort Estimate:** Medium (2-3 sessions)

#### Tasks

1. **Planning & Research**
   - Review LlamaIndex SentenceWindowNodeParser
   - Define window sizes (sentence=3-5, parent=10-20 sentences)
   - Design metadata schema for parent-child relationships
   - Plan PostgreSQL (pgvector) storage strategy

2. **Parsing & Indexing**
   - Implement SentenceWindowNodeParser integration
   - Update document processor to create parent-child node pairs
   - Modify PostgreSQL storage to handle hierarchical nodes
   - Add metadata for parent references

3. **Retrieval Pipeline**
   - Update hybrid retriever to fetch parent nodes
   - Modify reranker to work with parent context
   - Add configuration toggle (`enable_parent_retrieval`)
   - Integration tests

4. **Evaluation & Tuning**
   - Run evaluation suite with parent retrieval enabled
   - Compare metrics against baseline
   - Tune window sizes for optimal performance

### Query Fusion

**Description:** Generate multiple variations of the user's query using an LLM, retrieve context for each variation, then merge and rerank results. Improves recall by capturing different phrasings and perspectives.

**Why Important:** Users often phrase queries suboptimally. Query fusion compensates by exploring semantic variations, improving retrieval recall by 15-25%.

**Effort Estimate:** Small-Medium (2-3 sessions)

#### Tasks

1. **Planning & Design**
   - Review LlamaIndex query transformation techniques
   - Define number of query variations (3-5 typical)
   - Plan result fusion strategy (RRF or concatenation)
   - Estimate latency impact

2. **Query Generation**
   - Implement multi-query generation using LLM
   - Create prompt template for query variations
   - Add caching for generated queries (PostgreSQL table or in-memory)
   - Unit tests for query generation

3. **Retrieval & Fusion**
   - Execute parallel retrieval for all query variations
   - Implement result fusion (RRF across all retrievals)
   - Add reranking after fusion
   - Configuration toggle (`enable_query_fusion`)

4. **Evaluation & Optimization**
   - Benchmark latency impact
   - Evaluate retrieval metrics
   - Optimize number of variations vs. quality tradeoff



### Evaluation Dataset Expansion

**Description:** Expand golden Q&A dataset from current 10 pairs to 100+ pairs for production-grade confidence in evaluation results.

**Why Important:** 10 samples is insufficient for statistical confidence. 100+ samples provides reliable evaluation and catches edge cases.

**Effort Estimate:** Small (1-2 sessions + ongoing curation)

#### Tasks

1. **Synthetic Generation**
   - Use evaluation CLI to generate Q&A pairs from documents
   - Review and curate generated pairs
   - Add diversity (different question types, complexity levels)
   - Target 50 synthetic pairs

2. **Manual Curation**
   - Identify gaps in question coverage
   - Write manual Q&A pairs for edge cases
   - Add adversarial examples (trick questions, ambiguous queries)
   - Target 50 manual pairs

3. **Dataset Validation**
   - Run full evaluation suite on expanded dataset
   - Identify low-quality pairs (low agreement scores)
   - Refine ground truth context
   - Document dataset composition

### Multi-user Support & Authentication

**Description:** Add user authentication and session isolation to support multiple concurrent users. Includes user management, access control, and per-user session history.

**Why Important:** Current system is single-user. Multi-user support is required for team deployments and production use cases.

**Effort Estimate:** Large (5-6 sessions)

#### Tasks

1. **Planning & Architecture**
   - Choose auth strategy (JWT, OAuth, LDAP)
   - Design user database schema
   - Plan session isolation strategy
   - Define authorization model (RBAC or simple user/admin)

2. **User Management**
   - Implement user registration and login
   - Add password hashing (bcrypt)
   - Create user CRUD endpoints
   - Add user database (PostgreSQL or SQLite)

3. **Authentication Middleware**
   - Implement JWT token generation and validation
   - Add auth middleware to protected endpoints
   - Create login/logout flows
   - Add token refresh mechanism

4. **Session Isolation**
   - Associate sessions with user IDs
   - Update session endpoints to filter by user
   - Add user context to PostgreSQL session storage
   - Implement session access control

5. **Frontend Integration**
   - Add login/logout UI
   - Store and manage JWT tokens
   - Add user profile display
   - Implement session filtering by user

6. **Testing & Security**
   - Add auth integration tests
   - Test session isolation
   - Add rate limiting
   - Security audit (XSS, CSRF, injection)

---


## Future Considerations

These features require significant architectural decisions and SLA definitions before implementation:

### Meilisearch as Unified Search Service

The current hybrid search uses an in-memory BM25 index (via LlamaIndex BM25Retriever) that must be rebuilt on every rag-server restart and cannot be shared across services. Meilisearch is a self-hosted, open-source search engine that could replace or complement this approach:

**Advantages:**
- **Persistence**: Native disk persistence, no rebuild on restart
- **Service architecture**: Separate container queryable by both rag-server and workers
- **Hybrid search built-in**: Combines full-text + vector search with automatic score normalization (since v1.3)
- **Fast**: Sub-50ms queries, optimized for instant search
- **Typo tolerance**: Built-in fuzzy matching for user-facing search

**Trade-offs vs BM25:**
- Meilisearch uses rule-based ranking (words, typo, proximity, exactness) rather than BM25's TF-IDF formula with length normalization
- BM25 has more IR research backing for document retrieval; Meilisearch optimized for end-user search UX
- Could consolidate vector + full-text into single service, or run alongside PostgreSQL

**Decision required**: Whether to use Meilisearch for full-text only (with PostgreSQL/pgvector for vectors) or as a unified search service for both.

### Security Hardening
RAG-specific vulnerability testing: prompt injection, content injection, adversarial retrieval attacks.

### Secrets Management
We currently use environment variables to store API KEYS and passwords. In a strictly secure environment, 
a secrets management solution would need to be used instead (Hashicorp Vault, Doppler, 1Password SDK, IAM,...)
to remove any risk of secrets being leaked via app logging.

### Infrastructure Monitoring & Alerting
Prometheus + Grafana metrics, alerting rules, SLA monitoring, uptime tracking.

### Support Additional File Formats
Including CSV, JSON, XLS, ...

### Multi-modal Support
Images, video, and voice in prompts and responses. Requires vision model integration and media processing pipeline.

### Large-scale Data Load Optimization
Bulk indexing, parallel processing, and incremental updates for datasets with millions of documents.

### Multi-tenancy
Isolated data and configuration per tenant (organization or user group). Requires tenant-aware document storage, session isolation, usage quotas, and billing integration. Key decisions include shared vs. dedicated infrastructure per tenant, data isolation strategy (logical vs. physical separation), and tenant provisioning workflows.

### High Availability
Redundant service instances with automatic failover to eliminate single points of failure. Includes load balancing across multiple rag-server replicas and PostgreSQL high availability (streaming replication + failover) for vector, BM25, and queue resilience. Requires health monitoring, graceful degradation (e.g., vector-only search if BM25 unavailable), and defined SLAs for uptime targets (99.9%+).

### Disaster Recovery
Cross-region replication, automated failover, RTO/RPO targets. Includes automated backup verification, point-in-time recovery capabilities, and documented runbooks for incident response.

### Data Retention Policies
Document lifecycle management, automatic archival, GDPR compliance, right-to-be-forgotten.

### Performance Under High Load
Concurrent user support, horizontal scaling, caching strategies, SLA targets (P95 latency, throughput).

### Enterprise Authentication & Authorization
LDAP/IAM/SSO integration, role-based access control, audit logging, group management.
