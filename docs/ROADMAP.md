# Roadmap

This document outlines planned features and enhancements for the RAG system, organized by feature area rather than timeline. Each feature includes effort estimates and is broken down into tasks suitable for single development sessions.

## Table of Contents

- [Completed Features](#completed-features)
- [Planned Features](#planned-features)
  - [Centralized Logging Infrastructure](#centralized-logging-infrastructure)
  - [Metrics Visualization](#metrics-visualization)
  - [PII Masking for Cloud LLM Providers](#pii-masking-for-cloud-llm-providers)
  - [Parent Document Retrieval](#parent-document-retrieval)
  - [Query Fusion](#query-fusion)
  - [Evaluation Dataset Expansion](#evaluation-dataset-expansion)
  - [Multi-user Support & Authentication](#multi-user-support--authentication)
- [Future Considerations](#future-considerations)

---

## Completed Features

### Redis-backed Chat Memory (Oct 2025)
- Session-based conversation history
- Persistent storage with configurable TTL
- Progress tracking for async uploads

### ChromaDB Backup/Restore (Oct 2025)
- Automated backup scripts
- Health verification
- 30-day retention policy

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

### Parent Document Retrieval

**Description:** Implement sentence window retrieval method where embeddings are generated for sentence-level chunks but larger surrounding context is returned to the LLM. Improves answer quality by providing more coherent context.

**Why Important:** Current fixed-size chunking (500 tokens) can split semantic units. Parent retrieval retrieves precise matches but provides broader context, improving answer quality by 10-20%.

**Effort Estimate:** Medium (2-3 sessions)

#### Tasks

1. **Planning & Research**
   - Review LlamaIndex SentenceWindowNodeParser
   - Define window sizes (sentence=3-5, parent=10-20 sentences)
   - Design metadata schema for parent-child relationships
   - Plan ChromaDB storage strategy

2. **Parsing & Indexing**
   - Implement SentenceWindowNodeParser integration
   - Update document processor to create parent-child node pairs
   - Modify ChromaDB storage to handle hierarchical nodes
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
   - Add caching for generated queries (Redis)
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
   - Add user context to Redis session storage
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

### Security Hardening
RAG-specific vulnerability testing: prompt injection, content injection, adversarial retrieval attacks.

### Infrastructure Monitoring & Alerting
Prometheus + Grafana metrics, alerting rules, SLA monitoring, uptime tracking.

### Support Additional File Formats
Including CSV, JSON, XLS, ...

### Multi-modal Support
Images, video, and voice in prompts and responses. Requires vision model integration and media processing pipeline.

### Large-scale Data Load Optimization
Bulk indexing, parallel processing, and incremental updates for datasets with millions of documents.

### Disaster Recovery & High Availability
Cross-region replication, automated failover, RTO/RPO targets, load balancing.

### Data Retention Policies
Document lifecycle management, automatic archival, GDPR compliance, right-to-be-forgotten.

### Performance Under High Load
Concurrent user support, horizontal scaling, caching strategies, SLA targets (P95 latency, throughput).

### Enterprise Authentication & Authorization
LDAP/IAM/SSO integration, role-based access control, audit logging, group management.
