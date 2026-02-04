# Evaluation API Design

**Purpose**: High-level API for webapp to trigger, monitor, and retrieve evaluation results.

**Last Updated**: 2026-01-23

**Implementation Status**: Phase 1 complete, Phase 2 complete. See [Implementation Plan](#implementation-plan) for details.

---

## Implementation Status Summary

### Completed Files (Phase 1 + 2)

| File | Status | Description |
|------|--------|-------------|
| `schemas/eval.py` | ✅ Created | Pydantic models for Eval API |
| `schemas/metrics.py` | ✅ Updated | System/model schemas (eval schemas moved to eval.py) |
| `api/routes/eval.py` | ✅ Created | All `/metrics/eval/*` endpoints |
| `evals/datasets/golden.py` | ✅ Created | Golden dataset loader |
| `evals/config.py` | ✅ Updated | Added `GOLDEN` to `DatasetName` enum |
| `evals/datasets/registry.py` | ✅ Updated | Registered golden loader |
| `infrastructure/tasks/eval_progress.py` | ✅ Created | PostgreSQL progress tracking |
| `infrastructure/tasks/eval_worker.py` | ✅ Created | PGMQ task for async evaluation |
| `infrastructure/tasks/pgmq_queue.py` | ✅ Updated | Added eval queue routing |
| `services/eval/__init__.py` | ✅ Created | Consolidated eval service exports |
| `services/eval/history.py` | ✅ Created | Eval run storage/retrieval |
| `services/eval/baseline.py` | ✅ Created | Golden baseline management (functions) |
| `services/eval/comparison.py` | ✅ Created | Run comparison functions |
| `services/eval/recommendation.py` | ✅ Created | Config recommendation functions |
| `services/metrics.py` | ✅ Updated | System/model services only |
| `main.py` | ✅ Updated | Included eval router |
| `docker-compose.yml` | ✅ Updated | Added `eval-worker` service |

### Remaining Work

1. **Test SSE with frontend client** - Manual testing required
2. **Add OpenAPI response examples** - Deferred to Phase 3

### Key Implementation Notes

- **Lazy imports**: `api/routes/eval.py` and `infrastructure/tasks/eval_worker.py` use lazy imports to avoid loading heavy evaluation dependencies (HuggingFace `datasets`) during API startup
- **Schema architecture**: Two schema files - `schemas/metrics.py` for system/model configs, `schemas/eval.py` for all eval-related schemas
- **Service architecture**: Eval services consolidated in `services/eval/` with functions (not classes) per project guidelines
- **Progress tracking**: Coarse-grained (phase-level) progress updates; per-question updates deferred (PostgreSQL)

---

## Executive Summary

This document proposes an Evaluation API that enables the webapp frontend to:
1. **Discover** available evaluations (metrics and metric groups)
2. **Trigger** evaluation runs (at group level, with optional metric selection)
3. **Monitor** real-time progress during execution
4. **Retrieve** results when complete

### Recommendation: Group-Level API with Optional Metric Selection

**Rationale:**
- **Simplicity**: Users think in terms of "test retrieval quality" or "test generation quality", not individual metrics
- **Coherence**: Metrics within a group are designed to work together (e.g., Recall@K and Precision@K both need the same retrieval results)
- **Efficiency**: Running related metrics together avoids redundant RAG queries
- **Flexibility**: Allow users to select specific metrics within groups for advanced use cases

**Proposed Approach**: Expose **5 metric groups** (retrieval, generation, citation, abstention, performance) with the ability to select specific metrics within each group.

---

## API Design

### Base URL
```
/metrics/eval
```

### Endpoint Migration from Existing API

The following existing endpoints will be migrated to the new `/metrics/eval/` namespace:

| Old Endpoint | New Endpoint | Notes |
|--------------|--------------|-------|
| `GET /metrics/evaluation/history` | `GET /metrics/eval/runs` | List runs with pagination |
| `GET /metrics/evaluation/{run_id}` | `GET /metrics/eval/runs/{run_id}` | Get run details |
| `DELETE /metrics/evaluation/{run_id}` | `DELETE /metrics/eval/runs/{run_id}` | Delete/cancel run |
| `GET /metrics/evaluation/summary` | `GET /metrics/eval/summary` | Trends and best run |
| `GET /metrics/evaluation/definitions` | `GET /metrics/eval/groups` | Replaced by groups endpoint |
| `GET /metrics/baseline` | `GET /metrics/eval/baseline` | Get baseline |
| `POST /metrics/baseline/{run_id}` | `POST /metrics/eval/baseline/{run_id}` | Set baseline |
| `DELETE /metrics/baseline` | `DELETE /metrics/eval/baseline` | Clear baseline |
| `GET /metrics/compare/{a}/{b}` | `GET /metrics/eval/compare/{a}/{b}` | Compare runs |
| `GET /metrics/compare-to-baseline/{id}` | `GET /metrics/eval/compare-to-baseline/{id}` | Compare to baseline |
| `POST /metrics/recommend` | `POST /metrics/eval/recommend` | Get recommendation |

**Endpoints staying at `/metrics/`** (non-eval system info):
- `GET /metrics/system` - System overview
- `GET /metrics/models` - Model information
- `GET /metrics/retrieval` - Retrieval configuration

---

### 1. Discovery Endpoints

#### GET `/metrics/eval/groups`
List available evaluation groups with their metrics.

**Response:**
```json
{
  "groups": [
    {
      "id": "retrieval",
      "name": "Retrieval Quality",
      "description": "Measures how well the system retrieves relevant documents",
      "metrics": [
        {
          "id": "recall_at_k",
          "name": "Recall@K",
          "description": "Fraction of relevant documents in top K results",
          "parameters": {"k": [1, 3, 5, 10]},
          "requires_judge": false
        },
        {
          "id": "precision_at_k",
          "name": "Precision@K",
          "description": "Fraction of top K results that are relevant",
          "parameters": {"k": [1, 3, 5]},
          "requires_judge": false
        },
        {
          "id": "mrr",
          "name": "Mean Reciprocal Rank",
          "description": "Rank of first relevant result",
          "requires_judge": false
        },
        {
          "id": "ndcg",
          "name": "NDCG",
          "description": "Normalized Discounted Cumulative Gain",
          "parameters": {"k": 10},
          "requires_judge": false
        }
      ],
      "estimated_duration_per_sample_ms": 100,
      "requires_judge": false,
      "recommended_datasets": ["ragbench", "msmarco", "hotpotqa"]
    },
    {
      "id": "generation",
      "name": "Answer Quality",
      "description": "Evaluates generated answer quality using LLM-as-judge",
      "metrics": [
        {
          "id": "faithfulness",
          "name": "Faithfulness",
          "description": "Is the answer grounded in the retrieved context?",
          "requires_judge": true
        },
        {
          "id": "answer_correctness",
          "name": "Answer Correctness",
          "description": "Semantic match with expected answer",
          "requires_judge": true
        },
        {
          "id": "answer_relevancy",
          "name": "Answer Relevancy",
          "description": "Does the answer address the question?",
          "requires_judge": true
        }
      ],
      "estimated_duration_per_sample_ms": 3000,
      "requires_judge": true,
      "recommended_datasets": ["ragbench", "qasper", "hotpotqa"]
    },
    {
      "id": "citation",
      "name": "Citation Quality",
      "description": "Measures source attribution accuracy",
      "metrics": [
        {
          "id": "citation_precision",
          "name": "Citation Precision",
          "description": "Fraction of citations that are relevant"
        },
        {
          "id": "citation_recall",
          "name": "Citation Recall",
          "description": "Fraction of relevant passages that are cited"
        },
        {
          "id": "section_accuracy",
          "name": "Section Accuracy",
          "description": "Accuracy at document+section level"
        }
      ],
      "estimated_duration_per_sample_ms": 100,
      "requires_judge": false,
      "recommended_datasets": ["qasper"]
    },
    {
      "id": "abstention",
      "name": "Abstention Quality",
      "description": "Handles unanswerable question detection",
      "metrics": [
        {
          "id": "unanswerable_accuracy",
          "name": "Unanswerable Accuracy",
          "description": "Correct abstention on unanswerable questions"
        },
        {
          "id": "false_positive_rate",
          "name": "False Positive Rate",
          "description": "Incorrectly abstaining on answerable questions"
        },
        {
          "id": "false_negative_rate",
          "name": "False Negative Rate",
          "description": "Incorrectly answering unanswerable questions (hallucination risk)"
        }
      ],
      "estimated_duration_per_sample_ms": 100,
      "requires_judge": false,
      "recommended_datasets": ["squad_v2"]
    },
    {
      "id": "performance",
      "name": "Performance Metrics",
      "description": "Latency and cost tracking",
      "metrics": [
        {
          "id": "latency_p50",
          "name": "Latency P50",
          "description": "Median query latency"
        },
        {
          "id": "latency_p95",
          "name": "Latency P95",
          "description": "95th percentile latency"
        },
        {
          "id": "cost_per_query",
          "name": "Cost Per Query",
          "description": "Cost in USD per query"
        }
      ],
      "estimated_duration_per_sample_ms": 0,
      "requires_judge": false,
      "recommended_datasets": ["ragbench"]
    }
  ]
}
```

#### GET `/metrics/eval/datasets`
List available evaluation datasets.

**Response:**
```json
{
  "datasets": [
    {
      "id": "ragbench",
      "name": "RAGBench",
      "description": "Cross-domain RAG evaluation benchmark",
      "size": 10000,
      "domains": ["technology", "science", "history", "finance"],
      "primary_aspects": ["generation", "retrieval"],
      "requires_download": true,
      "download_size_mb": 50
    },
    {
      "id": "qasper",
      "name": "Qasper",
      "description": "Long-document Q&A with citation requirements",
      "size": 5049,
      "domains": ["academic papers"],
      "primary_aspects": ["citation", "generation"],
      "requires_download": true,
      "download_size_mb": 200
    },
    {
      "id": "squad_v2",
      "name": "SQuAD v2",
      "description": "Reading comprehension with unanswerable questions",
      "size": 11873,
      "domains": ["wikipedia"],
      "primary_aspects": ["abstention"],
      "requires_download": true,
      "download_size_mb": 40
    },
    {
      "id": "hotpotqa",
      "name": "HotpotQA",
      "description": "Multi-hop reasoning questions",
      "size": 7405,
      "domains": ["wikipedia"],
      "primary_aspects": ["retrieval", "generation"],
      "requires_download": true,
      "download_size_mb": 600
    },
    {
      "id": "msmarco",
      "name": "MS MARCO",
      "description": "Large-scale retrieval benchmark",
      "size": 101093,
      "domains": ["web"],
      "primary_aspects": ["retrieval"],
      "requires_download": true,
      "download_size_mb": 1000
    },
    {
      "id": "golden",
      "name": "Golden Dataset (Local)",
      "description": "Curated Q&A pairs from your indexed documents",
      "size": 10,
      "domains": ["user documents"],
      "primary_aspects": ["generation", "retrieval"],
      "requires_download": false,
      "download_size_mb": 0
    }
  ]
}
```

---

### 2. Execution Endpoints

#### POST `/metrics/eval/runs`
Start a new evaluation run.

**Request:**
```json
{
  "name": "Hybrid Search Evaluation",
  "groups": ["retrieval", "generation"],
  "metrics": {
    "retrieval": ["recall_at_k", "precision_at_k", "mrr"],
    "generation": ["faithfulness", "answer_correctness"]
  },
  "datasets": ["ragbench"],
  "samples_per_dataset": 50,
  "judge": {
    "enabled": true,
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514"
  },
  "seed": 42
}
```

**Notes:**
- `groups`: Required. List of metric groups to run.
- `metrics`: Optional. If omitted, all metrics in selected groups run. If specified, only listed metrics run.
- `samples_per_dataset`: Optional. Defaults to 100. Use smaller values (5-20) for quick tests.
- `judge`: Optional. Required if any group needs LLM-as-judge (generation).

**Response:**
```json
{
  "run_id": "abc12345",
  "status": "pending",
  "created_at": "2026-01-21T10:30:00Z",
  "estimated_duration_seconds": 300,
  "config": {
    "groups": ["retrieval", "generation"],
    "datasets": ["ragbench"],
    "samples_per_dataset": 50,
    "total_samples": 50,
    "judge_enabled": true
  }
}
```

#### GET `/metrics/eval/runs/{run_id}`
Get run details and current status.

**Response (In Progress):**
```json
{
  "run_id": "abc12345",
  "name": "Hybrid Search Evaluation",
  "status": "running",
  "created_at": "2026-01-21T10:30:00Z",
  "progress": {
    "phase": "querying",
    "total_questions": 50,
    "completed_questions": 23,
    "current_dataset": "ragbench",
    "percent_complete": 46,
    "metrics_computed": [],
    "metrics_pending": ["recall_at_k", "precision_at_k", "mrr", "faithfulness", "answer_correctness"]
  },
  "config": {
    "groups": ["retrieval", "generation"],
    "datasets": ["ragbench"],
    "samples_per_dataset": 50
  }
}
```

**Response (Completed):**
```json
{
  "run_id": "abc12345",
  "name": "Hybrid Search Evaluation",
  "status": "completed",
  "created_at": "2026-01-21T10:30:00Z",
  "completed_at": "2026-01-21T10:35:00Z",
  "duration_seconds": 300,
  "progress": {
    "phase": "completed",
    "total_questions": 50,
    "completed_questions": 50,
    "percent_complete": 100,
    "metrics_computed": ["recall_at_k", "precision_at_k", "mrr", "faithfulness", "answer_correctness"],
    "metrics_pending": []
  },
  "config": {
    "llm_model": "gemma3:4b",
    "llm_provider": "ollama",
    "embedding_model": "nomic-embed-text",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "hybrid_search_enabled": true,
    "contextual_retrieval_enabled": false
  },
  "results": {
    "weighted_score": 0.78,
    "groups": {
      "retrieval": {
        "average": 0.82,
        "metrics": [
          {"name": "recall_at_1", "value": 0.72, "sample_size": 50},
          {"name": "recall_at_3", "value": 0.84, "sample_size": 50},
          {"name": "recall_at_5", "value": 0.88, "sample_size": 50},
          {"name": "recall_at_10", "value": 0.92, "sample_size": 50},
          {"name": "precision_at_1", "value": 0.72, "sample_size": 50},
          {"name": "precision_at_3", "value": 0.65, "sample_size": 50},
          {"name": "precision_at_5", "value": 0.58, "sample_size": 50},
          {"name": "mrr", "value": 0.79, "sample_size": 50}
        ]
      },
      "generation": {
        "average": 0.74,
        "metrics": [
          {"name": "faithfulness", "value": 0.85, "sample_size": 50},
          {"name": "answer_correctness", "value": 0.68, "sample_size": 50}
        ]
      }
    },
    "performance": {
      "latency_p50_ms": 1250,
      "latency_p95_ms": 2340,
      "latency_avg_ms": 1380,
      "cost_total_usd": 0.0
    }
  },
  "question_count": 50,
  "error_count": 2
}
```

#### GET `/metrics/eval/runs/{run_id}/progress`
Server-Sent Events (SSE) endpoint for real-time progress updates.

**Event Stream:**
```
event: progress
data: {"phase": "loading", "message": "Loading dataset ragbench..."}

event: progress
data: {"phase": "querying", "completed": 1, "total": 50, "percent": 2}

event: progress
data: {"phase": "querying", "completed": 2, "total": 50, "percent": 4}

...

event: progress
data: {"phase": "computing_metrics", "group": "retrieval", "metric": "recall_at_k"}

event: progress
data: {"phase": "computing_metrics", "group": "generation", "metric": "faithfulness"}

event: complete
data: {"run_id": "abc12345", "weighted_score": 0.78}

event: error
data: {"error": "Connection to RAG server failed", "recoverable": false}
```

#### DELETE `/metrics/eval/runs/{run_id}`
Cancel a running evaluation or delete a completed run.

**Response:**
```json
{
  "run_id": "abc12345",
  "status": "cancelled",
  "message": "Evaluation run cancelled"
}
```

#### GET `/metrics/eval/runs`
List evaluation runs with pagination.

**Query Parameters:**
- `limit`: Max runs to return (default: 20)
- `offset`: Pagination offset
- `status`: Filter by status (pending, running, completed, failed, cancelled)

**Response:**
```json
{
  "runs": [
    {
      "run_id": "abc12345",
      "name": "Hybrid Search Evaluation",
      "status": "completed",
      "created_at": "2026-01-21T10:30:00Z",
      "completed_at": "2026-01-21T10:35:00Z",
      "weighted_score": 0.78,
      "groups": ["retrieval", "generation"],
      "datasets": ["ragbench"],
      "question_count": 50
    }
  ],
  "total": 15,
  "limit": 20,
  "offset": 0
}
```

---

### 3. Analysis Endpoints

#### GET `/metrics/eval/summary`
Get evaluation trends and summary statistics.

**Response:**
```json
{
  "latest_run": { ... },
  "total_runs": 15,
  "metric_trends": [
    {
      "metric_name": "faithfulness",
      "values": [0.72, 0.75, 0.78, 0.85],
      "timestamps": ["2026-01-18T...", "2026-01-19T...", "2026-01-20T...", "2026-01-21T..."],
      "trend_direction": "improving",
      "latest_value": 0.85,
      "average_value": 0.775
    }
  ],
  "best_run": { ... },
  "configuration_impact": { ... }
}
```

#### GET `/metrics/eval/baseline`
Get current golden baseline.

#### POST `/metrics/eval/baseline/{run_id}`
Set a run as the golden baseline.

#### DELETE `/metrics/eval/baseline`
Clear the golden baseline.

#### GET `/metrics/eval/compare/{run_a}/{run_b}`
Compare two evaluation runs.

#### GET `/metrics/eval/compare-to-baseline/{run_id}`
Compare a run against the baseline.

#### POST `/metrics/eval/recommend`
Get recommended configuration based on evaluation history.

**Query Parameters:**
- `accuracy_weight`: Weight for accuracy (default: 0.5)
- `speed_weight`: Weight for speed (default: 0.3)
- `cost_weight`: Weight for cost (default: 0.2)

---

## Architecture

### Async Execution with PGMQ

Evaluation runs are executed asynchronously via PGMQ using a **dedicated queue** (`evals`) to isolate evaluation workloads from document processing.

```
┌─────────────────┐  POST /metrics/eval/runs  ┌─────────────────┐
│     Webapp      │──────────────────────────▶│    RAG Server   │
│   (Frontend)    │                            │     (FastAPI)   │
└────────┬────────┘                            └────────┬────────┘
         │                                              │
         │ SSE /progress                                │ queue task (eval queue)
         │                                              ▼
         │                                     ┌─────────────────┐
         │                                     │  PostgreSQL     │
         │                                     │   (pgmq)        │
         │                                     └────────┬────────┘
         │                                              │
         │                                              ▼
         │                                     ┌─────────────────┐
         │◀──────────progress updates──────────│  PGMQ Worker    │
         │                                     │  (eval queue)   │
         │                                     └────────┬────────┘
         │                                              │
         │                                              │ query
         │                                              ▼
         │                                     ┌─────────────────┐
         │                                     │   RAG Server    │
         │                                     │    /query       │
         │                                     └─────────────────┘
```

### PGMQ Queue Configuration

```python
from pgmq import PGMQueue

queue = PGMQueue()
QUEUE_NAME = "evals"

msg_id: int = queue.send(QUEUE_NAME, {"run_id": run_id, "dataset": dataset})
```

**Worker startup:**
```bash
# Document worker
.venv/bin/python -m infrastructure.tasks.pgmq_worker

# Evaluation worker (planned)
.venv/bin/python -m infrastructure.tasks.pgmq_eval_worker
```

### Progress Tracking

Progress is stored in PostgreSQL with the following structure:

```json
{
  "run_id": "abc12345",
  "status": "running",
  "phase": "querying",
  "created_at": "2026-01-21T10:30:00Z",
  "total_questions": 50,
  "completed_questions": 23,
  "current_dataset": "ragbench",
  "current_question_id": "q-42",
  "metrics_computed": ["recall_at_k"],
  "metrics_pending": ["precision_at_k", "mrr", "faithfulness"],
  "errors": [],
  "last_updated": "2026-01-21T10:32:15Z"
}
```

Primary key: `eval_runs.id`
Retention: Keep indefinitely (or add a cleanup job)

### Result Storage

Results are stored as JSON files in `data/eval_runs/` (same as current CLI implementation). This ensures:
- Consistency between CLI and API runs
- Easy backup and migration
- Human-readable format for debugging

---

## Metric Groups Mapping

### Current evals Metrics → New Groups

The existing `evals/config.py` `MetricConfig` will be refactored to align with the new group structure:

| Current MetricConfig Flag | New Group | Metrics |
|---------------------------|-----------|---------|
| `retrieval=True` | **retrieval** | recall_at_k, precision_at_k, mrr, ndcg |
| `generation=True` | **generation** | faithfulness, answer_correctness, answer_relevancy |
| `citation=True` | **citation** | citation_precision, citation_recall, section_accuracy |
| `abstention=True` | **abstention** | unanswerable_accuracy, false_positive_rate, false_negative_rate |
| (new) | **performance** | latency_p50, latency_p95, cost_per_query |

### Implementation Notes

1. **MetricConfig Refactor**: Update `evals/config.py` to use group-based structure
2. **Backward Compatibility**: Keep CLI flags working, map to new groups internally
3. **Scoring Weights**: Maintain existing `DEFAULT_WEIGHTS` but organize by group

---

## Golden Dataset

### Current Implementation

The golden dataset uses the existing `evals/data/golden_qa.json` file containing 10 curated Q&A pairs.

### Dataset Loader

Create `evals/datasets/golden.py`:

```python
class GoldenDatasetLoader(BaseDatasetLoader):
    """Loader for local golden Q&A dataset."""

    GOLDEN_PATH = Path("evals/data/golden_qa.json")

    def load(self, split: str = "test", max_samples: int | None = None) -> EvalDataset:
        with open(self.GOLDEN_PATH) as f:
            data = json.load(f)
        # Convert to EvalDataset format
        ...
```

### TODO: Configuration Options

> **Future Enhancement**: Make the golden dataset path configurable. Options to evaluate:
> - YAML config file (`config/eval.yml` with `golden_dataset_path`)
> - API endpoint to set path (`POST /metrics/eval/config`)
> - Environment variable (`GOLDEN_DATASET_PATH`)
> - Upload via API (`POST /metrics/eval/datasets/golden`)
>
> Decision needed on: single file vs directory, format validation, user permissions.

---

## Implementation Plan

### Phase 1: Core API (MVP)

#### 1.1 API Route Structure
- [x] Create `api/routes/eval.py` with new router
- [x] Update `main.py` to include eval router at `/metrics/eval`
- [x] Remove/deprecate old `/metrics/evaluation/*` routes (replaced by `/metrics/eval/*`)
- [x] Remove/deprecate old `/metrics/baseline/*` routes (replaced by `/metrics/eval/baseline/*`)
- [x] Remove/deprecate old `/metrics/compare/*` routes (replaced by `/metrics/eval/compare/*`)
- [x] Remove/deprecate old `/metrics/recommend` route (replaced by `/metrics/eval/recommend`)

#### 1.2 Discovery Endpoints
- [x] Implement `GET /metrics/eval/groups` - return metric group definitions
- [x] Implement `GET /metrics/eval/datasets` - return dataset definitions
- [x] Create Pydantic schemas: `MetricGroupResponse`, `DatasetResponse`
- [x] Add dataset metadata (size, domains, aspects) to registry

#### 1.3 Golden Dataset Loader
- [x] Create `evals/datasets/golden.py` with `GoldenDatasetLoader`
- [x] Register golden dataset in `evals/datasets/registry.py`
- [x] Add `DatasetName.GOLDEN` enum value
- [x] Test loader with existing `evals/data/golden_qa.json`

#### 1.4 PGMQ Eval Task
- [x] Create `infrastructure/tasks/eval_worker.py` with `run_evaluation_task`
- [x] Configure `eval` queue in `celery_app.py`
- [x] Update docker-compose to start worker with `-Q eval` flag
- [ ] Adapt `EvaluationRunner` to accept progress callback (deferred - using coarse phase updates)
- [x] Implement progress callback that writes to PostgreSQL

#### 1.5 Run Management Endpoints
- [x] Implement `POST /metrics/eval/runs` - start evaluation, return run_id
- [x] Implement `GET /metrics/eval/runs/{run_id}` - get run status/results
- [x] Implement `GET /metrics/eval/runs` - list runs with pagination
- [x] Implement `DELETE /metrics/eval/runs/{run_id}` - delete completed run
- [x] Create Pydantic schemas: `EvalRunRequest`, `EvalRunResponse`, `EvalRunListResponse`

#### 1.6 Progress Tracking
- [x] Create `infrastructure/tasks/eval_progress.py` with PostgreSQL helpers
- [x] Implement `create_eval_run()`, `update_eval_progress()`, `get_eval_progress()`
- [x] Implement `GET /metrics/eval/runs/{run_id}/progress` SSE endpoint
- [ ] Test SSE with frontend client

#### 1.7 Metric Groups Refactor
- [x] Update `evals/config.py` `MetricConfig` to use group structure (already had group flags)
- [x] Create `MetricGroup` enum: retrieval, generation, citation, abstention, performance (already exists in evals/schemas)
- [x] Update `EvaluationRunner` to compute metrics by group (already does this)
- [x] Ensure CLI backward compatibility (maintained)

---

### Phase 2: Dashboard Integration

#### 2.1 Analysis Endpoints Migration
- [x] Migrate `GET /metrics/evaluation/summary` → `GET /metrics/eval/summary`
- [x] Migrate baseline endpoints to `/metrics/eval/baseline/*`
- [x] Migrate compare endpoints to `/metrics/eval/compare/*`
- [x] Migrate recommend endpoint to `/metrics/eval/recommend`
- [x] Update all service layer calls

#### 2.2 Pydantic Schema Refinement
- [x] Review and consolidate `schemas/metrics.py` with new eval schemas
- [x] Fix schema imports (services were importing from wrong schema file)
- [x] Add request validation for all endpoints (using Pydantic)
- [ ] Add response examples for OpenAPI docs (deferred)
- [x] Ensure all nested objects have proper schemas

#### 2.3 OpenAPI Documentation
- [x] Add detailed descriptions to all endpoints (docstrings)
- [ ] Add request/response examples (deferred)
- [x] Document error responses (EvalErrorResponse schema)
- [ ] Generate TypeScript types for frontend (skipped per decision)

#### 2.4 Error Handling
- [x] Implement consistent error response format (EvalErrorResponse)
- [x] Add validation for dataset names, metric groups (in start_evaluation_run)
- [x] Add judge requirement validation (error if generation group without judge)
- [x] Add helpful error messages with resolution hints

#### 2.5 Service Layer Cleanup
- [x] Consolidate eval-related services under `services/eval/`
  - Created `services/eval/__init__.py` with clean exports
  - Created `services/eval/history.py` for run storage/retrieval
  - Created `services/eval/baseline.py` (refactored from class to functions)
  - Created `services/eval/comparison.py` for run comparison
  - Created `services/eval/recommendation.py` for config recommendations
- [x] Remove duplicate code between CLI and API paths
- [x] Ensure `EvaluationRunner` is the single source of truth (already was)

---

### Phase 3: Advanced Features

#### 3.1 Run Cancellation
- [ ] Add `cancel_evaluation_task()` using pgmq delete + status update
- [ ] Implement graceful shutdown (finish current question, skip remaining)
- [ ] Update progress to "cancelled" status
- [ ] Clean up partial results

#### 3.2 Concurrent Run Limits
- [ ] Add check for running evaluations before starting new one
- [ ] Return 409 Conflict if evaluation already running
- [ ] Add config option for max concurrent evals (default: 1)

#### 3.3 Export API
- [ ] Implement `GET /metrics/eval/runs/{run_id}/export`
- [ ] Support formats: JSON, CSV, Markdown
- [ ] Include full results with per-question breakdown
- [ ] Add download headers for file response

#### 3.4 Batch Execution
- [ ] Implement `POST /metrics/eval/batch` for multiple configs
- [ ] Queue multiple evaluation tasks
- [ ] Track batch progress across all runs
- [ ] Compare results when batch completes

#### 3.5 Webhooks (Optional)
- [ ] Add webhook URL to run request
- [ ] POST to webhook on completion/failure
- [ ] Include run summary in webhook payload

---

## Group vs Individual Metric Control

### Recommendation: Group-Level with Optional Metric Selection

**Why Group-Level?**

1. **User Mental Model**: End users think "I want to test retrieval quality" not "I want Recall@5 and NDCG@10"

2. **Metric Dependencies**:
   - All retrieval metrics need the same retrieved documents
   - All generation metrics need the same RAG responses
   - Running them together is more efficient

3. **Dashboard Simplicity**:
   - 5 toggle switches (one per group) vs 15+ individual checkboxes
   - Easier to show aggregate scores per group

4. **Sensible Defaults**: Most users want all metrics in a group

**When to Allow Individual Metric Selection?**

1. **Advanced Users**: Power users may want specific metrics
2. **Quick Tests**: Skip expensive LLM-judge metrics for rapid iteration
3. **Focused Analysis**: Only care about recall, not precision

**Implementation**: Allow both via the `metrics` field in the request:
- Omit `metrics` → run all metrics in selected groups
- Include `metrics` → run only specified metrics within each group

---

## Error Handling

| HTTP Code | Scenario |
|-----------|----------|
| 400 | Invalid request (bad dataset name, invalid metric) |
| 404 | Run not found |
| 409 | Run already exists / evaluation already running |
| 422 | Configuration error (judge required but not enabled) |
| 500 | Internal error (RAG server unavailable) |
| 503 | Service unavailable (PGMQ worker down) |

**Error Response Format:**
```json
{
  "error": "validation_error",
  "message": "Generation metrics require LLM judge to be enabled",
  "details": {
    "field": "judge.enabled",
    "required_by": ["faithfulness", "answer_correctness", "answer_relevancy"]
  }
}
```

---

## Security Considerations

1. **Rate Limiting**: Prevent DoS via excessive evaluation runs
2. **Cost Control**: Warn/limit when judge metrics would incur high API costs
3. **Resource Limits**: Cap concurrent runs and samples per run
4. **Input Validation**: Sanitize run names and metadata

---

## Example Dashboard Workflows

### Quick Test
```
User selects: retrieval group, ragbench dataset, 10 samples
→ ~10 seconds, no API costs
→ Shows Recall@K, Precision@K, MRR, NDCG
```

### Full Evaluation
```
User selects: all groups, ragbench + qasper, 100 samples each
→ ~5-10 minutes, ~$0.50 API costs (judge)
→ Complete scorecard with trends vs baseline
```

### A/B Test Workflow
```
1. Run eval with current config → save as baseline
2. Change config (enable hybrid search)
3. Run eval again
4. Compare to baseline → show improvement/regression
```

---

## Open Questions

1. ~~**Should we support custom datasets?**~~ → Yes, golden dataset implemented. Future: configurable path.
2. ~~**Should we expose run cancellation?**~~ → Yes, Phase 3 with graceful shutdown.
3. **Should we add webhooks for completion notification?** → Optional in Phase 3.
4. **Should we support scheduled/recurring evaluations?** → Deferred to future release.

---

## File Structure After Implementation

```
services/rag_server/
├── api/routes/
│   ├── eval.py                    # All /metrics/eval/* endpoints
│   └── metrics.py                 # Only /metrics/system, /models, /retrieval
├── schemas/
│   ├── eval.py                    # Eval API schemas (discovery, execution, results, analysis)
│   └── metrics.py                 # System/model schemas (ConfigSnapshot, LatencyMetrics, etc.)
├── services/
│   ├── eval/                      # Consolidated eval service layer
│   │   ├── __init__.py            # Clean exports for all eval functions
│   │   ├── history.py             # Run storage/retrieval, metric definitions
│   │   ├── baseline.py            # Golden baseline management (functions)
│   │   ├── comparison.py          # Run comparison functions
│   │   └── recommendation.py      # Config recommendation functions
│   └── metrics.py                 # System/model info services only
├── infrastructure/tasks/
│   ├── celery_app.py              # Eval queue config
│   ├── eval_worker.py             # Evaluation PGMQ task
│   └── eval_progress.py           # PostgreSQL progress helpers
├── evals/
│   ├── config.py                  # Group-based MetricConfig
│   └── datasets/
│       ├── registry.py            # Dataset registry with golden
│       └── golden.py              # Golden dataset loader
└── evals/data/
    └── golden_qa.json             # Golden Q&A pairs
```
