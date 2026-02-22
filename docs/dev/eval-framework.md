# Evaluation Framework

## Why Evaluate RAG Systems

RAG systems combine two failure-prone components: retrieval (finding relevant context) and generation (producing accurate answers). Evaluation ensures:
- **Retrieval Quality**: Are we finding the right chunks?
- **Answer Quality**: Is the generated response accurate and relevant?
- **Safety**: Are we hallucinating or producing unsupported claims?

Without systematic evaluation, configuration changes (chunk size, top-k, reranking) are guesswork.

## Evaluation Approach

**Test Dataset**: Golden Q&A pairs (question, expected answer, ground truth context)
- Current: 10 pairs from Paul Graham essays
- Target: 100+ pairs for production confidence
- Location: `evals/data/golden_qa.json`

**Evaluation Types**:
- **Retrieval Metrics**: Measure if correct chunks are retrieved
- **Generation Metrics**: Measure answer accuracy and relevance
- **Safety Metrics**: Detect hallucinations and unsupported claims

**Public Datasets**: Five additional datasets available for comprehensive evaluation (retrieval, generation, citation, abstention). See [docs/RAG_EVALUATION_DATASETS.md](../RAG_EVALUATION_DATASETS.md).

## Framework: DeepEval

**Why DeepEval**: Migrated from RAGAS (2025-12-07) for better CI/CD integration and pytest compatibility.

**LLM Judge**: Claude Sonnet 4 (Anthropic) - evaluates retrieval relevance, answer faithfulness, and hallucination detection.

**Integration**:
- Pytest integration with custom markers (`@pytest.mark.eval`)
- CLI tool for standalone evaluation
- CI/CD compatible (optional eval tests on demand)
- Results stored in `evals/data/runs/` for metrics API

## Metrics & Thresholds

**Retrieval Metrics**:
- **Contextual Precision** (threshold: 0.7): Are retrieved chunks relevant to the query?
- **Contextual Recall** (threshold: 0.7): Did we retrieve all information needed to answer?

**Generation Metrics**:
- **Faithfulness** (threshold: 0.7): Is the answer grounded in retrieved context?
- **Answer Relevancy** (threshold: 0.7): Does the answer address the question?

**Safety Metrics**:
- **Hallucination** (threshold: 0.5): Rate of claims not supported by context

Higher scores are better (except hallucination - lower is better).

## Running Evaluations

**Prerequisites**: `ANTHROPIC_API_KEY` secret file, Docker services running.

**Via API** (recommended for webapp integration):
```bash
# Trigger a run
curl -X POST http://localhost:8002/eval/runs \
  -H 'Content-Type: application/json' \
  -d '{"tier": "generation", "datasets": ["ragbench"], "samples": 5}'

# Poll progress
curl http://localhost:8002/eval/runs/active

# View results
curl http://localhost:8002/eval/runs

# Dashboard summary
curl http://localhost:8002/eval/dashboard
```

**Via CLI** (inside the running evals container):
```bash
# Quick evaluation (5 samples)
docker compose exec evals .venv/bin/python -m evals.cli eval --tier generation --datasets ragbench --samples 5

# Full evaluation
docker compose exec evals .venv/bin/python -m evals.cli eval --tier end_to_end --datasets ragbench

# List datasets
docker compose exec evals .venv/bin/python -m evals.cli datasets
```

**Via just**:
```bash
just test-eval              # Quick end-to-end (5 samples)
just test-eval-generation   # Tier 1 generation test
just test-eval-end-to-end   # Tier 2 end-to-end test
just test-eval-full         # Full dataset
just eval-datasets          # List datasets
just eval-compare id1 id2   # Compare runs
```

**CI/CD**: Evaluation tests are optional (expensive, ~2-5min). Trigger via commit message containing `[eval]` or manual workflow dispatch.

## Eval Service API (port 8002)

The eval service runs as a standalone FastAPI app. The webapp proxies `/api/eval/*` to it.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/eval/runs` | Trigger eval run (202 / 409 if busy) |
| GET | `/eval/runs/active` | Current job progress (null if idle) |
| DELETE | `/eval/runs/active` | Cancel running job |
| GET | `/eval/runs` | List completed runs (paginated) |
| GET | `/eval/runs/{run_id}` | Full run detail with scorecard |
| GET | `/eval/runs/compare?ids=a,b` | Compare runs with metric deltas |
| GET | `/eval/dashboard` | Latest run + active job summary |
| GET | `/eval/datasets` | Available datasets with tier support |
| GET | `/health` | Health check |

**Dashboard Metrics** (computed on-the-fly from scorecard):

| Metric | Scale | Source |
|--------|-------|--------|
| Retrieval Relevance | 0-1 | avg(recall@5, mrr) — null for generation tier |
| Faithfulness | 0-1 | faithfulness (LLM judge) |
| Answer Completeness | 0-1 | answer_correctness (LLM judge) |
| Answer Relevance | 0-1 | answer_relevancy (LLM judge) |
| Response Latency | seconds | latency_p50_ms / latency_p95_ms |

**Trigger request:**
```json
{
  "tier": "generation",
  "datasets": ["ragbench"],
  "samples": 20,
  "seed": 42,
  "judge_enabled": true
}
```

**Design decisions:**
- One job at a time (evals are resource-intensive)
- No database — JSON files on disk, in-memory index rebuilt on startup
- Polling via `GET /eval/runs/active` (every 2-3s during a run)
- Background `threading.Thread` runs `asyncio.run()` over the async runner
- Async parallelization: RAG queries and LLM judge calls run concurrently via `asyncio.gather()` + `Semaphore`
- Concurrency controlled by `query_concurrency` (default 10) and `judge_concurrency` (default 10) in `EvalConfig`
- Progress callback + cancellation via `threading.Event`

## Legacy Evaluation API Endpoints (rag-server, port 8001)

**Core Endpoints**:
- `GET /metrics/evaluation/definitions`: Metric descriptions and thresholds
- `GET /metrics/evaluation/history`: Past evaluation runs
- `GET /metrics/evaluation/summary`: Latest run with trend analysis
- `GET /metrics/evaluation/{run_id}`: Specific run details
- `DELETE /metrics/evaluation/{run_id}`: Delete evaluation run

**Baseline & Comparison**:
- `GET /metrics/baseline`: Get golden baseline run
- `POST /metrics/baseline/{run_id}`: Set run as golden baseline
- `DELETE /metrics/baseline`: Clear baseline
- `GET /metrics/compare/{run_a}/{run_b}`: Compare two runs
- `GET /metrics/compare-to-baseline/{run_id}`: Compare run to baseline

**Configuration Tuning**:
- `POST /metrics/recommend`: Get configuration recommendation based on evaluation history

## Research References

- [Evidently AI - RAG Evaluation Guide](https://www.evidentlyai.com/llm-guide/rag-evaluation)
- [Braintrust - RAG Evaluation Tools 2025](https://www.braintrust.dev/articles/best-rag-evaluation-tools)
- [Patronus AI - RAG Best Practices](https://www.patronus.ai/llm-testing/rag-evaluation-metrics)
