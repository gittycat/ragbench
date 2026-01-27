# Evaluation Framework

Standalone evaluation framework for measuring RAG server quality. Sends questions from benchmark datasets to the running RAG server over HTTP, compares responses against ground truth, and produces scored results grouped by metric category.

## Metric Groups

Results are organized into five groups. These groups are used throughout the codebase and in the API response when reporting evaluation results.

### Retrieval

Measures how well the retriever finds the right chunks. All compare retrieved chunks against gold (ground truth) passages using `chunk_id` matching.

| Metric | What it measures | Range |
|---|---|---|
| `recall_at_k` | Fraction of gold chunks found in top K results | 0-1, higher is better |
| `precision_at_k` | Fraction of top K results that are gold chunks | 0-1, higher is better |
| `mrr` | Reciprocal rank of the first relevant result | 0-1, higher is better |
| `ndcg_at_k` | Ranking quality accounting for position (DCG/IDCG) | 0-1, higher is better |

Default K values: recall at 1, 3, 5, 10; precision at 1, 3, 5.

### Generation

Measures answer quality using an **LLM-as-judge** (Claude Sonnet by default). Requires `ANTHROPIC_API_KEY`. Disabled with `--no-judge`.

| Metric | What it measures | Range |
|---|---|---|
| `faithfulness` | Whether the answer is grounded in the retrieved context (no hallucination) | 0-1, higher is better |
| `answer_correctness` | Semantic equivalence to the expected answer | 0-1, higher is better |
| `answer_relevancy` | Whether the answer addresses the question asked | 0-1, higher is better |

### Citation

Measures how accurately the system cites its sources.

| Metric | What it measures | Range |
|---|---|---|
| `citation_precision` | Fraction of citations pointing to gold passages | 0-1, higher is better |
| `citation_recall` | Fraction of gold passages that are cited | 0-1, higher is better |
| `section_accuracy` | Whether citations point to the correct document AND section | 0-1, higher is better |

### Abstention

Measures how the system handles unanswerable questions. Uses phrase matching against the answer text (e.g., "I don't have enough information").

| Metric | What it measures | Range |
|---|---|---|
| `unanswerable_accuracy` | Correctly abstains on unanswerable, correctly answers answerable | 0-1, higher is better |
| `abstention_false_positive_rate` | Rate of incorrect abstention on answerable questions | 0-1, lower is better |
| `abstention_false_negative_rate` | Rate of hallucinated answers on unanswerable questions | 0-1, lower is better |

### Performance

Operational metrics. Not factored into accuracy scoring.

| Metric | What it measures | Unit |
|---|---|---|
| `latency_p50` | Median query latency | milliseconds |
| `latency_p95` | 95th percentile query latency | milliseconds |
| `cost_per_query` | Dollar cost based on model pricing and token usage | USD |

### Weighted Scoring

All metric groups (except performance) are combined into a single weighted score using configurable objective weights:

| Objective | Default Weight | Fed by |
|---|---|---|
| `accuracy` | 0.30 | Generation metrics + Abstention metrics |
| `faithfulness` | 0.20 | Generation metrics |
| `citation` | 0.20 | Citation metrics |
| `retrieval` | 0.15 | Retrieval metrics |
| `cost` | 0.10 | Cost per query |
| `latency` | 0.05 | Latency P50 (inverted: lower is better) |

## Running Evaluations

### Prerequisites

1. RAG server running at `localhost:8001` (via `docker compose up -d`)
2. Documents already uploaded and indexed
3. `ANTHROPIC_API_KEY` set if using LLM judge (generation metrics)

### CLI

Run from `services/rag_server/`:

```bash
# Quick eval (10 samples from RAGBench, no LLM judge)
python -m evals.cli eval --samples 10 --no-judge

# Full eval with LLM judge
export ANTHROPIC_API_KEY=sk-ant-...
python -m evals.cli eval --samples 100

# Multiple datasets
python -m evals.cli eval --datasets ragbench,squad_v2,hotpotqa --samples 20

# Custom RAG server URL
python -m evals.cli eval --rag-url http://my-server:8001 --samples 10

# From YAML config
python -m evals.cli eval --config eval_config.yml

# List available datasets
python -m evals.cli datasets

# Show dataset statistics
python -m evals.cli stats

# Export a run to CSV
python -m evals.cli export --run-id abc123 --format csv

# Compare runs (with Pareto analysis)
python -m evals.cli compare run1 run2 --pareto
```

### Programmatic

```python
from evals import EvalConfig, run_evaluation, DatasetName

config = EvalConfig(
    datasets=[DatasetName.RAGBENCH, DatasetName.SQUAD_V2],
    samples_per_dataset=50,
    rag_server_url="http://localhost:8001",
)
result = run_evaluation(config)
```

### Via pytest

```bash
pytest tests/test_rag_eval.py --run-eval --eval-samples=5
```

Note: the pytest tests primarily validate the metric calculations and dataset loading in isolation. They do not run the full eval-against-server flow.

## Execution Flow

When you run an evaluation, the runner performs these steps:

```
1. Health check         GET /health on RAG server
2. Snapshot config      GET /models/info (captures LLM, embedding, reranker settings)
3. Load datasets        Download from HuggingFace (RAGBench, SQuAD, etc.)
4. Query loop           For each question:
                          POST /query → RAG server does full pipeline
                          (embed query → hybrid retrieval → rerank → LLM generation)
                          → measure latency, parse response
5. Compute metrics      Run all metric classes against question/response pairs
6. Score                Compute weighted score across objectives
7. Save                 Write run results as JSON to data/eval_runs/
```

The framework treats the RAG server as a **black box over HTTP**. It never imports server internals. The `RAGClient` class sends questions via `POST /query` and parses the JSON response (answer text, sources, citations, token usage).

## Available Datasets

Each dataset targets specific evaluation aspects:

| Dataset | Aspects | Source | Notes |
|---|---|---|---|
| `ragbench` | generation, retrieval | HuggingFace: rungalileo/ragbench | Multi-domain (legal, finance, medical, tech) |
| `qasper` | citation, generation | HuggingFace: allenai/qasper | Long-document evidence grounding. Broken with `datasets>=4.0` |
| `squad_v2` | abstention | HuggingFace: rajpurkar/SQuAD_v2.0 | ~50% unanswerable questions |
| `hotpotqa` | retrieval, generation | HuggingFace: hotpot_qa | Multi-hop reasoning |
| `msmarco` | retrieval | HuggingFace: ms_marco | Retrieval ranking |
| `golden` | generation, retrieval | Local: `evals/data/golden_qa.json` | Curated Q&A pairs for your own documents |

## Directory Structure

```
evals/
├── __init__.py              Re-exports public API (EvalConfig, run_evaluation, etc.)
├── __main__.py              Entry point for `python -m evals`
├── cli.py                   CLI commands: eval, stats, datasets, export, compare
├── config.py                EvalConfig, DatasetName enum, model cost table, weights
├── runner.py                EvaluationRunner + RAGClient (HTTP client to RAG server)
├── export.py                Export results to JSON/CSV/Markdown for manual review
│
├── schemas/
│   ├── dataset.py           EvalQuestion, GoldPassage, EvalDataset
│   ├── response.py          EvalResponse, Citation, RetrievedChunk, TokenUsage
│   └── results.py           MetricResult, MetricGroup, Scorecard, WeightedScore,
│                             ParetoPoint, EvalRun
│
├── metrics/
│   ├── base.py              BaseMetric ABC (compute, compute_batch)
│   ├── retrieval.py         RecallAtK, PrecisionAtK, MRR, NDCG
│   ├── generation.py        Faithfulness, AnswerCorrectness, AnswerRelevancy
│   ├── citation.py          CitationPrecision, CitationRecall, SectionAccuracy
│   ├── abstention.py        UnanswerableAccuracy, FalsePositiveRate, FalseNegativeRate
│   └── performance.py       LatencyP50, LatencyP95, CostPerQuery
│
├── judges/
│   └── llm_judge.py         LLMJudge — calls Claude to score faithfulness/correctness/relevancy
│
├── datasets/
│   ├── base.py              BaseDatasetLoader ABC
│   ├── registry.py          Dataset registry (register, get_loader, list_available)
│   ├── ragbench.py          RAGBench loader (15 subsets across 5 domains)
│   ├── qasper.py            Qasper loader
│   ├── squad_v2.py          SQuAD v2 loader
│   ├── hotpotqa.py          HotpotQA loader
│   ├── msmarco.py           MS MARCO loader
│   └── golden.py            Local golden dataset loader
│
└── data/
    └── golden_qa.json       Curated Q&A pairs for local evaluation
```

### Key files by role

**Orchestration:** `runner.py` contains `EvaluationRunner` which drives the entire eval. `RAGClient` is the HTTP client that talks to the RAG server. `cli.py` provides the command-line interface.

**Data contracts:** `schemas/` defines the dataclasses used everywhere. `EvalQuestion` + `GoldPassage` represent inputs. `EvalResponse` + `RetrievedChunk` + `Citation` represent outputs. `MetricResult` + `Scorecard` + `EvalRun` represent results.

**Metrics:** Each metric class inherits from `BaseMetric`, implements `compute(question, response) -> MetricResult`, and declares its `MetricGroup`. Generation metrics additionally require an `LLMJudge` instance.

**Datasets:** Each loader inherits from `BaseDatasetLoader`, downloads from HuggingFace, and converts to `EvalDataset` containing `EvalQuestion` objects with gold passages.

## Run Output

Results are saved to `data/eval_runs/{run_id}_{timestamp}.json` with this structure:

```json
{
  "id": "a1b2c3d4",
  "name": "eval-a1b2c3d4",
  "created_at": "2025-01-15T10:30:00",
  "completed_at": "2025-01-15T10:35:00",
  "config": {
    "llm_model": "gemma3:4b",
    "llm_provider": "ollama",
    "embedding_model": "nomic-embed-text:latest",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "retrieval_top_k": 10,
    "hybrid_search_enabled": true,
    "contextual_retrieval_enabled": false
  },
  "datasets": ["ragbench"],
  "scorecard": {
    "metrics": [
      {"name": "recall_at_5", "value": 0.72, "group": "retrieval", "sample_size": 100},
      {"name": "faithfulness", "value": 0.85, "group": "generation", "sample_size": 100}
    ],
    "by_group": {
      "retrieval": ["recall_at_1", "recall_at_3", "recall_at_5", "precision_at_5", "mrr", "ndcg_at_10"],
      "generation": ["faithfulness", "answer_correctness", "answer_relevancy"]
    }
  },
  "weighted_score": {
    "score": 0.73,
    "objectives": {"accuracy": 0.85, "retrieval": 0.72, "citation": 0.60},
    "weights": {"accuracy": 0.30, "retrieval": 0.15, "citation": 0.20}
  },
  "question_count": 100,
  "error_count": 2
}
```
