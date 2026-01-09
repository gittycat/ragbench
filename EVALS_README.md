# RAG Evaluation Framework (evaluation_cc)

A comprehensive evaluation framework for RAG systems that measures retrieval quality, answer generation, citation accuracy, and abstention handling. Supports multiple public datasets and LLM-as-judge evaluation with configurable models and weighted scoring.

## Usage

```bash
# Install dependencies
uv sync --group bench --group eval

# Run evaluation (requires RAG server running on localhost:8001)
python -m evaluation_cc.cli eval --datasets ragbench --samples 10

# Run with multiple datasets
python -m evaluation_cc.cli eval --datasets ragbench,squad_v2,qasper --samples 50

# Disable LLM judge (faster, retrieval metrics only)
python -m evaluation_cc.cli eval --datasets ragbench --samples 20 --no-judge

# List available datasets
python -m evaluation_cc.cli datasets

# Show dataset statistics
python -m evaluation_cc.cli stats

# Export results for manual review
python -m evaluation_cc.cli export --run-id abc123 --format markdown
python -m evaluation_cc.cli export --run-id abc123 --format csv

# Compare multiple evaluation runs
python -m evaluation_cc.cli compare run1 run2 run3
```

### Programmatic Usage

```python
from evaluation_cc import EvalConfig, run_evaluation, DatasetName

config = EvalConfig(
    datasets=[DatasetName.RAGBENCH, DatasetName.SQUAD_V2],
    samples_per_dataset=100,
    rag_server_url="http://localhost:8001",
)
results = run_evaluation(config)
print(f"Weighted Score: {results.weighted_score.score:.3f}")
```

## Evaluation Metrics

### Retrieval Metrics
Measure how well the system retrieves relevant documents/chunks.

| Metric | Description | Output |
|--------|-------------|--------|
| `recall_at_k` | Fraction of gold passages retrieved in top K | 0-1 (higher=better) |
| `precision_at_k` | Fraction of top K results that are relevant | 0-1 (higher=better) |
| `mrr` | Reciprocal rank of first relevant result | 0-1 (higher=better) |
| `ndcg_at_k` | Normalized discounted cumulative gain | 0-1 (higher=better) |

### Generation Metrics
Measure answer quality using LLM-as-judge evaluation.

| Metric | Description | Output |
|--------|-------------|--------|
| `faithfulness` | Answer grounded in retrieved context | 0-1 (higher=better) |
| `answer_correctness` | Semantic match with expected answer | 0-1 (higher=better) |
| `answer_relevancy` | Answer addresses the question asked | 0-1 (higher=better) |

### Citation Metrics
Measure citation accuracy and completeness.

| Metric | Description | Output |
|--------|-------------|--------|
| `citation_precision` | Cited chunks are actually relevant | 0-1 (higher=better) |
| `citation_recall` | All gold evidence is cited | 0-1 (higher=better) |
| `section_accuracy` | Correct doc_id + chunk_id cited | 0-1 (higher=better) |

### Abstention Metrics
Measure handling of unanswerable questions.

| Metric | Description | Output |
|--------|-------------|--------|
| `unanswerable_accuracy` | Correct refusal on unanswerable questions | 0-1 (higher=better) |
| `false_positive_rate` | Answered when should have refused | 0-1 (lower=better) |
| `false_negative_rate` | Refused when should have answered | 0-1 (lower=better) |

### Performance Metrics
Measure system performance characteristics.

| Metric | Description | Output |
|--------|-------------|--------|
| `latency_p50_ms` | Median query latency | milliseconds |
| `latency_p95_ms` | 95th percentile latency | milliseconds |
| `latency_avg_ms` | Average query latency | milliseconds |

## Available Datasets

| Dataset | Primary Focus | Description |
|---------|---------------|-------------|
| **ragbench** | Generation, Retrieval | Multi-domain RAG benchmark (legal, finance, tech, medical) |
| **qasper** | Citation, Generation | Scientific papers with evidence annotations |
| **squad_v2** | Abstention | Reading comprehension with unanswerable questions |
| **hotpotqa** | Retrieval, Generation | Multi-hop reasoning across documents |
| **msmarco** | Retrieval | Large-scale retrieval ranking benchmark |

## Weighted Scoring

Results include a weighted overall score based on configurable objectives:

| Objective | Default Weight | Source Metrics |
|-----------|----------------|----------------|
| accuracy | 0.30 | answer_correctness, unanswerable_accuracy |
| faithfulness | 0.20 | faithfulness |
| citation | 0.20 | citation_precision, citation_recall |
| retrieval | 0.15 | recall, precision, mrr, ndcg |
| cost | 0.10 | token usage (if available) |
| latency | 0.05 | latency_p50 |

## Output Files

Evaluation runs are saved to `data/eval_runs/` as JSON files containing:
- Configuration snapshot (LLM, embedding, reranker settings)
- Complete scorecard with all metrics
- Weighted score with objective breakdown
- Run metadata (duration, error count, etc.)
