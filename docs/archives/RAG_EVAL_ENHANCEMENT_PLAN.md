# RAG Evaluation Enhancement Plan

**Date**: 2025-12-29
**Status**: Proposed
**Author**: Claude Code

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Research Findings](#research-findings)
3. [Framework Comparison](#framework-comparison)
4. [Decision Justification](#decision-justification)
5. [Implementation Plan](#implementation-plan)
6. [Data Models](#data-models)
7. [API Design](#api-design)
8. [Dashboard Components](#dashboard-components)
9. [CLI Enhancements](#cli-enhancements)
10. [File Changes Summary](#file-changes-summary)

---

## Executive Summary

### Goal
Enhance the RAG evaluation system to:
1. **Compare LLM models** (GPT-4 vs Claude vs Ollama) against a golden baseline
2. **Track trends** with config change annotations
3. **Provide side-by-side comparison** of evaluation runs
4. **Auto-recommend** optimal configurations based on accuracy/speed/cost tradeoffs

### Decision
**Keep DeepEval + Enhance** rather than migrating to a different framework.

### Scope
- Backend: New services, API endpoints, enhanced data models
- Frontend: New dashboard components for comparison, trends, recommendations
- CLI: New subcommands for baseline management and comparison

---

## Research Findings

### 2025 RAG Evaluation Landscape

The RAG evaluation space has matured significantly. Key trends from research:

1. **LLM-as-Judge is standard**: Per [Databricks research](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG), LLM judges achieve 80%+ agreement with human grading.

2. **Traditional metrics still dominate**: The [arXiv comprehensive survey (April 2025)](https://arxiv.org/abs/2504.14891) notes that "traditional metrics predominantly dominate evaluation usage, while LLM-based methods have not yet gained widespread acceptance among researchers."

3. **Continuous evaluation loops**: [Braintrust](https://www.braintrust.dev/articles/rag-evaluation-metrics) emphasizes that production monitoring should feed back into evaluation datasets.

4. **Low-precision grading scales work best**: Binary (0/1) or 4-point (0-3) scales retain precision while improving consistency between different LLM judges.

### Key Academic Papers

| Paper | Year | Key Contribution |
|-------|------|------------------|
| [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217) | 2023 | Reference-free evaluation framework, industry-standard metrics |
| [Evaluation of RAG: A Survey](https://arxiv.org/abs/2405.07437) | 2024 | Unified evaluation process (Auepora), 12 frameworks compared |
| [RAG Evaluation in the Era of LLMs](https://arxiv.org/abs/2504.14891) | 2025 | Comprehensive survey of traditional vs LLM-based methods |
| [RAGBench](https://arxiv.org/abs/2407.11005) | 2024 | TRACe framework, 100k example benchmark, found fine-tuned RoBERTa outperforms LLM judges |

### Industry Benchmarks

Notable RAG benchmarks for future consideration:
- **ChatRAG-Bench**: Conversational QA across document types
- **FRAMES**: Multi-hop reasoning with 800+ test samples
- **RAGTruth**: Hallucination detection with 18k responses
- **MTRAG (IBM)**: Enterprise conversational RAG benchmark

---

## Framework Comparison

### Evaluated Frameworks

| Framework | Type | License | Strengths | Weaknesses | Best For |
|-----------|------|---------|-----------|------------|----------|
| **DeepEval** (current) | Eval | Apache 2.0 | Self-explaining scores, 14+ metrics, CI/CD ready, red-teaming, pytest-like interface | Steep learning curve, LLM costs, throttling issues | Production testing, regression testing |
| **RAGAS** | Eval | Apache 2.0 | Industry-standard RAG metrics, reference-free, extensible | Opaque scores (not self-explanatory), no built-in UI, evolving/brittle | Research, establishing baselines |
| **TruLens** | Eval + Observability | MIT | LangChain/LlamaIndex integration, feedback functions, guardrails | Less production-ready, limited standalone use | Agentic workflows, debugging |
| **Langfuse** | Observability + Eval | MIT | Open-source, self-hostable, 19k GitHub stars, RAGAS integration | Requires hosting infrastructure | Tracing + evaluation combo |
| **Phoenix (Arize)** | Observability | Elastic 2.0 | OpenTelemetry-based, framework-agnostic, visual UI | Not MIT license, less eval-focused | Tracing-heavy orgs |
| **Braintrust** | Platform | Proprietary | 80x faster queries, Notion/Stripe use, production-to-eval loop | Closed-source, enterprise pricing | Enterprise production |
| **promptfoo** | Eval + Security | MIT | Side-by-side model comparison, red-teaming, CI/CD, local execution | Less RAG-specific metrics | Model A/B testing, security testing |

### DeepEval vs RAGAS (Direct Comparison)

From [research](https://research.aimultiple.com/rag-evaluation-tools/):

| Aspect | DeepEval | RAGAS |
|--------|----------|-------|
| Score Explanation | Self-explaining with reasoning | Opaque (score only) |
| Debugging | Can inspect LLM judge decisions | Limited visibility |
| Metrics | 14+ including RAG, fine-tuning, safety | 5 core RAG metrics |
| Integration | pytest-like, CI/CD native | Code-centric, no UI |
| Red-teaming | Built-in (40+ attack types) | Not included |

---

## Decision Justification

### Why Keep DeepEval

1. **Already integrated**: Current system uses DeepEval with 5 metrics, CLI, pytest integration. Migration would require significant rework.

2. **Self-explaining scores**: DeepEval's metrics include reasoning for why scores are what they are, making debugging easier than RAGAS.

3. **CI/CD ready**: Native pytest integration means evaluations already run in CI pipeline.

4. **Sufficient metrics**: The 5 current metrics (Contextual Precision, Contextual Recall, Faithfulness, Answer Relevancy, Hallucination) cover retrieval and generation quality.

5. **No vendor lock-in**: Open-source Apache 2.0 license, can use any LLM as judge.

### What We're Adding (Not Replacing)

Instead of migrating frameworks, we enhance the existing system with:

1. **Config snapshots**: Track exactly what LLM/retrieval settings produced each result
2. **Latency tracking**: P50/P95 query times during evaluation
3. **Cost tracking**: Token usage and estimated costs per model
4. **Golden baseline**: Single "target to beat" for pass/fail comparison
5. **Comparison service**: Side-by-side diff of any two runs
6. **Recommendation engine**: Weighted scoring to suggest optimal config

### Why Not Other Options

| Option | Why Not |
|--------|---------|
| Migrate to RAGAS | Would lose self-explaining scores, requires new integration work |
| Add Langfuse | Adds hosting complexity, overkill for current needs (no production tracing requirement) |
| Add promptfoo | Better for model A/B testing but less RAG-specific, would duplicate eval capability |
| Braintrust | Proprietary, enterprise pricing, data residency concerns |

---

## Implementation Plan

### Phase 1: Data Models

**Goal**: Extend EvaluationRun with config snapshots, latency, cost, baseline comparison.

**File**: `services/rag_server/schemas/metrics.py`

New models:
- `ConfigSnapshot`: Full config at eval time (LLM, embedding, retrieval, reranker)
- `LatencyMetrics`: avg, p50, p95, min, max query times
- `CostMetrics`: tokens, estimated cost, cost per query
- `GoldenBaseline`: Target run to beat with thresholds
- `ComparisonResult`: Delta between two runs
- `Recommendation`: Suggested config with reasoning

### Phase 2: New Services

**Goal**: Business logic for baseline, comparison, and recommendations.

New files:
- `services/rag_server/services/baseline.py`: Get/set/clear golden baseline
- `services/rag_server/services/comparison.py`: Compare two runs
- `services/rag_server/services/recommendation.py`: Weighted scoring algorithm
- `services/rag_server/services/cost_tracker.py`: Token pricing per model

### Phase 3: API Endpoints

**Goal**: Expose new functionality via REST API.

**File**: `services/rag_server/api/routes/metrics.py`

New endpoints:
```
GET    /metrics/baseline                     # Get current baseline
POST   /metrics/baseline/{run_id}            # Set baseline
DELETE /metrics/baseline                     # Clear baseline
GET    /metrics/compare/{run_a}/{run_b}      # Compare two runs
GET    /metrics/compare-to-baseline/{run_id} # Compare to baseline
POST   /metrics/recommend                    # Get recommendation
```

### Phase 4: Evaluation Runner Enhancement

**Goal**: Capture latency and cost during evaluation.

**File**: `services/rag_server/evaluation/live_eval.py`

Changes:
- Add LatencyTracker to measure query times
- Add CostTracker to count tokens
- Capture full config snapshot at start
- Include metrics in results

### Phase 5: Dashboard Components

**Goal**: Visual comparison, trends, and recommendations.

New Svelte components:
- `BaselineIndicator.svelte`: Pass/fail badges against baseline
- `RunComparison.svelte`: Side-by-side two-column comparison
- `TrendChart.svelte`: Chart.js with annotation markers
- `RecommendationPanel.svelte`: Weight sliders + recommendation card

**File**: `services/webapp/src/routes/dashboard/+page.svelte`

Integration points:
- Baseline indicator below eval metrics
- Expandable comparison section
- Enhanced trend chart (or alongside sparklines)
- Recommendation panel at bottom

### Phase 6: CLI Enhancements

**Goal**: Command-line access to new features.

**File**: `services/rag_server/evaluation/cli.py`

New subcommands:
```bash
python -m evaluation.cli baseline show
python -m evaluation.cli baseline set <run_id>
python -m evaluation.cli baseline clear
python -m evaluation.cli baseline check
python -m evaluation.cli compare <run_a> <run_b> [--json]
python -m evaluation.cli recommend --accuracy=0.5 --speed=0.3 --cost=0.2
```

---

## Data Models

### ConfigSnapshot

```python
class ConfigSnapshot(BaseModel):
    """Complete configuration snapshot at evaluation time."""
    # LLM
    llm_provider: str           # ollama, openai, anthropic, google, deepseek, moonshot
    llm_model: str              # e.g., "gpt-4o", "claude-sonnet-4", "gemma3:4b"
    llm_base_url: Optional[str]

    # Embedding
    embedding_provider: str
    embedding_model: str

    # Retrieval
    retrieval_top_k: int
    hybrid_search_enabled: bool
    rrf_k: int
    contextual_retrieval_enabled: bool

    # Reranker
    reranker_enabled: bool
    reranker_model: Optional[str]
    reranker_top_n: Optional[int]
```

### LatencyMetrics

```python
class LatencyMetrics(BaseModel):
    """Query latency statistics from evaluation run."""
    avg_query_time_ms: float
    p50_query_time_ms: float
    p95_query_time_ms: float
    min_query_time_ms: float
    max_query_time_ms: float
    total_queries: int
```

### CostMetrics

```python
class CostMetrics(BaseModel):
    """Token usage and cost tracking."""
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    cost_per_query_usd: float
```

### GoldenBaseline

```python
class GoldenBaseline(BaseModel):
    """Golden baseline configuration and thresholds."""
    run_id: str
    set_at: datetime
    set_by: Optional[str] = None

    # Thresholds to beat (from the baseline run's scores)
    target_metrics: dict[str, float]

    # Reference configuration
    config_snapshot: ConfigSnapshot

    # Optional performance targets
    target_latency_p95_ms: Optional[float] = None
    target_cost_per_query_usd: Optional[float] = None
```

### ComparisonResult

```python
class ComparisonResult(BaseModel):
    """Result of comparing two evaluation runs."""
    run_a_id: str
    run_b_id: str

    # Metric deltas (positive = A is better)
    metric_deltas: dict[str, float]

    # Latency comparison
    latency_delta_ms: Optional[float] = None
    latency_improvement_pct: Optional[float] = None

    # Cost comparison
    cost_delta_usd: Optional[float] = None
    cost_improvement_pct: Optional[float] = None

    # Winner determination
    winner: Literal["run_a", "run_b", "tie"]
    winner_reason: str
```

### Recommendation

```python
class Recommendation(BaseModel):
    """Configuration recommendation based on historical analysis."""
    recommended_config: ConfigSnapshot
    source_run_id: str

    reasoning: str

    # Normalized scores (0-1)
    accuracy_score: float
    speed_score: float
    cost_score: float

    # Weights used
    weights: dict[str, float]

    # Runner-up options
    alternatives: list[dict]
```

---

## API Design

### Baseline Endpoints

```python
@router.get("/metrics/baseline", response_model=Optional[GoldenBaseline])
async def get_golden_baseline():
    """Get current golden baseline, or null if not set."""

@router.post("/metrics/baseline/{run_id}", response_model=GoldenBaseline)
async def set_golden_baseline(run_id: str, set_by: Optional[str] = None):
    """Set a specific evaluation run as the golden baseline.

    The baseline's metric scores become the thresholds to beat.
    """

@router.delete("/metrics/baseline", status_code=204)
async def clear_golden_baseline():
    """Clear the current golden baseline."""
```

### Comparison Endpoints

```python
@router.get("/metrics/compare/{run_a_id}/{run_b_id}", response_model=ComparisonResult)
async def compare_runs(run_a_id: str, run_b_id: str):
    """Compare two evaluation runs side-by-side.

    Returns metric deltas, latency/cost comparison, and winner determination.
    """

@router.get("/metrics/compare-to-baseline/{run_id}", response_model=ComparisonResult)
async def compare_to_baseline(run_id: str):
    """Compare a run against the golden baseline.

    Returns 404 if no baseline is set.
    """
```

### Recommendation Endpoint

```python
@router.post("/metrics/recommend", response_model=Recommendation)
async def get_recommendation(
    accuracy_weight: float = Query(0.5, ge=0, le=1),
    speed_weight: float = Query(0.3, ge=0, le=1),
    cost_weight: float = Query(0.2, ge=0, le=1),
    limit_to_runs: int = Query(10, ge=1, le=100)
):
    """Get recommended configuration based on weighted preferences.

    Algorithm:
    1. Load last N evaluation runs
    2. Normalize metrics across runs (0-1 scale)
    3. Calculate composite score per run
    4. Return config from highest-scoring run
    """
```

---

## Dashboard Components

### BaselineIndicator

Shows current baseline status with pass/fail badges per metric.

```svelte
<div class="baseline-indicator">
  {#if baseline}
    <div class="flex items-center gap-2">
      <span class="badge badge-outline">Golden Baseline</span>
      <span class="text-xs">{baseline.config_snapshot.llm_model}</span>
      <span class="text-xs text-base-content/60">Set {timeAgo(baseline.set_at)}</span>
    </div>

    <!-- Pass/fail per metric -->
    <div class="flex gap-1 mt-1">
      {#each Object.entries(passStatus) as [metric, passed]}
        <span class="badge badge-xs {passed ? 'badge-success' : 'badge-error'}">
          {metric}
        </span>
      {/each}
    </div>
  {:else}
    <button class="btn btn-xs" onclick={setCurrentAsBaseline}>
      Set Current as Baseline
    </button>
  {/if}
</div>
```

### RunComparison

Side-by-side comparison of two runs.

```
+------------------+------------------+
|     Run A        |     Run B        |
|  gpt-4o-mini     |  claude-sonnet   |
+------------------+------------------+
| Faithfulness 85% | Faithfulness 92% |
| Relevancy    78% | Relevancy    81% |
| ...              | ...              |
+------------------+------------------+
| P50: 245ms       | P50: 312ms       |
| Cost: $0.002     | Cost: $0.008     |
+------------------+------------------+
|        Winner: Run B (higher accuracy)    |
+-------------------------------------------+
```

### TrendChart

Interactive Chart.js chart with:
- Multiple metric lines (selectable)
- Vertical annotation lines for config changes
- Hover tooltips showing run details
- Color-coded by trend direction

### RecommendationPanel

Weight sliders + recommendation display:

```
+------------------------------------------+
| Configuration Recommendation              |
+------------------------------------------+
| Accuracy [=======---] 0.7                |
| Speed    [====------] 0.4                |
| Cost     [===-------] 0.3                |
+------------------------------------------+
| [Recommended] gpt-4o-mini                |
| "Best balance of accuracy (82%) and     |
|  cost ($0.002/query). Speed is moderate |
|  but acceptable for most use cases."    |
+------------------------------------------+
| Accuracy: 82%  Speed: 65%  Cost: 91%    |
+------------------------------------------+
| Alternatives:                            |
| - claude-sonnet-4 (higher accuracy, 4x cost) |
| - gemma3:4b (free, lower accuracy)       |
+------------------------------------------+
```

---

## CLI Enhancements

### Baseline Commands

```bash
# Show current baseline
$ python -m evaluation.cli baseline show
Golden Baseline:
  Run ID: eval-2025-01-15-abc123
  Set at: 2025-01-15 14:30:00
  Config: gpt-4o | nomic-embed-text | hybrid=true | reranker=true
  Thresholds:
    faithfulness: 0.85
    answer_relevancy: 0.80
    contextual_precision: 0.75
    contextual_recall: 0.78
    hallucination: 0.45 (lower is better)

# Set a run as baseline
$ python -m evaluation.cli baseline set eval-2025-01-20-def456
Baseline set to eval-2025-01-20-def456

# Check latest run against baseline
$ python -m evaluation.cli baseline check
Comparing latest run (eval-2025-01-21-ghi789) to baseline...
  faithfulness:        0.88 vs 0.85  PASS (+3.5%)
  answer_relevancy:    0.79 vs 0.80  FAIL (-1.3%)
  contextual_precision: 0.82 vs 0.75  PASS (+9.3%)
  contextual_recall:   0.80 vs 0.78  PASS (+2.6%)
  hallucination:       0.42 vs 0.45  PASS (-6.7%)

Result: 4/5 metrics pass baseline
```

### Compare Command

```bash
$ python -m evaluation.cli compare eval-abc eval-def
Comparison: eval-abc vs eval-def

              eval-abc    eval-def    Delta
              --------    --------    -----
faithfulness     0.85        0.92    +8.2%
relevancy        0.78        0.81    +3.8%
precision        0.75        0.73    -2.7%
recall           0.80        0.85    +6.3%
hallucination    0.42        0.38    -9.5%  (lower=better)

Latency (P95):   312ms       425ms   +36.2%
Cost/query:     $0.002      $0.008   +300%

Winner: eval-def (higher accuracy, acceptable latency increase)
```

### Recommend Command

```bash
$ python -m evaluation.cli recommend --accuracy=0.6 --speed=0.2 --cost=0.2
Analyzing last 10 evaluation runs...

Recommended Configuration:
  LLM: gpt-4o-mini (OpenAI)
  Embedding: nomic-embed-text (Ollama)
  Retrieval: hybrid=true, top_k=10, rrf_k=60
  Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2, top_n=5

Scores:
  Accuracy: 82%
  Speed:    71%
  Cost:     94%
  Composite: 82.4% (weighted)

Reasoning:
  Best accuracy-to-cost ratio. Claude models score 5% higher on
  faithfulness but cost 4x more per query. Ollama models are free
  but score 15% lower on answer relevancy.

Alternatives:
  1. claude-sonnet-4: +5% accuracy, -300% cost efficiency
  2. gemma3:4b: +100% cost efficiency, -15% accuracy
```

---

## File Changes Summary

### New Files

| File | Purpose |
|------|---------|
| `services/rag_server/services/baseline.py` | Golden baseline management |
| `services/rag_server/services/comparison.py` | Run comparison logic |
| `services/rag_server/services/recommendation.py` | Weighted recommendation algorithm |
| `services/rag_server/services/cost_tracker.py` | Token pricing and cost calculation |
| `services/webapp/src/lib/components/BaselineIndicator.svelte` | Baseline pass/fail UI |
| `services/webapp/src/lib/components/RunComparison.svelte` | Side-by-side comparison UI |
| `services/webapp/src/lib/components/TrendChart.svelte` | Interactive trend chart |
| `services/webapp/src/lib/components/RecommendationPanel.svelte` | Recommendation UI |
| `evals/data/golden_baseline.json` | Baseline storage |

### Modified Files

| File | Changes |
|------|---------|
| `services/rag_server/schemas/metrics.py` | Add ConfigSnapshot, LatencyMetrics, CostMetrics, GoldenBaseline, ComparisonResult, Recommendation |
| `services/rag_server/api/routes/metrics.py` | Add baseline, compare, recommend endpoints |
| `services/rag_server/services/metrics.py` | Integrate baseline checking, enhance config capture |
| `services/rag_server/evaluation/live_eval.py` | Add latency/cost tracking during eval |
| `services/rag_server/evaluation/cli.py` | Add baseline, compare, recommend subcommands |
| `services/webapp/src/routes/dashboard/+page.svelte` | Integrate new components |
| `services/webapp/src/lib/api.ts` | Add API functions for new endpoints |

### Dependencies

**Backend**: No new packages (uses existing Pydantic, FastAPI)

**Frontend**:
```bash
npm install chart.js chartjs-plugin-annotation
```

---

## Storage Design

### Golden Baseline File

Location: `evals/data/golden_baseline.json`

```json
{
  "run_id": "eval-2025-01-15-abc123",
  "set_at": "2025-01-15T14:30:00Z",
  "set_by": "developer",
  "target_metrics": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "contextual_precision": 0.75,
    "contextual_recall": 0.78,
    "hallucination": 0.45
  },
  "config_snapshot": {
    "llm_provider": "openai",
    "llm_model": "gpt-4o",
    "embedding_model": "nomic-embed-text:latest",
    "retrieval_top_k": 10,
    "hybrid_search_enabled": true,
    "rrf_k": 60,
    "contextual_retrieval_enabled": false,
    "reranker_enabled": true,
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "reranker_top_n": 5
  },
  "target_latency_p95_ms": 500,
  "target_cost_per_query_usd": 0.01
}
```

### Enhanced Evaluation Run File

Location: `evals/data/results/eval-YYYY-MM-DD-XXXXXX.json`

Added fields to existing structure:
```json
{
  "run_id": "eval-2025-01-20-def456",
  "timestamp": "2025-01-20T10:15:30Z",
  "framework": "DeepEval",
  "eval_model": "claude-sonnet-4-20250514",

  "config_snapshot": { ... },

  "latency": {
    "avg_query_time_ms": 285.4,
    "p50_query_time_ms": 245.0,
    "p95_query_time_ms": 512.3,
    "min_query_time_ms": 180.2,
    "max_query_time_ms": 1250.8,
    "total_queries": 10
  },

  "cost": {
    "total_input_tokens": 15420,
    "total_output_tokens": 3850,
    "total_tokens": 19270,
    "estimated_cost_usd": 0.0245,
    "cost_per_query_usd": 0.00245
  },

  "is_golden_baseline": false,
  "compared_to_baseline": {
    "baseline_run_id": "eval-2025-01-15-abc123",
    "metrics_pass": ["faithfulness", "contextual_precision", "contextual_recall"],
    "metrics_fail": ["answer_relevancy"],
    "overall_pass": false
  },

  "metric_averages": { ... },
  "metric_pass_rates": { ... },
  "test_cases": [ ... ]
}
```

---

## Testing Strategy

### Unit Tests

New test files:
- `tests/unit/test_baseline_service.py`
- `tests/unit/test_comparison_service.py`
- `tests/unit/test_recommendation_service.py`
- `tests/unit/test_cost_tracker.py`

### Integration Tests

Add to existing integration test suite:
- Test baseline CRUD operations
- Test comparison with real eval runs
- Test recommendation with various weights

### Manual Testing

Dashboard components require manual verification:
1. Set baseline via CLI, verify indicator shows in dashboard
2. Run eval with different LLM, verify comparison works
3. Adjust weight sliders, verify recommendations update
4. Verify trend chart annotations appear on config changes

---

## Future Considerations

### Not in Scope (Potential Future Work)

1. **Prompt variation testing**: Compare different system prompts
2. **Retrieval strategy comparison**: Hybrid vs vector-only A/B testing
3. **Langfuse integration**: Production tracing + eval correlation
4. **Automated regression alerts**: Notify when eval drops below baseline
5. **Multi-baseline support**: Different baselines for different use cases

### Upgrade Path

If requirements grow beyond DeepEval capabilities:
1. **Add Langfuse** for production tracing (complements DeepEval)
2. **Add promptfoo** for security/red-teaming (parallel to DeepEval)
3. **Consider Braintrust** if enterprise features needed (replaces DeepEval)
