# RAG Evaluation Datasets

This document outlines publicly available datasets for evaluating our RAG system, including requirements, recommendations, and implementation guidance.

## Requirements

### Evaluation Goals
- **Retrieval accuracy**: Measure if the system finds the correct chunks/passages
- **Answer correctness**: Verify generated answers are factually correct
- **Faithfulness**: Ensure answers are grounded in retrieved context (no hallucination)

### Dataset Requirements
- **Gold passages required**: Need ground truth chunks to measure retrieval precision/recall
- **Multiple sizes**: Options for quick validation and comprehensive benchmarking
- **Domain agnostic**: Open to any domain that tests RAG capabilities well
- **Corpus ingestion**: Documents must be ingestable into ChromaDB for end-to-end testing

---

## Recommended Datasets

### Primary: RAGBench
**Purpose-built for RAG evaluation with industry-relevant documents**

| Attribute | Value |
|-----------|-------|
| Size | 100K examples |
| Domains | 5 industry domains (user manuals, tech docs) |
| Gold passages | Yes |
| Format | Hugging Face dataset |
| Link | [rungalileo/ragbench](https://huggingface.co/datasets/rungalileo/ragbench) |

**Why recommended:**
- Specifically designed for RAG systems
- Includes TRACe evaluation framework (explainable metrics)
- Industry-relevant corpus (not just Wikipedia)
- Documents + questions + gold passages + answers included

### Secondary: HotpotQA
**Multi-hop reasoning benchmark with sentence-level evidence**

| Attribute | Value |
|-----------|-------|
| Size | 113K QA pairs |
| Corpus | Wikipedia paragraphs |
| Gold passages | Yes (sentence-level supporting facts) |
| Format | JSON with context paragraphs |
| Link | [hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa) |

**Why recommended:**
- Tests complex multi-hop reasoning
- Sentence-level gold evidence for precise retrieval evaluation
- Well-established benchmark with published baselines
- Distractor setting provides realistic retrieval challenge

### Tertiary: SQuAD 2.0
**Quick-start option with unanswerable questions**

| Attribute | Value |
|-----------|-------|
| Size | 150K (including 50K unanswerable) |
| Corpus | Wikipedia paragraphs (self-contained) |
| Gold passages | Yes (context included per example) |
| Format | JSON |
| Link | [rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad) |

**Why recommended:**
- Smallest corpus, easiest to ingest
- Tests handling of unanswerable questions
- Good for pipeline validation before larger datasets

### Alternative: Natural Questions
**Real user queries from Google Search**

| Attribute | Value |
|-----------|-------|
| Size | 323K questions |
| Corpus | Full Wikipedia |
| Gold passages | Yes (long/short answer contexts) |
| Format | JSON |
| Link | [google-research-datasets/natural_questions](https://huggingface.co/datasets/google-research-datasets/natural_questions) |

**Why recommended:**
- Real user queries (not synthetic)
- Tests real-world retrieval scenarios
- Large scale for comprehensive benchmarking

---

## Dataset Comparison

| Aspect | RAGBench | HotpotQA | SQuAD 2.0 | Natural Questions |
|--------|----------|----------|-----------|-------------------|
| Corpus size | Medium | Medium | Small | Large |
| Ingestion complexity | Low | Low | Very Low | High |
| Multi-hop reasoning | Some | Strong | Limited | Limited |
| Industry relevance | High | Low | Low | Medium |
| Unanswerable Qs | No | No | Yes | Some |
| Established baselines | Emerging | Yes | Yes | Yes |

---

## Other Notable Datasets

### Specialized Benchmarks

| Dataset | Focus | Size | Link |
|---------|-------|------|------|
| **CLAPnq** | Long-form answers, unanswerable Qs | ~5K | [MIT Press](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00729/127456) |
| **Open RAGBench** | PDF/multimodal evaluation | 3K QA pairs | [vectara/open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench) |
| **MS MARCO** | Passage ranking | 1M+ | [microsoft/ms_marco](https://huggingface.co/datasets/microsoft/ms_marco) |
| **TriviaQA** | Factoid questions | 110K | [trivia_qa](https://huggingface.co/datasets/trivia_qa) |
| **BEIR** | Retrieval robustness (18 datasets) | Varies | [beir-cellar/beir](https://github.com/beir-cellar/beir) |
| **T²-RAGBench** | Tables + Text (financial) | - | [HuggingFace](https://huggingface.co/papers/2506.12071) |
| **FRAMES** | Factuality + Reasoning | - | [HuggingFace](https://huggingface.co/papers/2409.12941) |

---

## Domain-Specific Datasets

Datasets for specialized document types. Implement on a need basis when deploying to specific industries.

### Legal

| Dataset | Focus | Notes |
|---------|-------|-------|
| **CUAD** | Contract understanding | 500+ contracts, clause extraction |
| **LegalBench-RAG** | Compliance | Citation accuracy for legal docs |

### Finance & Accounting

| Dataset | Focus | Notes |
|---------|-------|-------|
| **FinQA** | Financial reasoning | Numerical QA over earnings reports |
| **TAT-QA** | Tables + text | Hybrid tabular/text financial docs |
| **T²-RAGBench** | Tables + text | 2025 benchmark for structured data |

### Healthcare & Biomedical

| Dataset | Focus | Notes |
|---------|-------|-------|
| **PubMedQA** | Medical literature | Yes/no/maybe reasoning over abstracts |
| **BioASQ** | Biomedical QA | Semantic indexing and QA |

### Government & Policy

| Dataset | Focus | Notes |
|---------|-------|-------|
| **GovReport** | Long-form summarization | Government reports, 9K+ words avg |

### Technical & Product Documentation

| Dataset | Focus | Notes |
|---------|-------|-------|
| **TechQA** | IT support | IBM technical QA (may require access) |
| **ExpertQA** | Expert knowledge | Subject matter expert answers |

### Procedural & How-To

| Dataset | Focus | Notes |
|---------|-------|-------|
| **WikiHowQA** | Step-by-step | Procedural question answering |

### Conversational & Multi-Turn

| Dataset | Focus | Notes |
|---------|-------|-------|
| **MultiDoc2Dial** | Multi-doc chat | Gated access, conversational QA |
| **ChatRAG-Bench** | Chat continuity | Multi-turn RAG evaluation |

---

## Implementation Plan

### Phase 1: Pipeline Validation
1. Start with **SQuAD 2.0** (smallest corpus)
2. Validate ingestion into ChromaDB works correctly
3. Run baseline evaluation with existing DeepEval setup
4. Verify gold passage retrieval metrics can be computed

### Phase 2: Comprehensive Benchmarking
1. Ingest **RAGBench** corpus
2. Run full evaluation across all 100K examples (or stratified sample)
3. Establish baseline metrics for retrieval and generation

### Phase 3: Specialized Testing
1. Add **HotpotQA** for multi-hop reasoning evaluation
2. Compare performance on single-hop vs multi-hop queries
3. Identify areas for improvement

---

## Integration with DeepEval

Our current evaluation uses DeepEval with the following metrics:
- Faithfulness
- Answer Relevancy
- Contextual Precision
- Contextual Recall

### Mapping Dataset Gold Data to DeepEval

| DeepEval Metric | Required Data | Dataset Field |
|-----------------|---------------|---------------|
| Faithfulness | Retrieved context, Answer | Generated at query time |
| Answer Relevancy | Question, Answer | `question`, `answer` |
| Contextual Precision | Retrieved contexts, Expected context | Compare against gold passages |
| Contextual Recall | Retrieved contexts, Expected context | Compare against gold passages |

### Additional Metrics for Retrieval

With gold passages available, we can compute:
- **Recall@K**: Is gold passage in top-K retrieved?
- **Precision@K**: What fraction of top-K are relevant?
- **MRR**: Mean Reciprocal Rank of first gold passage
- **NDCG**: Normalized Discounted Cumulative Gain

---

## Data Format Requirements

For ingestion into our RAG system, datasets should be converted to:

### Document Format
```json
{
  "doc_id": "unique_identifier",
  "content": "document text content",
  "metadata": {
    "source": "dataset_name",
    "original_id": "original_dataset_id"
  }
}
```

### Evaluation Format
```json
{
  "question": "the question text",
  "expected_answer": "gold standard answer",
  "gold_passages": ["passage_id_1", "passage_id_2"],
  "metadata": {
    "difficulty": "easy|medium|hard",
    "reasoning_type": "single_hop|multi_hop"
  }
}
```

---

## Resources

### Papers
- [RAGBench: Explainable Benchmark for RAG Systems](https://arxiv.org/abs/2407.11005) (2024)
- [Evaluation of Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2405.07437) (2024)
- [HotpotQA: Multi-hop Question Answering](https://hotpotqa.github.io/)

### Tools
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) - Dataset loading
- [RAGAS](https://docs.ragas.io/) - RAG evaluation framework
- [BEIR Benchmark](https://github.com/beir-cellar/beir) - Retrieval evaluation

### GitHub Repositories
- [Awesome RAG Evaluation](https://github.com/YHPeter/Awesome-RAG-Evaluation) - Curated list of RAG evaluation resources
- [Large QA Datasets](https://github.com/ad-freiburg/large-qa-datasets) - Collection of QA datasets

---

## Future Considerations

### Criteria for Adding New Datasets
When evaluating new datasets for inclusion:
1. Must include gold passages/contexts
2. Prefer datasets with established baselines for comparison
3. Consider domain relevance to target use cases
4. Evaluate corpus size vs ingestion/storage requirements
5. Check license compatibility for intended use

### Potential Expansions
- **Domain-specific**: Add legal, medical, or technical datasets if targeting those domains
- **Multilingual**: Consider mMARCO for multilingual evaluation
- **Multimodal**: Open RAGBench for PDF/table evaluation
- **Conversational**: Multi-turn QA datasets for chat evaluation

---

## Benchmark Infrastructure

This section describes the recommended setup for loading datasets, running evaluations, and viewing results.

### Infrastructure Requirements

These choices drive the architecture recommendations below. Update these if requirements change.

| Aspect | Current Choice | Alternatives |
|--------|----------------|--------------|
| **Run frequency** | Ad-hoc/manual | On every PR, Scheduled (nightly/weekly) |
| **Data isolation** | Ephemeral per test run | Separate ChromaDB instance, Separate collection |
| **Results display** | Dashboard/UI (local) | CLI only, JSON files, Cloud service |
| **Result persistence** | Store all runs | Keep last N runs, No persistence |
| **Runtime budget** | 5-30 minutes | Under 5 min (CI), Hours (comprehensive) |
| **Integration** | Extend existing eval/ | Separate module, Replace current |

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        justfile commands                         │
├─────────────────────────────────────────────────────────────────┤
│  just bench-load <dataset>     # Download & prepare dataset     │
│  just bench-run <dataset>      # Run evaluation                 │
│  just bench-dashboard          # Launch results dashboard       │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Ephemeral Test Environment                    │
├─────────────────────────────────────────────────────────────────┤
│  docker-compose.bench.yml                                        │
│  ├── chromadb-bench (fresh instance, port 8002)                 │
│  ├── redis-bench (isolated)                                      │
│  └── rag-server-bench (points to bench DBs)                     │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                evaluation/ module (extended)                     │
├─────────────────────────────────────────────────────────────────┤
│  datasets/                     # Dataset loaders                 │
│  ├── base.py                   # Abstract dataset interface     │
│  ├── squad.py                  # SQuAD 2.0 loader               │
│  ├── ragbench.py               # RAGBench loader                │
│  └── hotpotqa.py               # HotpotQA loader                │
│                                                                  │
│  benchmark.py                  # Orchestrates load → run → save │
│  results_store.py              # SQLite for result persistence  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Results Dashboard (Streamlit)                 │
├─────────────────────────────────────────────────────────────────┤
│  - Run history with timestamps                                   │
│  - Metric trends over time (charts)                             │
│  - Per-dataset breakdown                                         │
│  - Drill-down to individual questions                           │
└─────────────────────────────────────────────────────────────────┘
```

### Component Recommendations

| Component | Recommendation | Why |
|-----------|----------------|-----|
| **Runner** | justfile commands | Already in use, familiar workflow |
| **Ephemeral infra** | docker-compose.bench.yml | Isolated from prod, easy teardown |
| **Dataset loading** | HuggingFace `datasets` lib | Standard, handles caching |
| **Results storage** | SQLite | Simple, no extra services, query-friendly |
| **Dashboard** | Streamlit | Python-native, fast to build, local |

### Dashboard Options

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **Streamlit** (recommended) | Python-native, quick setup, interactive | Basic styling | Ad-hoc local use |
| **Grafana + SQLite** | Powerful visualizations, time series | More setup overhead | Scheduled/CI runs |
| **Evidence** | Beautiful SQL-based reports | Node.js dependency | Shareable reports |
| **Static HTML (Jinja2)** | Zero dependencies to view | No interactivity | CI artifacts |

### Workflow

1. **Load dataset**: `just bench-load squad` downloads and prepares SQuAD 2.0
2. **Start ephemeral env**: Spins up isolated ChromaDB + Redis
3. **Ingest corpus**: Uploads dataset documents to ephemeral ChromaDB
4. **Run queries**: Executes all test questions against the RAG
5. **Compute metrics**: DeepEval metrics + retrieval metrics (Recall@K, MRR)
6. **Store results**: Saves to SQLite with timestamp and run metadata
7. **Teardown**: Stops ephemeral containers
8. **View results**: `just bench-dashboard` launches Streamlit UI

### Alternative Configurations

**For CI/CD integration** (change: frequency → "On every PR"):
- Use smaller sample sizes (100-500 questions)
- Output JSON artifacts instead of dashboard
- Add to `.forgejo/workflows/ci.yml` as optional job

**For comprehensive overnight runs** (change: runtime → "Hours"):
- Run full datasets (100K+ questions)
- Use scheduled Forgejo workflow
- Store detailed per-question results for analysis

**For quick CLI feedback** (change: results display → "CLI only"):
- Skip SQLite storage
- Print summary table to terminal
- Useful for rapid iteration
