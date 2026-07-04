# RAG Evaluation Dashboard Design Research

This document captures research conducted for the dashboard refactoring, including best practices, recommended tools, and well-designed examples for displaying RAG/LLM evaluation metrics.

## Executive Summary

The dashboard was designed to support:
- **4-8 evaluation runs** compared simultaneously
- **LLM model comparison** as the primary use case
- **Live monitoring** workflow with auto-refresh
- **Baseline comparison** as a critical feature

---

## Key RAG Evaluation Metrics

Industry-standard metrics for RAG evaluation (source: [Confident AI](https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more)):

| Metric | Category | Description |
|--------|----------|-------------|
| **Answer Relevancy** | Generation | How relevant the generated response is to the input query |
| **Faithfulness** | Generation | Whether the response contains hallucinations vs retrieval context |
| **Contextual Relevancy** | Retrieval | How relevant the retrieved context is to the input |
| **Contextual Recall** | Retrieval | Whether context contains all information for ideal output |
| **Contextual Precision** | Retrieval | Whether retrieved context is ranked correctly (higher relevancy first) |

Additional metrics from [Meilisearch](https://www.meilisearch.com/blog/rag-evaluation):
- **NDCG Score**: Normalized Discounted Cumulative Gain for ranking quality
- **DCG Metric**: Discounted Cumulative Gain for retrieved results ordering

---

## Recommended Dashboard Tools & Examples

### Open Source Platforms

#### 1. MLflow Tracking UI
**URL**: https://mlflow.org/docs/latest/ml/tracking/

**Key Features**:
- Table View: Tabular overview with run name, duration, metrics
- Chart View: Graphical comparison of parameters and metrics
- Side-by-side run comparison with sortable columns
- Experiment-based run listing

**Best For**: Teams already using MLflow for ML lifecycle management

#### 2. Langfuse
**URL**: https://langfuse.com/docs/evaluation/overview

**Key Features**:
- Most popular open-source LLM observability tool
- Comprehensive tracing, evaluations, prompt management
- Model and framework agnostic
- Self-hosting options
- OpenTelemetry integration

**Best For**: Production LLM applications needing full observability

#### 3. Phoenix (Arize AI)
**URL**: https://phoenix.arize.com/

**Key Features**:
- Open-source LLM observability platform
- Diagnosing, troubleshooting, and monitoring across lifecycle
- OpenTelemetry foundation prevents vendor lock-in
- Tracing + evaluation + prompt management

**Best For**: RAG applications needing strong evaluation features

#### 4. DeepEval
**URL**: https://github.com/confident-ai/deepeval

**Key Features**:
- Unit-test mindset for LLM evaluation (like Pytest for AI)
- Benchmark apps using datasets
- Compare with previous iterations
- Sharable cloud testing reports
- RAG-specific metrics built-in

**Best For**: CI/CD integration and regression testing

#### 5. TruLens
**URL**: https://www.trulens.org/

**Key Features**:
- Feedback functions score each pipeline stage
- Document retrieval to final generation tracking
- Automated checks for missing context, inaccurate info
- Specialized for RAG metrics (groundedness, context relevance, answer relevance)

**Best For**: RAG-specific evaluation with detailed pipeline insights

### Commercial/Enterprise Solutions

#### 6. Weights & Biases (W&B)
**URL**: https://wandb.ai/site/tables/

**Key Features**:
- Central dashboard for ML experiments
- Search, filter, sort, group results
- Side-by-side and merged table comparison
- Interactive WYSIWYG report editor
- Nested trace trees for multi-agent workflows
- W&B Weave for LLM-specific features

**UI Patterns Worth Noting**:
- Tables with filter, group, sort operations applied in tandem
- View comparisons: merged view OR side-by-side view
- Charts: line plots, bar plots, scatter plots
- Rich media visualization (images, audio, text)

#### 7. LangSmith
**URL**: https://smith.langchain.com/

**Key Features**:
- Official platform by LangChain team
- Framework-agnostic despite LangChain integration
- Detailed traces of every application run
- Version control for datasets and regression tests
- Evaluators: LLM-as-judge, gold-standard, programmatic, pairwise

#### 8. Datadog LLM Observability
**URL**: https://www.datadoghq.com/product/llm-observability/

**Key Features**:
- Playground for prompt tweaks and model swaps
- Benchmark performance across configurations
- Production-grade monitoring
- Integrates with existing Datadog infrastructure

#### 9. Braintrust
**URL**: https://www.braintrust.dev/articles/best-rag-evaluation-tools

**Key Features**:
- Evaluation-first approach
- Experiment framework: define dataset, run variations, compare side-by-side
- Strong for systematic prompt iteration

---

## Visualization Best Practices

### Chart Type Selection

| Chart Type | Best For | Limitations |
|------------|----------|-------------|
| **Grouped Bar Chart** | Comparing 4-8 runs across metrics | Gets crowded with >10 runs |
| **Radar/Spider Chart** | Multi-variable comparison at a glance | Max 4-5 entities, 10-12 axes |
| **Heatmap** | Correlation matrices, metric-by-run grids | Needs clear color scale |
| **Parallel Coordinates** | Many variables across runs, trade-off visualization | Complex to interpret |
| **Small Multiples** | Individual charts per entity to avoid clutter | Requires more space |
| **Sparklines** | Inline trend visualization | Limited detail |

### Radar/Spider Chart Guidelines

Source: [Highcharts Guide](https://www.highcharts.com/blog/tutorials/radar-chart-explained-when-they-work-when-they-fail-and-how-to-use-them-right/)

**When to Use**:
- Comparing 2-4 entities across 5-10 variables
- Showing strengths/weaknesses profiles
- Performance evaluations with multiple dimensions

**When to Avoid**:
- More than 10-12 axes (labels become crowded)
- More than 4-5 entities (overlapping polygons create noise)
- When precise value comparison is needed

**Best Practices**:
- Apply transparency to overlapping regions
- Maintain consistent scaling across all axes
- Use small multiples for many entities
- Combine with other chart types for detailed views

### Heatmap Guidelines

Source: [Datylon](https://www.datylon.com/blog/types-of-charts-graphs-examples-data-visualization)

**Best For**:
- Activity-by-hour matrices
- Correlation matrices
- Feature-importance grids
- Spotting hotspots quickly

**Best Practices**:
- Sequential palette for magnitude (low→high)
- Diverging palette when values center around a midpoint
- Use colorblind-friendly palettes
- Add numeric labels or tooltips for precision
- Keep cell padding to prevent crowding

---

## Dashboard Design Patterns

### Key UI Components

Based on analysis of [Neptune](https://neptune.ai/blog/the-best-tools-for-machine-learning-model-visualization) and [W&B](https://wandb.ai/site/tables/):

1. **Comparison Table View**
   - Sortable columns for each parameter and metric
   - Highlight differences across runs
   - Filter and group capabilities

2. **Chart View**
   - Visualize comparisons with scatter plots, parallel coordinates
   - Interactive (download, zoom, remove experiments)
   - Legend-based filtering

3. **Metrics Dashboard**
   - Track accuracy, latency, resource usage
   - Real-time updates with anomaly detection
   - Regression charts showing retrieval vs generation quality

4. **Experiment Playground**
   - Test prompt tweaks, swap models
   - A/B testing support
   - Canary releases and rollback

### Layout Recommendations

Source: [LinkedIn - ML Dashboard Design](https://www.linkedin.com/advice/1/what-best-ways-design-machine-learning-dashboard-f3v5e)

1. **Prioritize key insights** - Most important metrics prominent
2. **Clear visualizations** - Appropriate chart type for data
3. **User-friendly controls** - Interactive exploration
4. **Simplicity in design** - Focus on relevant information
5. **Interactive features**:
   - Filtering
   - Drill-down capabilities
   - Hover-over tooltips

### Real-time Monitoring

- Put latency percentiles and accuracy on the same dashboard
- Speed trade-offs should be immediately visible
- Use regression charts to show if retrieval changes improved final answer quality

---

## Implementation Recommendations

### For This Project

Based on user requirements (4-8 runs, LLM model comparison, live monitoring):

1. **Primary Chart**: Grouped bar chart
   - Metrics as x-axis groups
   - Each run as a colored bar
   - Baseline as horizontal dashed reference line

2. **Run Labels**: Model + timestamp (e.g., "gemma3 14:30")

3. **Config Diff**: Git-style with green/red/yellow highlighting

4. **Layout**: Tabbed interface
   - Comparison (primary)
   - History
   - Config
   - System Health

5. **Export**: CSV/JSON for external analysis

### Technology Stack

- **Chart.js 4.x** - Popular, good bar chart support, ~60KB
- **chartjs-plugin-annotation** - For baseline reference lines
- **Svelte 5 actions** - Clean integration pattern (similar to existing Sparkline)

---

## Research Papers

Recent academic work on RAG evaluation:

1. **"Evaluation of Retrieval-Augmented Generation: A Survey"** (2024)
   - Authors: Hao Yu et al.
   - Comprehensive survey of RAG evaluation methods

2. **"Evaluating Retrieval Quality in Retrieval-Augmented Generation"** (2024)
   - Authors: A. Salemi et al.
   - Focus on retrieval component evaluation

3. **"Retrieval Augmented Generation Evaluation in the Era of Large Language Models"** (April 2025)
   - Authors: Aoran Gan et al.
   - Latest approaches to RAG evaluation

4. **"The Viability of Crowdsourcing for RAG Evaluation"** (April 2025)
   - Authors: Lukas Gienapp et al.
   - Human evaluation approaches

---

## Key References

### RAG Evaluation Best Practices
- [Mastering RAG Evaluation: Best Practices & Tools for 2025](https://orq.ai/blog/rag-evaluation)
- [Best 9 RAG Evaluation Tools of 2025](https://www.deepchecks.com/best-rag-evaluation-tools/)
- [RAG Evaluation: Metrics, Methodologies, Best Practices](https://www.meilisearch.com/blog/rag-evaluation)
- [Best Practices in RAG Evaluation - Qdrant](https://qdrant.tech/blog/rag-evaluation-guide/)
- [The 5 Best RAG Evaluation Tools in 2025 - Braintrust](https://www.braintrust.dev/articles/best-rag-evaluation-tools)

### LLM Observability & Dashboards
- [Building an LLM Evaluation Framework - Datadog](https://www.datadoghq.com/blog/llm-evaluation-framework-best-practices/)
- [LLM Observability Tools: 2026 Comparison](https://lakefs.io/blog/llm-observability-tools/)
- [Best LLM Evaluation Tools - ZenML](https://www.zenml.io/blog/best-llm-evaluation-tools)
- [LLM Observability: Fundamentals, Practices, and Tools - Neptune](https://neptune.ai/blog/llm-observability)

### ML Experiment Tracking
- [Intro to MLOps: ML Experiment Tracking - W&B](https://wandb.ai/site/articles/intro-to-mlops-machine-learning-experiment-tracking/)
- [ML Model Monitoring Dashboard Tutorial - Evidently AI](https://www.evidentlyai.com/blog/ml-model-monitoring-dashboard-tutorial)
- [Machine Learning Experiment Tracking Guide - DagsHub](https://dagshub.com/blog/machine-learning-experiment-tracking-your-ultimate-guide/)
- [7 Best Tools for ML Experiment Tracking - KDnuggets](https://www.kdnuggets.com/2023/02/7-best-tools-machine-learning-experiment-tracking.html)

### Data Visualization
- [Radar Charts: Best Practices and Examples - Bold BI](https://www.boldbi.com/blog/radar-charts-best-practices-and-examples/)
- [24 Types of Charts for Data Visualization - ThoughtSpot](https://www.thoughtspot.com/data-trends/data-visualization/types-of-charts-graphs)
- [80 Types of Charts & Graphs - Datylon](https://www.datylon.com/blog/types-of-charts-graphs-examples-data-visualization)
- [The Radar Chart and its Caveats](https://www.data-to-viz.com/caveat/spider.html)

---

## Conclusion

The dashboard design follows industry best practices from leading ML/LLM observability tools. Key decisions:

1. **Grouped bar chart** chosen over radar charts due to better precision for comparing values
2. **Tabbed layout** to organize dense information without excessive scrolling
3. **Baseline as reference line** provides immediate visual comparison
4. **Git-style config diff** for clear change visibility
5. **Auto-append runs** supports live monitoring workflow

The implementation uses Chart.js for flexibility and bundle size optimization, following patterns established by tools like MLflow, W&B, and DeepEval.
