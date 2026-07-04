# RAG Analytics Dashboard Research

Comprehensive research for designing an outstanding analytics dashboard to display RAG evaluation metrics. This document covers metric fields, best visualization practices, and curated examples of outstanding analytics UIs.

---

## Part 1: RAG Evaluation Metrics Catalog

### Currently Implemented Metrics

| Metric | Category | Type | Threshold | Description |
|--------|----------|------|-----------|-------------|
| `contextual_precision` | Retrieval | float (0-1) | 0.7 | Are retrieved chunks relevant to the query? |
| `contextual_recall` | Retrieval | float (0-1) | 0.7 | Did we retrieve all needed information? |
| `faithfulness` | Generation | float (0-1) | 0.7 | Is the answer grounded in retrieved context? |
| `answer_relevancy` | Generation | float (0-1) | 0.7 | Does the answer address the question? |
| `hallucination` | Safety | float (0-1) | 0.5 | Rate of unsupported information (lower = better) |

### Performance Metrics (Implemented)

| Metric | Category | Type | Description |
|--------|----------|------|-------------|
| `avg_query_time_ms` | Latency | float | Average query response time |
| `p50_query_time_ms` | Latency | float | Median (50th percentile) |
| `p95_query_time_ms` | Latency | float | 95th percentile latency |
| `min_query_time_ms` | Latency | float | Minimum response time |
| `max_query_time_ms` | Latency | float | Maximum response time |
| `total_input_tokens` | Cost | int | Total input tokens used |
| `total_output_tokens` | Cost | int | Total output tokens generated |
| `estimated_cost_usd` | Cost | float | Estimated cost in USD |
| `cost_per_query_usd` | Cost | float | Average cost per query |

### Aggregate Metrics (Implemented)

| Metric | Type | Description |
|--------|------|-------------|
| `pass_rate` | float (0-100) | Overall test pass percentage |
| `metric_averages` | dict[str, float] | Average score per metric |
| `metric_pass_rates` | dict[str, float] | Pass rate per metric |
| `trend_direction` | string | "improving", "declining", "stable" |

### Configuration Variables to Track

| Variable | Category | Values | Description |
|----------|----------|--------|-------------|
| `llm_provider` | LLM | ollama, openai, anthropic, google, deepseek, moonshot | LLM provider |
| `llm_model` | LLM | string | Model name (e.g., gemma3:4b) |
| `embedding_model` | Embedding | string | Embedding model name |
| `hybrid_search_enabled` | Retrieval | boolean | BM25 + Vector enabled |
| `rrf_k` | Retrieval | int | RRF fusion constant (default: 60) |
| `contextual_retrieval_enabled` | Retrieval | boolean | Anthropic contextual retrieval |
| `reranker_enabled` | Reranker | boolean | Reranking enabled |
| `reranker_model` | Reranker | string | Reranker model name |
| `retrieval_top_k` | Retrieval | int | Initial chunks to retrieve |
| `reranker_top_n` | Reranker | int | Results after reranking |

### Recommended Additional Metrics

| Metric | Category | Source | Description |
|--------|----------|--------|-------------|
| `ndcg` | Retrieval | Industry standard | Normalized Discounted Cumulative Gain |
| `mrr` | Retrieval | Industry standard | Mean Reciprocal Rank |
| `hit_rate@k` | Retrieval | Industry standard | % of queries where relevant doc in top-k |
| `answer_correctness` | Generation | DeepEval | Factual accuracy vs ground truth |
| `coherence` | Generation | TruLens | Logical flow and consistency |
| `toxicity` | Safety | TruLens | Harmful content detection |
| `time_to_first_token` | Latency | Helicone | TTFT for streaming responses |
| `tokens_per_second` | Throughput | Common | Generation speed |

---

## Part 2: Best Practices for Communicating RAG Metrics

### Information Hierarchy

**Level 1 - Overview (Top of Dashboard)**
- Pass rate as large KPI number with trend arrow
- Overall health status indicator (green/yellow/red)
- Latest evaluation timestamp

**Level 2 - Key Metrics (Primary View)**
- Grouped bar chart comparing runs across all 5 core metrics
- Baseline reference line for comparison
- Quick model + config identifier

**Level 3 - Trends (Secondary View)**
- Time-series line charts for metric trends
- Sparklines for compact trend visualization
- Trend direction indicators

**Level 4 - Details (Drill-Down)**
- Per-test-case results table
- Configuration diff view
- Full run metadata

### Chart Selection Guide

| Data Type | Recommended Chart | When to Use |
|-----------|------------------|-------------|
| Compare 4-8 runs across metrics | Grouped Bar Chart | Primary comparison view |
| Show metric trends over time | Line Chart with points | History/trends tab |
| Display multi-dimensional profile | Radar/Spider Chart | Max 4 entities, 5-10 axes |
| Show metric-by-run grid | Heat Map | Large comparison matrices |
| Highlight trade-offs | Parallel Coordinates | Advanced analysis |
| Compact trend indicators | Sparklines | Inline in tables |
| Single KPI with trend | Big Number + Arrow | Dashboard headers |

### Visual Design Principles

1. **Color Coding**
   - Green: Passing/improving metrics
   - Red: Failing/declining metrics
   - Yellow: Warning threshold
   - Gray: Baseline/reference

2. **Baseline Visualization**
   - Horizontal dashed line on bar charts
   - "★" indicator for golden baseline run
   - Green/red delta values vs baseline

3. **Run Identification**
   - Format: `{model_name} {HH:MM}` (e.g., "gemma3 14:30")
   - Show full config on hover
   - Git-style config diff with color highlighting

4. **Layout**
   - Tabbed interface: Comparison | History | Config | System
   - Maximum 5-6 cards in initial view
   - Single-screen focus for key insights

### Comparison Workflows

**Workflow 1: Model A/B Testing**
- Side-by-side grouped bars
- Highlight winning metric per category
- Show overall winner with reasoning

**Workflow 2: Configuration Optimization**
- Config diff view (what changed)
- Metric impact arrows (↑ improved, ↓ declined)
- Recommendation engine output

**Workflow 3: Regression Detection**
- Comparison to golden baseline
- Pass/fail badges per metric
- Alert on any regression

---

## Part 3: Outstanding Analytics Dashboard Examples

### LLM/RAG Observability Platforms

| Platform | URL | Stack | What Makes It Outstanding |
|----------|-----|-------|--------------------------|
| **Langfuse** | [langfuse.com](https://langfuse.com) | React, Recharts | Clean trace inspection UI, cost/latency dashboards, open-source. Polished dark mode with consistent design system. |
| **Phoenix (Arize)** | [phoenix.arize.com](https://phoenix.arize.com) | React, Plotly | Notebook-first approach, hierarchical span visualization, built-in evaluators. Timeline view excels at showing agent execution paths. |
| **Helicone** | [helicone.ai](https://www.helicone.ai) | Next.js, Recharts | Beautiful cost trend visualizations, clean metric cards, one-line integration. Focuses on developer experience. |
| **Portkey** | [portkey.ai](https://portkey.ai) | Next.js | Real-time observability dashboard, centralized prompt management. Strong on routing visualization. |
| **DeepEval (Confident AI)** | [confident-ai.com](https://www.confident-ai.com) | React | Unit-test mindset for LLM evaluation, sharable cloud reports. Focuses on metric comparison across iterations. |
| **TruLens** | [trulens.org](https://www.trulens.org) | Python, Streamlit | Feedback function visualization, metrics leaderboard, side-by-side comparison. Specialized for RAG evaluation. |
| **Opik (Comet)** | [comet.com](https://www.comet.com) | React | High-level performance view with drill-down, comparison to previous runs. Clean tabular layouts. |

### ML Experiment Tracking

| Platform | URL | Stack | What Makes It Outstanding |
|----------|-----|-------|--------------------------|
| **Weights & Biases** | [wandb.ai](https://wandb.ai) | React, D3.js, Plotly | Industry leader in experiment visualization. Real-time monitoring, sophisticated customization, interactive dashboards. Side-by-side and merged table views. |
| **MLflow** | [mlflow.org](https://mlflow.org) | React, Plotly | Open-source standard. Tabular overview with run comparison, chart view for parameter analysis. Sortable columns, experiment-based grouping. |
| **Neptune.ai** | [neptune.ai](https://neptune.ai) | React | Lightweight yet powerful. Excellent comparison tables, metadata tracking, flexible queries. |
| **Comet ML** | [comet.com](https://www.comet.com) | React | Track datasets, code changes, experimentation history. Strong on image/audio/text sample visualization. |
| **DagsHub** | [dagshub.com](https://dagshub.com) | React | Git-like experiment management. Excellent diff views, version comparison. |

### LLM Leaderboards & Benchmarks

| Platform | URL | Stack | What Makes It Outstanding |
|----------|-----|-------|--------------------------|
| **HuggingFace Open LLM Leaderboard** | [huggingface.co/spaces/open-llm-leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) | Gradio, React | Interactive filtering, real-time model comparison, benchmark breakdowns. 2M+ visitors. Model comparator tool is excellent. |
| **Artificial Analysis** | [artificialanalysis.ai](https://artificialanalysis.ai/leaderboards/models) | Next.js | Clean comparison of 100+ models across price, performance, speed. Interactive scatter plots for trade-off analysis. |
| **Vellum LLM Leaderboard** | [vellum.ai/llm-leaderboard](https://www.vellum.ai/llm-leaderboard) | React | SOTA model tracking, independent evaluations, clear benchmark categories. |

### Product Analytics

| Platform | URL | Stack | What Makes It Outstanding |
|----------|-----|-------|--------------------------|
| **PostHog** | [posthog.com](https://posthog.com) | React, D3.js | Open-source, all-in-one platform. Jump from graph to session recording. Dashboard templates for various use cases. Developer-first design. |
| **Mixpanel** | [mixpanel.com](https://mixpanel.com) | React | Intuitive data perception, hybrid dashboard/storytelling interface. Enriched with charts, text, and media. Highly collaborative. |
| **Amplitude** | [amplitude.com](https://amplitude.com) | React | Excellent cohort and path visualization. Separate notebooks for analysis. Deep behavioral analytics. |

### Infrastructure & DevOps

| Platform | URL | Stack | What Makes It Outstanding |
|----------|-----|-------|--------------------------|
| **Grafana** | [grafana.com](https://grafana.com) | React, D3.js | Industry standard for infrastructure monitoring. 40+ visualization types, dynamic templating, real-time updates. Excellent plugin ecosystem. |
| **Datadog** | [datadoghq.com](https://www.datadoghq.com) | React | Unified metrics, logs, traces. 900+ integrations. Drag-and-drop dashboards, CI pipeline visualization. Service map and network map are outstanding. |
| **Vercel Analytics** | [vercel.com/analytics](https://vercel.com/analytics) | Next.js, Recharts | Developer-focused speed insights, Core Web Vitals, privacy-friendly. Clean, minimal design that matches their brand. |

### BI & Data Visualization

| Platform | URL | Stack | What Makes It Outstanding |
|----------|-----|-------|--------------------------|
| **Tableau Public** | [public.tableau.com](https://public.tableau.com) | Proprietary | 9M+ visualizations, daily Viz of the Day. KPI scorecards trend in 2024. Enclosure-based organization, custom legends, network diagrams. |
| **Apache Superset** | [superset.apache.org](https://superset.apache.org) | React, D3.js, ECharts | Open-source BI with 40+ visualization types. Plugin architecture for custom viz. SQL IDE for analysts. |
| **Metabase** | [metabase.com](https://www.metabase.com) | React, D3.js | Simple setup, automatic data exploration. Clean question-based interface. Great for non-technical users. |

### Design Inspiration Sources

| Source | URL | What Makes It Outstanding |
|--------|-----|--------------------------|
| **SaaS Interface** | [saasinterface.com](https://saasinterface.com/pages/dashboard/) | 142+ SaaS dashboard examples. Categorized by type, searchable. Real production examples. |
| **Dribbble Analytics** | [dribbble.com/tags/analytics-dashboard](https://dribbble.com/tags/analytics-dashboard) | 1,600+ designs. Emerging trends, color palettes, experimental concepts. |
| **Muzli Dashboard Inspiration** | [muz.li/inspiration/dashboard-inspiration](https://muz.li/inspiration/dashboard-inspiration) | 60+ curated examples. Credit score, finance, HR, sales dashboards. |
| **Observable D3 Gallery** | [observablehq.com/@d3/gallery](https://observablehq.com/@d3/gallery) | Official D3.js examples. Interactive notebooks, cutting-edge techniques. |
| **Shadcn Dashboard** | [ui.shadcn.com/examples/dashboard](https://ui.shadcn.com/examples/dashboard) | Modern React components. Clean, accessible, highly customizable. |
| **Tremor Blocks** | [blocks.tremor.so](https://blocks.tremor.so) | Vercel-backed dashboard components. Tailwind + Recharts. Production-ready. |

### SaaS Dashboard Design Leaders

| Product | What Makes It Outstanding |
|---------|--------------------------|
| **Linear** | Dark mode elegance, Inter typography, gradient accents, professional "engineering tool" vibe. Influenced "Linear design" trend. |
| **Stripe Dashboard** | Financial metrics clarity, real-time balance, revenue charts. Component-based extensibility via Stripe Apps. |
| **Notion** | Clean minimalism, flexible layouts, excellent information hierarchy. |
| **Figma** | Collaborative design, layer panels, properties inspectors. Strong on real-time updates. |

---

## Part 4: Technology Stack Recommendations

### Charting Libraries Comparison

| Library | Bundle Size | Best For | Framework | Rendering |
|---------|-------------|----------|-----------|-----------|
| **Chart.js** | ~60KB | Simple charts, quick setup | Any | Canvas |
| **Recharts** | ~120KB | React dashboards, Shadcn/Tremor | React | SVG |
| **ECharts** | ~300KB | Large datasets, enterprise, maps | Any | Canvas/WebGL |
| **ApexCharts** | ~140KB | Interactive charts, modern UX | Any | SVG |
| **D3.js** | ~80KB (core) | Custom visualizations, full control | Any | SVG/Canvas |
| **Plotly** | ~500KB | Scientific, 3D, interactive | Any | WebGL/SVG |

### For This Project (SvelteKit)

**Recommended Stack:**
- **Primary**: Chart.js 4.x (currently used, good ecosystem)
- **Plugin**: chartjs-plugin-annotation (baseline lines)
- **Alternative**: Consider Recharts via svelte-recharts or Layer Cake for Svelte-native

**Svelte-Specific Options:**
- **Layer Cake**: Svelte-native, composable, responsive charts
- **Pancake**: By Svelte creator, SVG-based
- **svelte-chartjs**: Chart.js wrapper for Svelte

---

## Part 5: Implementation Recommendations

### Dashboard Layout (Tabbed)

```
┌─────────────────────────────────────────────────────────────┐
│  [Comparison] [History] [Config] [System Health]            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────────────────────────────┐  │
│  │ Pass Rate   │  │                                      │  │
│  │   85%  ↑5%  │  │     Grouped Bar Chart                │  │
│  └─────────────┘  │     (5 metrics × n runs)             │  │
│  ┌─────────────┐  │     + Baseline reference line        │  │
│  │ Latest Run  │  │                                      │  │
│  │ gemma3 14:30│  │                                      │  │
│  └─────────────┘  └──────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Run Selector: [✓] Run A  [✓] Run B  [★] Baseline       ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Key Visual Elements

1. **KPI Cards**: Big numbers with trend arrows, color-coded
2. **Grouped Bar Chart**: Primary comparison, 4-8 runs max
3. **Baseline Reference**: Dashed horizontal line per metric
4. **Run Labels**: `{model} {time}` format
5. **Config Diff**: Git-style green/red/yellow highlighting
6. **Sparklines**: Inline trends in history table

### Export Options

- CSV: For spreadsheet analysis
- JSON: For programmatic access
- PNG: Chart screenshots

---

## Sources

### LLM Observability & Evaluation
- [Langfuse Documentation](https://langfuse.com/docs)
- [Phoenix Arize GitHub](https://github.com/Arize-ai/phoenix)
- [Helicone Blog](https://www.helicone.ai/blog)
- [DeepEval Documentation](https://docs.confident-ai.com)
- [TruLens](https://www.trulens.org/)

### ML Experiment Tracking
- [Weights & Biases Articles](https://wandb.ai/site/articles/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [DagsHub ML Experiment Tracking Guide](https://dagshub.com/blog/best-8-experiment-tracking-tools-for-machine-learning-2023/)

### Dashboard Design
- [Muzli Dashboard Inspirations 2024](https://muz.li/blog/dashboard-design-inspirations-in-2024/)
- [Eleken Dashboard Design Examples](https://www.eleken.co/blog-posts/dashboard-design-examples-that-catch-the-eye)
- [Toptal Data Visualization Examples](https://www.toptal.com/designers/dashboard-design/top-data-visualization-dashboard-examples)
- [Pencil & Paper Dashboard UX Patterns](https://www.pencilandpaper.io/articles/ux-pattern-analysis-data-dashboards)

### Charting Libraries
- [LogRocket Best React Chart Libraries 2025](https://blog.logrocket.com/best-react-chart-libraries-2025/)
- [Embeddable JavaScript Charting Libraries](https://embeddable.com/blog/javascript-charting-libraries)
- [Observable D3.js Gallery](https://observablehq.com/@d3/gallery)

### Visualization Best Practices
- [Metabase Time Series Visualization](https://www.metabase.com/blog/how-to-visualize-time-series-data)
- [Grafana Time Series Documentation](https://grafana.com/docs/grafana/latest/panels-visualizations/visualizations/time-series/)
- [Highcharts Radar Chart Guide](https://www.highcharts.com/blog/tutorials/radar-chart-explained-when-they-work-when-they-fail-and-how-to-use-them-right/)
- [Datylon Chart Types](https://www.datylon.com/blog/types-of-charts-graphs-examples-data-visualization)

### Product Analytics
- [PostHog Documentation](https://posthog.com/docs)
- [Mixpanel vs Amplitude Comparison](https://mixpanel.com/compare/amplitude/)
- [Vercel Dashboard Redesign](https://vercel.com/blog/dashboard-redesign)

### UI Component Libraries
- [Shadcn/ui Charts](https://ui.shadcn.com/docs/components/chart)
- [Tremor React Components](https://www.tremor.so/)
- [Tableau Public Gallery](https://public.tableau.com/app/discover)
