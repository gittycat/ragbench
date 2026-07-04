# Eval Results Dashboard Redesign — Implementation Plan
2026-07-04

Self-contained handoff plan. Read the **Trip-up points** section fully before writing any code — every item is a place where a straightforward implementation goes wrong.

## Status

- **Part 1 (Backend evals API extensions): DONE.** Implemented and tested, see below. Do not redo.
- **Part 2 (Frontend redesigned analytics page): NOT STARTED.** Next up.
- **Part 3 (Styling): NOT STARTED.**

### Part 1 completion notes
Files changed (all in `services/evals/`):
- `api/schemas.py` — `DashboardMetrics` gained `latency_avg_seconds`, `avg_cost_usd`, `total_cost_usd`, `total_prompt_tokens`, `total_completion_tokens`, `cost_model` (all Optional/default None). `RunSummary` gained `metrics: dict[str, float]` (flat, all scorecard metrics) and `groups: dict[str, list[str]]` (from `scorecard.by_group`), both defaulting empty.
- `api/dashboard.py` — `compute_dashboard_metrics()` now also pulls the `cost_per_query` metric's `details{}` to populate the new telemetry fields (trip-up #4), plus `latency_avg_ms` → `latency_avg_seconds`.
- `api/job_manager.py` — `run_to_summary()` now builds the flat `metrics` map and `groups` dict from the scorecard and passes them into `RunSummary`.
- `api/routes.py` — `compare_runs()` deltas now computed over the **union** of both runs' flat metric names (via `run_to_summary`), plus a `weighted_score` delta; missing-on-one-side → `null` (trip-up #6). No new endpoint added, per plan.
- `services/rag_server/tests/test_metrics_api.py` — checked; already clean, no dead-endpoint expectations existed to remove (trip-up #2 was a non-issue in current tree).
- New `services/evals/tests/test_api.py` (8 tests, all passing) covering: flat metrics map + groups, empty-scorecard defaults, telemetry derivation (including missing cost details), retrieval null for generation tier, widened compare deltas with a metric missing on one side, weighted_score delta, and a legacy run dict (no new fields) still parsing.
- Run tests with: `cd services/evals && DEEPEVAL_TELEMETRY_OPT_OUT=YES uv run pytest tests/test_api.py -v` (the `DEEPEVAL_TELEMETRY_OPT_OUT=YES` works around a sandbox network/cert issue triggered by importing `evals.runner`/deepeval at collection time — unrelated to this change).
- Pre-existing unrelated failures: `tests/test_rag_eval.py` has ~20 failing tests (citation extraction, dataset loaders, query endpoint mocks) not touched by this change — environment/sandbox-dependent, not introduced here.
- **Caution for next session:** avoid `git stash` in this repo/sandbox — `docs/archives/*` and `secrets/*` are sandbox-denied paths and stash creation fails partway, which can leave a pre-existing stash entry (`stash@{0}`, unrelated old WIP) sitting on top of the stack; a subsequent `git stash pop` will apply *that* old stash instead of a new one and conflict with `services/rag_server` files. Use `git diff`/targeted `git checkout -- <path>` instead of stash for throwaway comparisons.

## Context

The webapp analytics page (`services/webapp/src/routes/analytics/`) is **currently broken**: it calls `/api/metrics/evaluation/*`, `/api/metrics/baseline*`, `/api/metrics/compare*`, `/api/metrics/recommend` — endpoints that were **removed** from rag_server (`services/rag_server/api/routes/metrics.py` now only serves `/metrics/system`, `/metrics/models`, `/metrics/retrieval`).

The live eval data comes from the **evals service** (port 8002), proxied at `/api/eval/*` by `services/webapp/src/hooks.server.ts`. Its run files hold ~20 metrics in 5 groups plus weighted score and cost/token telemetry, of which the current API surfaces only 6 derived numbers.

**Goal:** redesign the dashboard around the `/eval/*` API with a dense monospace/terminal observability aesthetic (Grafana-like: compact tables, monospace numerics, muted grays, green/amber/red threshold accents), working in both DaisyUI themes (`nord` light / `dim` dark).

**Confirmed user decisions:**
- Show: run-level scorecards, run comparison + trends, cost & token telemetry. **No per-question drill-down.**
- Drop the golden-baseline UI and RecommendationPanel entirely (their endpoints are dead). Do NOT re-implement them.
- Extending the evals API is preferred (small additive changes only).
- Keep the System Health tab, restyled.

---

## Trip-up points (read before coding)

### Backend / data model

1. **Two result formats exist; only one is live.** The frontend types in `src/lib/api.ts` (`EvaluationRun`, `metric_averages`, `pass_rate`, `MetricResult`) describe the legacy DeepEval format. The live evals API returns `scorecard.metrics[]` as `{name, value, group, sample_size, details{}}`. **Replace the old types, don't map them** — fields like `pass_rate` and `metric_averages` have no equivalent in the new format; do not synthesize them.
2. **Dead endpoints must be deleted, not fixed.** `/metrics/evaluation/*`, `/metrics/baseline*`, `/metrics/compare*`, `/metrics/recommend` no longer exist in rag_server. Do NOT re-add them there. The correct backend is the evals service (`/eval/*`). Delete the dead client functions and the components that depend on them (`BaselineIndicator.svelte`, `RecommendationPanel.svelte`, orphaned `RunComparison.svelte`), plus the stale expectations in `services/rag_server/tests/test_metrics_api.py`.
3. **Proxy path arithmetic.** Client calls `/api/eval/runs` → `hooks.server.ts` strips `/api` → evals service serves `/eval/runs`. Easy to double- or under-prefix.
4. **Cost/tokens are not top-level metrics.** They live in `details{}` of the `cost_per_query` metric: `avg_cost_usd`, `total_cost_usd`, `total_prompt_tokens`, `total_completion_tokens`, `model`. Surfacing them means digging into scorecard details in `api/dashboard.py` — there is no `tokens` metric name to look up.
5. **Metric-name traps.** The runner writes `latency_p50_ms` / `latency_p95_ms` / `latency_avg_ms` into the scorecard (the standalone `LatencyP50`/`LatencyP95` classes' names `latency_p50`/`latency_p95` are NOT what's stored). `answer_correctness` is displayed as "completeness" in `DashboardMetrics.answer_completeness`. Latency is **ms** in the scorecard but **seconds** in `DashboardMetrics`.
6. **Tier-dependent nulls.** Retrieval metrics don't exist for `generation`-tier runs (tiers: `generation`, `end_to_end`, in `metadata.tier`). Every retrieval cell must render as "—"/null, **never 0**. Compare deltas must return `null` (not raise KeyError) when a metric is missing on one side.
7. **`weighted_score` shape differs by endpoint.** Scalar float in `RunSummary`, full object `{score, weights, contributions, objectives}` in `RunDetailResponse`. Same field name — don't conflate.
8. **Backward compat with existing run files.** `JobManager` re-indexes old JSON files from `data/eval_runs/` at startup; all new schema fields must be Optional with `None`/empty defaults or startup breaks on old runs.

### Frontend

9. **Svelte 5 runes only.** The codebase uses `$state`/`$derived`/`$props()`/snippets. Do NOT write Svelte 4 (`export let`, `$:`, slots). Use the Svelte MCP server for any Svelte question (per CLAUDE.md).
10. **Tailwind v4 CSS-first config.** There is no `tailwind.config.js` — config lives in `src/app.css` via `@import "tailwindcss"; @plugin "daisyui" { themes: nord --default, dim --prefersdark; }`. Do not create a v3 config file.
11. **Theme-aware chart colors.** The current `ChartAction.ts` palette is hardcoded RGBA and breaks the nord/dim themes. New charts must derive colors from DaisyUI CSS variables at chart creation (`getComputedStyle(document.documentElement)`) and re-render when `data-theme` changes (MutationObserver on the attribute).
12. **Chart.js explicit registration.** `ChartAction.ts` registers only bar-chart pieces (CategoryScale, LinearScale, BarElement, BarController, Title, Tooltip, Legend, annotation plugin). Trend/line charts need `LineController`, `LineElement`, `PointElement` registered too, or you get a runtime "not a registered controller" error.
13. **Delta color direction.** Latency, cost, and `abstention_false_positive_rate`/`abstention_false_negative_rate` are **lower-is-better**; quality scores are higher-is-better. Green/red coloring in comparisons and threshold coloring must respect per-metric direction — the classic dashboard bug.
14. **Load the `dataviz` skill before writing any chart code** (harness requirement). `artifact-design` does not apply — this is app code, not an artifact.

### Scope guards

15. **No per-question drill-down.** Per-sample data is computed but never persisted in the live format — don't add UI or persistence for it.
16. **Baseline/recommendation features are dropped** per the user's decision — don't re-implement them "while at it". (The legacy `evals/data/golden_baseline.json` uses old DeepEval metric names — not reusable for thresholds.)
17. **Verification tooling:** `npm run build` + svelte-check in the webapp, and the evals service unit tests. There is **no frontend test suite** to rely on.

---

## Reference: live data shapes

Evals service run file (JSON in `data/eval_runs/`, indexed in-memory by `api/job_manager.py`):

```
id, name, created_at, completed_at
config{llm_model, llm_provider, embedding_model, reranker_model,
       retrieval_top_k, hybrid_search_enabled, contextual_retrieval_enabled}
datasets[]                      # e.g. ["ragbench"]
scorecard{
  metrics[]: {name, value, group, sample_size, details{}}
  by_group{group: [metric names]}
}
weighted_score{score, weights, contributions, objectives}
question_count, error_count, duration_seconds
metadata{samples_per_dataset, seed, tier}   # tier: generation | end_to_end
```

Metric names by group:
- **retrieval**: `recall_at_1/3/5/10`, `precision_at_1/3/5`, `mrr`, `ndcg_at_10` (absent for generation tier)
- **generation**: `faithfulness`, `answer_correctness`, `answer_relevancy` (details include `individual_scores[]`, `std_dev`, aggregate judge `reasoning`)
- **citation**: `citation_precision`, `citation_recall`, `section_accuracy`
- **abstention**: `unanswerable_accuracy`, `abstention_false_positive_rate`, `abstention_false_negative_rate`
- **performance**: `latency_p50_ms`, `latency_p95_ms`, `latency_avg_ms` (details: `min_ms`, `max_ms`), `cost_per_query` (details: `avg_cost_usd`, `total_cost_usd`, `total_prompt_tokens`, `total_completion_tokens`, `model`)

Weighted-score default objective weights: accuracy .30, faithfulness .20, citation .20, retrieval .15, cost .10, latency .05.

Existing endpoints (`services/evals/api/routes.py`, prefix `/eval`): `POST /runs`, `GET /runs/active` (progress: phase, current_question/total, elapsed), `DELETE /runs/active`, `GET /runs` (RunSummary list, limit≤100), `GET /runs/{id}` (RunDetailResponse), `GET /runs/compare?ids=` (details + deltas), `GET /dashboard` (latest summary + total_runs + active_job), `GET /datasets`.

---

## Part 1 — Backend: additive evals API extensions

Files: `services/evals/api/schemas.py`, `api/dashboard.py`, `api/job_manager.py`, `api/routes.py`.

1. **`RunSummary.metrics: dict[str, float]`** — flat `{metric_name: value}` of ALL scorecard metrics, built in `job_manager.run_to_summary()` (line ~229) from `scorecard.metrics[]`. Also add `groups: dict[str, list[str]]` from `scorecard.by_group` so the frontend can group without hardcoding. Defaults: empty dict (trip-up #8). One `GET /eval/runs?limit=50` then powers all trend charts and comparison tables — no new endpoint.
2. **Telemetry in `DashboardMetrics`** — extend `compute_dashboard_metrics()` in `api/dashboard.py` to read the `cost_per_query` metric's `details{}` (trip-up #4): add optional fields `avg_cost_usd`, `total_cost_usd`, `total_prompt_tokens`, `total_completion_tokens`, `latency_avg_seconds` (from `latency_avg_ms`/1000), `cost_model`. All Optional, default None.
3. **Widen `GET /eval/runs/compare` deltas** (`api/routes.py` ~line 126): compute `deltas` over the **union** of scorecard metric names of run[0] vs run[1] (second minus first, same semantics as today) + `weighted_score` delta + keep `duration_seconds`. Missing-on-one-side → `null` (trip-up #6). Use the flat metric map from step 1.
4. **Tests**: extend `services/evals/tests/` (pytest, `conftest.py` present) with unit tests using a synthetic scorecard dict fixture covering: flat metrics map + groups on RunSummary; telemetry derivation from `cost_per_query` details; widened deltas including a metric missing on one side; a legacy run dict (no new fields) still parsing.

No DB involved — runs are JSON files.

## Part 2 — Frontend: redesigned analytics page

### API client + types
- New `services/webapp/src/lib/api/evals.ts`: TS types mirroring evals schemas (`EvalRunSummary`, `EvalRunDetail`, `EvalDashboardMetrics`, `ActiveEvalJob`, `EvalCompareResponse`, `ScorecardMetric`) and fetchers `fetchEvalDashboard()`, `fetchEvalRuns(limit)`, `fetchEvalRun(id)`, `compareEvalRuns(ids)`, `fetchActiveEvalJob()` — all hitting `/api/eval/...` (trip-up #3).
- **Delete legacy** from `src/lib/api.ts`: `fetchEvaluationSummary`, `fetchEvaluationHistory`, baseline functions, `compareRuns`, `compareToBaseline`, `fetchRecommendation`, and types `EvaluationRun`, `GoldenBaseline`, `ComparisonResult`, `Recommendation`, `MetricTrend`, `EvaluationSummary`, `MetricResult`, `LatencyMetrics`, `CostMetrics`. **Keep** `SystemMetrics`/`ModelsConfig`/`RetrievalConfig`/`fetchSystemMetrics`/`fetchHealth` — still live.
- **Delete components**: `BaselineIndicator.svelte`, `RecommendationPanel.svelte`, `RunComparison.svelte`. Rework `utils/export.ts` around the new shapes (CSV/JSON of summaries + compare table).

### Page structure (`src/routes/analytics/+page.svelte` rewrite)

Persistent **header strip** (dense, monospace): service status dots (from `/metrics/system`), latest run id·name·tier·datasets badges, doc/chunk counts, **active-job progress** (poll `GET /api/eval/runs/active` every 5s while a job runs: phase, question i/N, elapsed), auto-refresh toggle, timestamp.

Four tabs (keep the `AnalyticsTabs.svelte` shell + `?tab=` URL sync, restyle):

1. **Scorecard** (new `ScorecardTab.svelte`) — run picker defaulting to latest; then:
   - Weighted score stat + horizontal **contribution bars** per objective from `weighted_score{score, weights, contributions}` (full object — this is the detail endpoint, trip-up #7).
   - Run meta line: tier, datasets, samples/dataset, seed, question_count, error_count (red if >0), duration.
   - **Metric-group tables** (dense, one per group): retrieval, generation (with ±std_dev and sample_size from details), citation, abstention. Values threshold-colored. Retrieval section shows "n/a (generation tier)" when absent (trip-up #6).
   - **Telemetry panel**: cost/query, total cost, prompt/completion/total tokens, judge model, latency p50/p95/avg/min/max.
   - **Config snapshot** card: llm model+provider, embedding, reranker, top_k, hybrid/contextual flags.
2. **Trends** (new `TrendsTab.svelte`) — from `GET /eval/runs?limit=50` summaries using the new `metrics` map: small-multiple line charts (weighted score, key quality metrics, latency p95, cost/query) across runs ordered by `created_at`; tier filter so generation and end_to_end series aren't silently mixed.
3. **Compare** (rework `ComparisonTab.svelte`) — reuse `RunSelector.svelte` (2–4 runs); side-by-side **dense table**: rows = all metrics grouped, columns = runs, delta column colored by per-metric direction (trip-up #13); grouped bar chart for quality metrics; reuse `ConfigDiff.svelte` + `utils/diff.ts` between the two selected runs; telemetry comparison rows; `ExportButton` rewired.
4. **System** (restyle `SystemHealthTab.svelte`) — keep models table, pipeline flow, index stats; remove the `RecommendationPanel` usage; apply the new visual language.

Delete `HistoryTab.svelte` (superseded by Trends) and `ConfigTab.svelte` (config diff folds into Compare).

### Empty/edge states
- No runs yet → empty state showing the command to trigger a run (`POST /api/eval/runs` or the evals CLI).
- Retrieval metrics for generation tier → "—", never 0.
- `error_count > 0` prominently flagged on scorecard and run lists.

## Part 3 — Styling: dense terminal look

- **Load the `dataviz` skill first** (trip-up #14).
- Scope analytics-only classes in `src/app.css` (Tailwind v4 CSS-first — trip-up #10); don't disturb chat/docs pages.
- Typography: `font-mono` + `tabular-nums` for all numerics/tables; 10–11px uppercase tracking-wide labels; DaisyUI `table table-xs` with tightened padding.
- Color: DaisyUI semantic tokens only so nord/dim both work — muted grays via `text-base-content/50..70`, panels `bg-base-200` with 1px `border-base-content/10`, minimal rounding.
- **Thresholds util** `src/lib/utils/thresholds.ts`: one map `{metricName → {good, warn, direction}}`; default quality bands ≥0.80 good / 0.60–0.80 warn / <0.60 bad; direction inverted for `abstention_false_*`, latency, cost (trip-up #13). Returns `text-success`/`text-warning`/`text-error`.
- **Theme-derived charts**: rewrite `ChartAction.ts` color handling per trip-up #11, register line-chart components per trip-up #12. Prefer Chart.js line charts over `@fnando/sparkline` for one consistent system (then drop the sparkline dep and `Sparkline.svelte`).

## Files touched (summary)

**Backend (modify):** `services/evals/api/schemas.py`, `api/dashboard.py`, `api/job_manager.py`, `api/routes.py`; add tests in `services/evals/tests/`.
**rag_server (cleanup):** remove stale legacy-endpoint expectations from `services/rag_server/tests/test_metrics_api.py`.
**Frontend (add):** `src/lib/api/evals.ts`, `src/lib/utils/thresholds.ts`, `analytics/ScorecardTab.svelte`, `analytics/TrendsTab.svelte`, small shared bits (`MetricValue.svelte`, panel wrapper) as needed.
**Frontend (rewrite/modify):** `routes/analytics/+page.svelte`, `analytics/AnalyticsTabs.svelte`, `analytics/ComparisonTab.svelte`, `analytics/SystemHealthTab.svelte`, `charts/ChartAction.ts`, `charts/MetricsBarChart.svelte`, `utils/export.ts`, `src/app.css`.
**Delete:** `analytics/HistoryTab.svelte`, `analytics/ConfigTab.svelte`, `BaselineIndicator.svelte`, `RecommendationPanel.svelte`, `RunComparison.svelte`, `Sparkline.svelte` (if migrating to Chart.js), legacy api.ts eval types/functions.

## Verification

1. **Backend**: `cd services/evals && uv run pytest tests/ -v` with the new fixtures (including a legacy-format run file — trip-up #8).
2. **Local API**: seed 2–3 synthetic run JSONs (new scorecard format, at least one `generation`-tier without retrieval metrics) into `data/eval_runs/` relative to the service cwd; `uv run uvicorn api.app:app --port 8002`; curl `/eval/runs`, `/eval/runs/compare?ids=…`, `/eval/dashboard` to confirm the new fields and null-safe deltas.
3. **Webapp**: `cd services/webapp && npm run dev` with `EVALS_SERVICE_URL=http://localhost:8002`; walk all four tabs in **both** themes (nord/dim via ThemeToggle): scorecard groups + telemetry, trends across seeded runs, compare deltas + ConfigDiff + direction-correct coloring, empty state with no runs, generation-tier "—" cells, active-job strip during a triggered run.
4. **Static checks**: `npm run build` and svelte-check (`npm run check`) pass — this is the safety net proving all legacy references are gone (trip-up #17; no frontend test suite exists).
