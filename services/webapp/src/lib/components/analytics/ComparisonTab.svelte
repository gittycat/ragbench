<script lang="ts">
	import { onMount } from 'svelte';
	import {
		fetchEvalRuns,
		compareEvalRuns,
		type EvalRunSummary,
		type EvalRunDetail,
		type EvalCompareResponse
	} from '$lib/api/evals';
	import RunSelector from '$lib/components/RunSelector.svelte';
	import ExportButton from '$lib/components/ExportButton.svelte';
	import ConfigDiff from '$lib/components/ConfigDiff.svelte';
	import MetricValue from './MetricValue.svelte';
	import InfoTip from './InfoTip.svelte';
	import { deltaColorClass } from '$lib/utils/thresholds';
	import { metricDescription, formatDelta, panelDescription } from '$lib/utils/metricInfo';

	interface Props {
		onRefresh?: () => void;
	}

	let { onRefresh }: Props = $props();

	let runs = $state<EvalRunSummary[]>([]);
	let selectedRunIds = $state<string[]>([]);
	let compareResult = $state<EvalCompareResponse | null>(null);
	let isLoading = $state(true);
	let isComparing = $state(false);
	let error = $state<string | null>(null);

	const RUN_LETTERS = ['A', 'B', 'C', 'D'];
	const GROUP_ORDER = ['retrieval', 'generation', 'citation', 'abstention'];
	const GROUP_LABELS: Record<string, string> = {
		retrieval: 'Retrieval',
		generation: 'Generation',
		citation: 'Citation',
		abstention: 'Abstention'
	};

	let selectedSummaries = $derived(runs.filter((r) => selectedRunIds.includes(r.id)));

	// Baseline (A) is always the oldest selected run, regardless of click order,
	// so Δ consistently reads "newer minus baseline".
	let orderedSelectedIds = $derived.by(() => {
		const createdAt = new Map(runs.map((r) => [r.id, new Date(r.created_at).getTime()]));
		return [...selectedRunIds].sort((a, b) => (createdAt.get(a) ?? 0) - (createdAt.get(b) ?? 0));
	});

	let allGroupedMetrics = $derived.by(() => {
		if (!compareResult) return [];
		const names = new Set<string>();
		const groupOf: Record<string, string> = {};
		for (const run of compareResult.runs) {
			for (const [group, metricNames] of Object.entries(run.scorecard?.by_group ?? {})) {
				if (group === 'performance') continue;
				for (const n of metricNames) {
					names.add(n);
					groupOf[n] = group;
				}
			}
		}
		const rank = (g: string) => {
			const i = GROUP_ORDER.indexOf(g);
			return i === -1 ? GROUP_ORDER.length : i;
		};
		return Array.from(names)
			.sort((a, b) => rank(groupOf[a]) - rank(groupOf[b]) || a.localeCompare(b))
			.map((name) => ({ name, group: groupOf[name] }));
	});

	// Headline quality/cost/speed per run, with deltas vs the baseline run (A)
	interface RunHeadline {
		id: string;
		letter: string;
		name: string;
		model: string | null;
		score: number | null;
		cost: number | null;
		p95: number | null;
		scoreDelta: number | null;
		costDelta: number | null;
		p95Delta: number | null;
	}

	let headlines = $derived.by((): RunHeadline[] => {
		if (!compareResult) return [];
		const base = compareResult.runs[0];
		const baseScore = base?.weighted_score?.score ?? null;
		const baseCost = base?.dashboard_metrics?.avg_cost_usd ?? null;
		const baseP95 = base?.dashboard_metrics?.latency_p95_seconds ?? null;
		return compareResult.runs.map((run, i) => {
			const score = run.weighted_score?.score ?? null;
			const cost = run.dashboard_metrics?.avg_cost_usd ?? null;
			const p95 = run.dashboard_metrics?.latency_p95_seconds ?? null;
			const delta = (v: number | null, b: number | null) =>
				i > 0 && v !== null && b !== null ? v - b : null;
			return {
				id: run.id,
				letter: RUN_LETTERS[i] ?? String(i + 1),
				name: run.name,
				model: run.config?.llm_model ?? null,
				score,
				cost,
				p95,
				scoreDelta: delta(score, baseScore),
				costDelta: delta(cost, baseCost),
				p95Delta: delta(p95, baseP95)
			};
		});
	});

	type TelemetryKey =
		| 'avg_cost_usd'
		| 'latency_p95_seconds'
		| 'total_prompt_tokens'
		| 'total_completion_tokens';

	// Token deltas stay uncolored: more/fewer tokens isn't better or worse per se.
	const TELEMETRY_ROWS: {
		key: TelemetryKey;
		label: string;
		format: 'usd' | 'seconds' | 'int';
		colorDelta: boolean;
	}[] = [
		{ key: 'avg_cost_usd', label: 'cost / query', format: 'usd', colorDelta: true },
		{ key: 'latency_p95_seconds', label: 'latency p95', format: 'seconds', colorDelta: true },
		{ key: 'total_prompt_tokens', label: 'prompt tokens', format: 'int', colorDelta: false },
		{ key: 'total_completion_tokens', label: 'completion tokens', format: 'int', colorDelta: false }
	];

	function telemetryValue(run: EvalRunDetail, key: TelemetryKey): number | null {
		return run.dashboard_metrics?.[key] ?? null;
	}

	function telemetryDelta(key: TelemetryKey): number | null {
		if (!compareResult || compareResult.runs.length !== 2) return null;
		const a = telemetryValue(compareResult.runs[0], key);
		const b = telemetryValue(compareResult.runs[1], key);
		return a !== null && b !== null ? b - a : null;
	}

	function metricValue(runId: string, metric: string): number | null {
		const run = compareResult?.runs.find((r) => r.id === runId);
		const m = run?.scorecard?.metrics.find((mm) => mm.name === metric);
		return m?.value ?? null;
	}

	onMount(() => {
		loadRuns();
	});

	async function loadRuns() {
		isLoading = true;
		error = null;
		try {
			const res = await fetchEvalRuns(50);
			runs = res.runs;
			if (selectedRunIds.length === 0 && runs.length >= 2) {
				selectedRunIds = runs.slice(0, 2).map((r) => r.id);
			}
			if (selectedRunIds.length >= 2) {
				await loadComparison();
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load data';
		} finally {
			isLoading = false;
		}
	}

	async function loadComparison() {
		if (selectedRunIds.length < 2) {
			compareResult = null;
			return;
		}
		isComparing = true;
		try {
			compareResult = await compareEvalRuns(orderedSelectedIds);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to compare runs';
		} finally {
			isComparing = false;
		}
	}

	function handleSelectionChange(ids: string[]) {
		selectedRunIds = ids;
		loadComparison();
	}

	function formatMetricName(metric: string): string {
		return metric.replace(/_/g, ' ');
	}
</script>

<div class="flex flex-col gap-3">
	{#if isLoading}
		<div class="flex items-center justify-center h-64">
			<span class="loading loading-spinner loading-lg"></span>
		</div>
	{:else if error}
		<div class="alert alert-error">
			<span>{error}</span>
			<button class="btn btn-sm" onclick={loadRuns}>Retry</button>
		</div>
	{:else if runs.length === 0}
		<div class="text-center py-8 text-base-content/50">
			No evaluation runs found. Run an evaluation to compare configurations.
		</div>
	{:else}
		<!-- Controls Row -->
		<div class="flex items-center gap-3 flex-wrap">
			<span class="text-xs text-base-content/50">
				Baseline (A) is the oldest selected run; Δ = newer run − A.
			</span>
			<div class="flex-1 min-w-0"></div>
			<ExportButton runs={selectedSummaries} compare={compareResult ?? undefined} disabled={selectedSummaries.length === 0} />
			<button class="btn btn-sm btn-ghost" onclick={loadRuns}>
				<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
				</svg>
				Refresh
			</button>
		</div>

		<!-- Run selection + headline trade-off summary -->
		<div class="grid grid-cols-1 lg:grid-cols-4 gap-3">
			<div class="lg:col-span-1">
				<RunSelector runs={runs} selected={selectedRunIds} onSelectionChange={handleSelectionChange} maxSelection={4} />
			</div>

			<div class="lg:col-span-3">
				<div class="term-panel p-3 h-full">
					<div class="term-label mb-2 flex items-center gap-1">
						Quality / Cost / Speed
						<InfoTip text={panelDescription('compare_headline')} />
					</div>
					{#if isComparing}
						<div class="flex items-center justify-center h-32">
							<span class="loading loading-spinner loading-md"></span>
						</div>
					{:else if headlines.length > 0}
						<div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
							{#each headlines as h}
								<div class="term-tile flex flex-col gap-1 text-xs">
									<div class="flex items-center gap-1.5 min-w-0">
										<span class="badge badge-ghost badge-sm shrink-0">{h.letter}</span>
										<span class="font-mono truncate" title={h.name}>{h.name}</span>
									</div>
									<div class="font-mono text-base-content/60 truncate" title={h.model ?? undefined}>
										{h.model ?? 'model unknown'}
									</div>
									<div class="flex items-baseline justify-between mt-1">
										<span class="text-base-content/50 inline-flex items-center gap-1">
											score
											<InfoTip text={metricDescription('weighted_score') ?? ''} />
										</span>
										<span class="font-mono tabular-nums text-lg">
											{h.score !== null ? (h.score * 100).toFixed(1) + '%' : '—'}
										</span>
									</div>
									{#if h.scoreDelta !== null}
										<div class="text-right font-mono tabular-nums {deltaColorClass('weighted_score', h.scoreDelta)}">
											{formatDelta(h.scoreDelta, 'pts')} vs A
										</div>
									{/if}
									<div class="flex items-baseline justify-between">
										<span class="text-base-content/50 inline-flex items-center gap-1">
											cost / query
											<InfoTip text={metricDescription('avg_cost_usd') ?? ''} />
										</span>
										<span class="font-mono tabular-nums">
											{h.cost !== null ? '$' + h.cost.toFixed(4) : '—'}
										</span>
									</div>
									{#if h.costDelta !== null}
										<div class="text-right font-mono tabular-nums {deltaColorClass('avg_cost_usd', h.costDelta)}">
											{formatDelta(h.costDelta, 'usd')} vs A
										</div>
									{/if}
									<div class="flex items-baseline justify-between">
										<span class="text-base-content/50 inline-flex items-center gap-1">
											latency p95
											<InfoTip text={metricDescription('latency_p95_ms') ?? ''} />
										</span>
										<span class="font-mono tabular-nums">
											{h.p95 !== null ? h.p95.toFixed(2) + 's' : '—'}
										</span>
									</div>
									{#if h.p95Delta !== null}
										<div class="text-right font-mono tabular-nums {deltaColorClass('latency_p95_seconds', h.p95Delta)}">
											{formatDelta(h.p95Delta, 'seconds')} vs A
										</div>
									{/if}
								</div>
							{/each}
						</div>
					{:else}
						<div class="flex items-center justify-center h-32 text-base-content/50 text-sm">
							Select at least two runs to compare.
						</div>
					{/if}
				</div>
			</div>
		</div>

		{#if isComparing}
			<div class="flex items-center justify-center h-32">
				<span class="loading loading-spinner loading-md"></span>
			</div>
		{:else if compareResult}
			<!-- Quality metric comparison -->
			<div class="term-panel overflow-x-auto">
				<div class="term-label mb-2 flex items-center gap-1">
					Quality Metrics
					<InfoTip text={panelDescription('compare_quality')} />
				</div>
				<table class="table table-xs term-table">
					<thead>
						<tr>
							<th>Metric</th>
							{#each compareResult.runs as run, i}
								<th class="text-right font-mono">
									<span class="badge badge-ghost badge-xs mr-1">{RUN_LETTERS[i] ?? i + 1}</span>{run.name}
									{#if run.config?.llm_model}
										<div class="font-normal text-base-content/40 normal-case">{run.config.llm_model}</div>
									{/if}
								</th>
							{/each}
							{#if compareResult.runs.length === 2}
								<th class="text-right">
									<span class="inline-flex items-center gap-1">
										Δ (B−A)
										<InfoTip text={panelDescription('compare_delta')} />
									</span>
								</th>
							{/if}
						</tr>
					</thead>
					<tbody>
						<tr class="font-semibold">
							<td>
								<span class="inline-flex items-center gap-1">
									weighted score
									<InfoTip text={metricDescription('weighted_score') ?? ''} />
								</span>
							</td>
							{#each compareResult.runs as run}
								<td class="text-right">
									<MetricValue metricName="weighted_score" value={run.weighted_score?.score ?? null} colored={false} />
								</td>
							{/each}
							{#if compareResult.runs.length === 2}
								{@const d = compareResult.deltas['weighted_score']}
								<td class="text-right term-num {deltaColorClass('weighted_score', d)}">
									{d !== undefined && d !== null ? formatDelta(d, 'pts') : '—'}
								</td>
							{/if}
						</tr>
						{#each allGroupedMetrics as { name, group }, i}
							{@const desc = metricDescription(name)}
							{#if i === 0 || allGroupedMetrics[i - 1].group !== group}
								<tr>
									<td
										colspan={compareResult.runs.length + (compareResult.runs.length === 2 ? 2 : 1)}
										class="term-label pt-2"
									>
										{GROUP_LABELS[group] ?? group}
									</td>
								</tr>
							{/if}
							<tr class="hover">
								<td class="capitalize">
									<span class="inline-flex items-center gap-1">
										{formatMetricName(name)}
										{#if desc}
											<InfoTip text={desc} />
										{/if}
									</span>
								</td>
								{#each compareResult.runs as run}
									<td class="text-right">
										<MetricValue metricName={name} value={metricValue(run.id, name)} colored={false} />
									</td>
								{/each}
								{#if compareResult.runs.length === 2}
									{@const d = compareResult.deltas[name]}
									<td class="text-right term-num {deltaColorClass(name, d)}">
										{d !== undefined && d !== null ? formatDelta(d, 'pts') : '—'}
									</td>
								{/if}
							</tr>
						{/each}
					</tbody>
				</table>
			</div>

			<!-- Cost & speed comparison -->
			<div class="term-panel overflow-x-auto">
				<div class="term-label mb-2 flex items-center gap-1">
					Cost &amp; Speed
					<InfoTip text={panelDescription('compare_cost_speed')} />
				</div>
				<table class="table table-xs term-table">
					<thead>
						<tr>
							<th>Metric</th>
							{#each compareResult.runs as run, i}
								<th class="text-right font-mono">
									<span class="badge badge-ghost badge-xs mr-1">{RUN_LETTERS[i] ?? i + 1}</span>{run.name}
								</th>
							{/each}
							{#if compareResult.runs.length === 2}
								<th class="text-right">
									<span class="inline-flex items-center gap-1">
										Δ (B−A)
										<InfoTip text={panelDescription('compare_delta')} />
									</span>
								</th>
							{/if}
						</tr>
					</thead>
					<tbody>
						{#each TELEMETRY_ROWS as row}
							{@const desc = metricDescription(row.key)}
							<tr class="hover">
								<td>
									<span class="inline-flex items-center gap-1">
										{row.label}
										{#if desc}
											<InfoTip text={desc} />
										{/if}
									</span>
								</td>
								{#each compareResult.runs as run}
									<td class="text-right">
										<MetricValue metricName={row.key} value={telemetryValue(run, row.key)} format={row.format} colored={false} />
									</td>
								{/each}
								{#if compareResult.runs.length === 2}
									{@const d = telemetryDelta(row.key)}
									<td class="text-right term-num {row.colorDelta ? deltaColorClass(row.key, d) : ''}">
										{d !== null ? formatDelta(d, row.format) : '—'}
									</td>
								{/if}
							</tr>
						{/each}
						<tr class="hover">
							<td>
								<span class="inline-flex items-center gap-1">
									run duration
									<InfoTip text={metricDescription('duration_seconds') ?? ''} />
								</span>
							</td>
							{#each compareResult.runs as run}
								<td class="text-right term-num">{run.duration_seconds?.toFixed(1) ?? '—'}s</td>
							{/each}
							{#if compareResult.runs.length === 2}
								{@const d = compareResult.deltas['duration_seconds']}
								<td class="text-right term-num">
									{d !== undefined && d !== null ? formatDelta(d, 'seconds', 1) : '—'}
								</td>
							{/if}
						</tr>
					</tbody>
				</table>
			</div>

			<!-- Config diff between baseline (A) and B -->
			{#if compareResult.runs.length >= 2}
				<ConfigDiff
					configA={compareResult.runs[0].config}
					configB={compareResult.runs[1].config}
					labelA={`A · ${compareResult.runs[0].name}`}
					labelB={`B · ${compareResult.runs[1].name}`}
					showUnchanged={false}
				/>
			{/if}
		{/if}
	{/if}
</div>
