<script lang="ts">
	import { onMount } from 'svelte';
	import { fetchEvalRuns, compareEvalRuns, type EvalRunSummary, type EvalCompareResponse } from '$lib/api/evals';
	import MetricsBarChart from '$lib/components/charts/MetricsBarChart.svelte';
	import RunSelector from '$lib/components/RunSelector.svelte';
	import ExportButton from '$lib/components/ExportButton.svelte';
	import ConfigDiff from '$lib/components/ConfigDiff.svelte';
	import MetricValue from './MetricValue.svelte';
	import { deltaColorClass } from '$lib/utils/thresholds';

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

	let selectedSummaries = $derived(runs.filter((r) => selectedRunIds.includes(r.id)));

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
		return Array.from(names)
			.sort((a, b) => (groupOf[a] || '').localeCompare(groupOf[b] || '') || a.localeCompare(b))
			.map((name) => ({ name, group: groupOf[name] }));
	});

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
			compareResult = await compareEvalRuns(selectedRunIds);
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
			<div class="flex-1 min-w-0"></div>
			<ExportButton runs={selectedSummaries} compare={compareResult ?? undefined} disabled={selectedSummaries.length === 0} />
			<button class="btn btn-sm btn-ghost" onclick={loadRuns}>
				<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
				</svg>
				Refresh
			</button>
		</div>

		<!-- Main Content Grid -->
		<div class="grid grid-cols-1 lg:grid-cols-4 gap-3">
			<div class="lg:col-span-1">
				<RunSelector runs={runs} selected={selectedRunIds} onSelectionChange={handleSelectionChange} maxSelection={4} />
			</div>

			<div class="lg:col-span-3">
				<div class="term-panel p-3">
					<div class="term-label mb-2">
						Metrics Comparison
					</div>
					<MetricsBarChart runs={selectedSummaries} height={300} />
				</div>
			</div>
		</div>

		{#if isComparing}
			<div class="flex items-center justify-center h-32">
				<span class="loading loading-spinner loading-md"></span>
			</div>
		{:else if compareResult}
			<!-- Dense comparison table -->
			<div class="term-panel overflow-x-auto">
				<div class="term-label mb-2">
					Metric Comparison
				</div>
				<table class="table table-xs term-table">
					<thead>
						<tr>
							<th>Metric</th>
							{#each compareResult.runs as run}
								<th class="text-right font-mono">{run.name}</th>
							{/each}
							{#if compareResult.runs.length === 2}
								<th class="text-right">Δ</th>
							{/if}
						</tr>
					</thead>
					<tbody>
						<tr class="font-semibold">
							<td>weighted_score</td>
							{#each compareResult.runs as run}
								<td class="text-right">
									<MetricValue metricName="weighted_score" value={run.weighted_score?.score ?? null} />
								</td>
							{/each}
							{#if compareResult.runs.length === 2}
								<td class="text-right term-num {deltaColorClass('weighted_score', compareResult.deltas['weighted_score'])}">
									{compareResult.deltas['weighted_score'] !== undefined && compareResult.deltas['weighted_score'] !== null
										? (compareResult.deltas['weighted_score'] * 100).toFixed(1) + '%'
										: '—'}
								</td>
							{/if}
						</tr>
						{#each allGroupedMetrics as { name, group }}
							<tr class="hover">
								<td class="capitalize" title={group}>{formatMetricName(name)}</td>
								{#each compareResult.runs as run}
									<td class="text-right">
										<MetricValue metricName={name} value={metricValue(run.id, name)} />
									</td>
								{/each}
								{#if compareResult.runs.length === 2}
									<td class="text-right term-num {deltaColorClass(name, compareResult.deltas[name])}">
										{compareResult.deltas[name] !== undefined && compareResult.deltas[name] !== null
											? (compareResult.deltas[name] * 100).toFixed(1) + '%'
											: '—'}
									</td>
								{/if}
							</tr>
						{/each}
						<tr>
							<td>duration_seconds</td>
							{#each compareResult.runs as run}
								<td class="text-right term-num">{run.duration_seconds?.toFixed(1) ?? '—'}s</td>
							{/each}
							{#if compareResult.runs.length === 2}
								<td class="text-right term-num {deltaColorClass('duration_seconds', compareResult.deltas['duration_seconds'])}">
									{compareResult.deltas['duration_seconds'] !== undefined && compareResult.deltas['duration_seconds'] !== null
										? compareResult.deltas['duration_seconds'] + 's'
										: '—'}
								</td>
							{/if}
						</tr>
					</tbody>
				</table>
			</div>

			<!-- Telemetry comparison -->
			<div class="term-panel overflow-x-auto">
				<div class="term-label mb-2">
					Telemetry
				</div>
				<table class="table table-xs term-table">
					<thead>
						<tr>
							<th>Metric</th>
							{#each compareResult.runs as run}
								<th class="text-right font-mono">{run.name}</th>
							{/each}
						</tr>
					</thead>
					<tbody>
						<tr>
							<td>avg_cost_usd</td>
							{#each compareResult.runs as run}
								<td class="text-right"><MetricValue metricName="avg_cost_usd" value={run.dashboard_metrics?.avg_cost_usd} format="usd" /></td>
							{/each}
						</tr>
						<tr>
							<td>latency_p95_seconds</td>
							{#each compareResult.runs as run}
								<td class="text-right"><MetricValue metricName="latency_p95_seconds" value={run.dashboard_metrics?.latency_p95_seconds} format="seconds" /></td>
							{/each}
						</tr>
						<tr>
							<td>total_prompt_tokens</td>
							{#each compareResult.runs as run}
								<td class="text-right"><MetricValue metricName="total_prompt_tokens" value={run.dashboard_metrics?.total_prompt_tokens} format="int" /></td>
							{/each}
						</tr>
						<tr>
							<td>total_completion_tokens</td>
							{#each compareResult.runs as run}
								<td class="text-right"><MetricValue metricName="total_completion_tokens" value={run.dashboard_metrics?.total_completion_tokens} format="int" /></td>
							{/each}
						</tr>
					</tbody>
				</table>
			</div>

			<!-- Config diff (first two selected runs) -->
			{#if compareResult.runs.length >= 2}
				<ConfigDiff
					configA={compareResult.runs[0].config}
					configB={compareResult.runs[1].config}
					labelA={compareResult.runs[0].name}
					labelB={compareResult.runs[1].name}
					showUnchanged={false}
				/>
			{/if}
		{/if}
	{/if}
</div>
