<script lang="ts">
	import { onMount } from 'svelte';
	import { browser } from '$app/environment';
	import { fetchEvalRuns, type EvalRunSummary } from '$lib/api/evals';
	import {
		chartAction,
		themeColor,
		denseScale,
		denseTooltip,
		CHART_FONT_FAMILY
	} from '$lib/components/charts/ChartAction';
	import { themeSignal } from '$lib/components/charts/theme.svelte';
	import type { ChartConfiguration } from 'chart.js';
	import InfoTip from './InfoTip.svelte';
	import { panelDescription } from '$lib/utils/metricInfo';

	let runs = $state<EvalRunSummary[]>([]);
	let isLoading = $state(true);
	let error = $state<string | null>(null);
	let tierFilter = $state<'all' | 'generation' | 'end_to_end'>('all');
	let mounted = $state(false);

	onMount(() => {
		mounted = true;
		loadData();
	});

	async function loadData() {
		isLoading = true;
		error = null;
		try {
			const res = await fetchEvalRuns(50);
			runs = [...res.runs].sort(
				(a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
			);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load runs';
		} finally {
			isLoading = false;
		}
	}

	let filteredRuns = $derived.by(() =>
		tierFilter === 'all' ? runs : runs.filter((r) => r.tier === tierFilter)
	);

	function shortLabel(run: EvalRunSummary): string {
		return new Date(run.created_at).toLocaleString('en-US', {
			month: 'short',
			day: 'numeric',
			hour: '2-digit',
			minute: '2-digit',
			hour12: false
		});
	}

	interface Series {
		title: string;
		values: (number | null)[];
	}

	function buildConfig(series: Series, unit: string): ChartConfiguration<'line'> {
		// Read the theme signal so this config (and the chart) rebuilds on theme change
		themeSignal();
		const line = themeColor('primary');
		return {
			type: 'line',
			data: {
				labels: filteredRuns.map(shortLabel),
				datasets: [
					{
						label: series.title,
						data: series.values,
						borderColor: line,
						backgroundColor: themeColor('primary', 0.08),
						fill: 'origin',
						pointBackgroundColor: line,
						pointBorderColor: line,
						borderWidth: 2,
						pointRadius: 2,
						pointHoverRadius: 4,
						pointHitRadius: 8,
						spanGaps: true,
						tension: 0.15
					}
				]
			},
			options: {
				responsive: true,
				maintainAspectRatio: false,
				scales: {
					y: denseScale({
						title: {
							display: true,
							text: unit,
							color: themeColor('base-content', 0.45),
							font: { family: CHART_FONT_FAMILY, size: 10 }
						}
					}),
					x: {
						...denseScale(),
						ticks: {
							color: themeColor('base-content', 0.45),
							font: { family: CHART_FONT_FAMILY, size: 9 },
							maxRotation: 45,
							minRotation: 0
						}
					}
				},
				plugins: {
					legend: { display: false },
					tooltip: {
						...denseTooltip(),
						callbacks: {
							label: (ctx) => {
								const v = ctx.raw;
								return typeof v === 'number' ? `${series.title}: ${v.toFixed(3)}` : '';
							}
						}
					}
				}
			}
		};
	}

	let weightedScoreSeries = $derived.by(
		(): Series => ({
			title: 'Weighted Score',
			values: filteredRuns.map((r) => r.weighted_score)
		})
	);

	let faithfulnessSeries = $derived.by(
		(): Series => ({
			title: 'Faithfulness',
			values: filteredRuns.map((r) => r.metrics['faithfulness'] ?? null)
		})
	);

	let answerCorrectnessSeries = $derived.by(
		(): Series => ({
			title: 'Answer Correctness',
			values: filteredRuns.map((r) => r.metrics['answer_correctness'] ?? null)
		})
	);

	let latencyP95Series = $derived.by(
		(): Series => ({
			title: 'Latency p95 (s)',
			values: filteredRuns.map((r) =>
				r.metrics['latency_p95_ms'] != null ? r.metrics['latency_p95_ms'] / 1000 : null
			)
		})
	);

	let costSeries = $derived.by(
		(): Series => ({
			title: 'Cost / Query (USD)',
			values: filteredRuns.map((r) => r.dashboard_metrics?.avg_cost_usd ?? null)
		})
	);
</script>

<div class="flex flex-col gap-3">
	{#if isLoading}
		<div class="flex items-center justify-center h-64">
			<span class="loading loading-spinner loading-lg"></span>
		</div>
	{:else if error}
		<div class="alert alert-error">
			<span>{error}</span>
			<button class="btn btn-sm" onclick={loadData}>Retry</button>
		</div>
	{:else if runs.length === 0}
		<div class="text-center py-8 text-base-content/50">
			No eval runs yet. Trigger a run to see trends here.
		</div>
	{:else}
		<div class="flex items-center gap-2">
			<label class="text-xs text-base-content/50" for="tier-filter">Tier</label>
			<select
				id="tier-filter"
				class="select select-sm font-mono text-xs"
				bind:value={tierFilter}
			>
				<option value="all">All</option>
				<option value="generation">Generation</option>
				<option value="end_to_end">End-to-end</option>
			</select>
			<span class="text-xs text-base-content/50">{filteredRuns.length} runs</span>
		</div>

		{#if filteredRuns.length === 0}
			<div class="text-center py-8 text-base-content/50">No runs for this tier.</div>
		{:else if browser && mounted}
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-3">
				<div class="term-panel">
					<div class="term-label mb-2 flex items-center gap-1">
						Weighted Score
						<InfoTip text={panelDescription('trend_weighted_score')} />
					</div>
					<div style="height: 220px;">
						<canvas use:chartAction={buildConfig(weightedScoreSeries, 'score')}></canvas>
					</div>
				</div>
				<div class="term-panel">
					<div class="term-label mb-2 flex items-center gap-1">
						Faithfulness
						<InfoTip text={panelDescription('trend_faithfulness')} />
					</div>
					<div style="height: 220px;">
						<canvas use:chartAction={buildConfig(faithfulnessSeries, 'score')}></canvas>
					</div>
				</div>
				<div class="term-panel">
					<div class="term-label mb-2 flex items-center gap-1">
						Answer Correctness
						<InfoTip text={panelDescription('trend_answer_correctness')} />
					</div>
					<div style="height: 220px;">
						<canvas use:chartAction={buildConfig(answerCorrectnessSeries, 'score')}></canvas>
					</div>
				</div>
				<div class="term-panel">
					<div class="term-label mb-2 flex items-center gap-1">
						Latency p95
						<InfoTip text={panelDescription('trend_latency_p95')} />
					</div>
					<div style="height: 220px;">
						<canvas use:chartAction={buildConfig(latencyP95Series, 's')}></canvas>
					</div>
				</div>
				<div class="term-panel lg:col-span-2">
					<div class="term-label mb-2 flex items-center gap-1">
						Cost / Query
						<InfoTip text={panelDescription('trend_cost')} />
					</div>
					<div style="height: 220px;">
						<canvas use:chartAction={buildConfig(costSeries, 'usd')}></canvas>
					</div>
				</div>
			</div>
		{/if}
	{/if}
</div>
