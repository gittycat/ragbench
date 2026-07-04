<script lang="ts">
	import { onMount } from 'svelte';
	import { browser } from '$app/environment';
	import {
		chartAction,
		getChartColors,
		chartInk,
		denseScale,
		denseTooltip,
		CHART_FONT_FAMILY
	} from './ChartAction';
	import { themeSignal } from './theme.svelte';
	import type { EvalRunSummary } from '$lib/api/evals';
	import type { ChartConfiguration } from 'chart.js';

	interface Props {
		runs: EvalRunSummary[];
		height?: number;
	}

	let { runs, height = 300 }: Props = $props();

	let mounted = $state(false);

	onMount(() => {
		mounted = true;
	});

	// Quality metrics only (exclude the "performance" group — ms/usd scale doesn't fit a 0-100% axis)
	let metrics = $derived.by(() => {
		const allMetrics = new Set<string>();
		for (const run of runs) {
			for (const [group, names] of Object.entries(run.groups)) {
				if (group === 'performance') continue;
				for (const n of names) allMetrics.add(n);
			}
		}
		return Array.from(allMetrics).sort();
	});

	function formatMetricName(metric: string): string {
		return metric.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
	}

	function formatRunLabel(run: EvalRunSummary): string {
		const time = new Date(run.created_at).toLocaleTimeString('en-US', {
			hour: '2-digit',
			minute: '2-digit',
			hour12: false
		});
		return `${run.name} ${time}`;
	}

	let chartConfig = $derived.by((): ChartConfiguration<'bar'> => {
		// Read the theme signal so colors re-resolve when data-theme changes
		themeSignal();
		const labels = metrics.map(formatMetricName);
		const colors = getChartColors(runs.length);
		const { inkMuted } = chartInk();

		const datasets = runs.map((run, i) => ({
			label: formatRunLabel(run),
			data: metrics.map((m) => (run.metrics[m] !== undefined ? run.metrics[m] * 100 : NaN)),
			backgroundColor: colors.background[i],
			borderColor: colors.border[i],
			borderWidth: 1,
			borderRadius: 3,
			maxBarThickness: 22
		}));

		return {
			type: 'bar',
			data: { labels, datasets },
			options: {
				responsive: true,
				maintainAspectRatio: false,
				scales: {
					y: denseScale({
						beginAtZero: true,
						max: 100,
						title: {
							display: true,
							text: 'Score (%)',
							color: inkMuted,
							font: { family: CHART_FONT_FAMILY, size: 10 }
						}
					}),
					x: {
						...denseScale(),
						grid: { display: false },
						ticks: {
							color: inkMuted,
							font: { family: CHART_FONT_FAMILY, size: 10 },
							maxRotation: 45,
							minRotation: 0
						}
					}
				},
				plugins: {
					legend: {
						position: 'bottom',
						labels: {
							boxWidth: 10,
							boxHeight: 10,
							color: inkMuted,
							font: { family: CHART_FONT_FAMILY, size: 10 },
							padding: 8
						}
					},
					tooltip: {
						...denseTooltip(),
						displayColors: true,
						callbacks: {
							label: (ctx) => {
								const value = typeof ctx.raw === 'number' ? ctx.raw.toFixed(1) : '—';
								return `${ctx.dataset.label}: ${value}%`;
							}
						}
					}
				}
			}
		};
	});
</script>

{#if browser && mounted && runs.length > 0}
	<div class="w-full" style="height: {height}px;">
		<canvas use:chartAction={chartConfig}></canvas>
	</div>
{:else if runs.length === 0}
	<div class="flex items-center justify-center h-48 text-base-content/50 text-sm">
		Select runs to compare
	</div>
{:else}
	<div class="flex items-center justify-center h-48">
		<span class="loading loading-spinner loading-md"></span>
	</div>
{/if}
