<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { fetchSystemMetrics, type SystemMetrics } from '$lib/api';
	import { fetchEvalDashboard, fetchActiveEvalJob, type EvalDashboardResponse, type ActiveEvalJob } from '$lib/api/evals';
	import AnalyticsTabs from '$lib/components/analytics/AnalyticsTabs.svelte';
	import ScorecardTab from '$lib/components/analytics/ScorecardTab.svelte';
	import TrendsTab from '$lib/components/analytics/TrendsTab.svelte';
	import ComparisonTab from '$lib/components/analytics/ComparisonTab.svelte';
	import SystemHealthTab from '$lib/components/analytics/SystemHealthTab.svelte';

	let metrics = $state<SystemMetrics | null>(null);
	let evalDashboard = $state<EvalDashboardResponse | null>(null);
	let activeJob = $state<ActiveEvalJob | null>(null);
	let isLoading = $state(true);
	let error = $state<string | null>(null);
	let autoRefresh = $state(true);
	let refreshInterval: number | null = null;
	let activeJobInterval: number | null = null;

	// Tab state - read from URL or default to 'scorecard'
	let activeTab = $state<string>('scorecard');

	const tabs = [
		{ id: 'scorecard', label: 'Scorecard', icon: 'scorecard' },
		{ id: 'trends', label: 'Trends', icon: 'trends' },
		{ id: 'compare', label: 'Compare', icon: 'compare' },
		{ id: 'system', label: 'System', icon: 'system' }
	];

	onMount(() => {
		const urlTab = $page.url.searchParams.get('tab');
		if (urlTab && tabs.some((t) => t.id === urlTab)) {
			activeTab = urlTab;
		}

		loadAll();
		startAutoRefresh();
		startActiveJobPolling();
		return () => {
			stopAutoRefresh();
			stopActiveJobPolling();
		};
	});

	function handleTabChange(tabId: string) {
		activeTab = tabId;
		const url = new URL(window.location.href);
		url.searchParams.set('tab', tabId);
		goto(url.pathname + url.search, { replaceState: true, keepFocus: true });
	}

	function startAutoRefresh() {
		if (autoRefresh && !refreshInterval) {
			refreshInterval = window.setInterval(() => loadAll(), 30000);
		}
	}

	function stopAutoRefresh() {
		if (refreshInterval) {
			clearInterval(refreshInterval);
			refreshInterval = null;
		}
	}

	function toggleAutoRefresh() {
		autoRefresh = !autoRefresh;
		if (autoRefresh) {
			startAutoRefresh();
		} else {
			stopAutoRefresh();
		}
	}

	function startActiveJobPolling() {
		if (!activeJobInterval) {
			activeJobInterval = window.setInterval(pollActiveJob, 5000);
		}
	}

	function stopActiveJobPolling() {
		if (activeJobInterval) {
			clearInterval(activeJobInterval);
			activeJobInterval = null;
		}
	}

	async function pollActiveJob() {
		try {
			activeJob = await fetchActiveEvalJob();
		} catch {
			// Ignore transient polling errors
		}
	}

	async function loadAll() {
		if (!autoRefresh && !isLoading) isLoading = true;
		error = null;
		try {
			const [m, e, a] = await Promise.all([
				fetchSystemMetrics(),
				fetchEvalDashboard().catch(() => null),
				fetchActiveEvalJob().catch(() => null)
			]);
			metrics = m;
			evalDashboard = e;
			activeJob = a;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load data';
		} finally {
			isLoading = false;
		}
	}
</script>

{#if isLoading && !metrics}
	<div class="flex items-center justify-center h-[calc(100vh-200px)]">
		<span class="loading loading-spinner loading-lg"></span>
	</div>
{:else if error && !metrics}
	<div class="flex flex-col items-center justify-center h-[calc(100vh-200px)] gap-4">
		<div class="alert alert-error max-w-md">
			<svg
				xmlns="http://www.w3.org/2000/svg"
				class="h-5 w-5"
				fill="none"
				viewBox="0 0 24 24"
				stroke="currentColor"
			>
				<path
					stroke-linecap="round"
					stroke-linejoin="round"
					stroke-width="2"
					d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
				/>
			</svg>
			<span>{error}</span>
		</div>
		<button class="btn btn-primary btn-sm" onclick={loadAll}>Retry</button>
	</div>
{:else if metrics}
	<div class="flex flex-col gap-3">
		<!-- Header Bar -->
		<div class="bg-base-200 border border-base-content/10 rounded-sm px-3 py-2 flex items-center justify-between text-xs font-mono tabular-nums flex-wrap gap-y-1">
			<div class="flex items-center gap-3 flex-wrap">
				<span class="font-bold text-sm">{metrics.system_name}</span>
				<span class="text-base-content/50">v{metrics.version}</span>
				<div class="divider divider-horizontal mx-0 h-4"></div>
				<span class="flex items-center gap-1">
					<span
						class="h-2 w-2 rounded-full {metrics.health_status === 'healthy'
							? 'bg-success animate-pulse'
							: 'bg-error'}"
					></span>
					<span class="capitalize">{metrics.health_status}</span>
				</span>
				<div class="divider divider-horizontal mx-0 h-4"></div>
				<span><strong>{metrics.document_count}</strong> docs</span>
				<span><strong>{metrics.chunk_count.toLocaleString()}</strong> chunks</span>
				{#if evalDashboard?.latest_run}
					<div class="divider divider-horizontal mx-0 h-4"></div>
					<span class="badge badge-ghost badge-sm">{evalDashboard.latest_run.name}</span>
					<span class="badge badge-ghost badge-sm">{evalDashboard.latest_run.tier}</span>
					{#each evalDashboard.latest_run.datasets as ds}
						<span class="badge badge-ghost badge-sm">{ds}</span>
					{/each}
					{#if evalDashboard.latest_run.error_count > 0}
						<span class="badge badge-error badge-sm">{evalDashboard.latest_run.error_count} errors</span>
					{/if}
				{/if}
				{#if activeJob}
					<div class="divider divider-horizontal mx-0 h-4"></div>
					<span class="flex items-center gap-1.5">
						<span class="loading loading-spinner loading-xs text-warning"></span>
						<span class="text-warning">
							{activeJob.progress.phase}
							{#if activeJob.progress.total_questions > 0}
								({activeJob.progress.current_question}/{activeJob.progress.total_questions})
							{/if}
							— {activeJob.progress.elapsed_seconds.toFixed(0)}s
						</span>
					</span>
				{/if}
			</div>
			<div class="flex items-center gap-1">
				<span class="text-base-content/50"
					>{new Date().toLocaleTimeString()}</span
				>
				<button
					class="btn btn-ghost btn-xs"
					onclick={toggleAutoRefresh}
					title={autoRefresh ? 'Auto-refresh ON (30s)' : 'Auto-refresh OFF'}
				>
					{#if autoRefresh}
						<svg
							xmlns="http://www.w3.org/2000/svg"
							class="h-3.5 w-3.5 text-success"
							fill="none"
							viewBox="0 0 24 24"
							stroke="currentColor"
						>
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								stroke-width="2"
								d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
							/>
						</svg>
					{:else}
						<svg
							xmlns="http://www.w3.org/2000/svg"
							class="h-3.5 w-3.5 text-base-content/50"
							fill="none"
							viewBox="0 0 24 24"
							stroke="currentColor"
						>
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								stroke-width="2"
								d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z"
							/>
						</svg>
					{/if}
				</button>
				<button
					class="btn btn-ghost btn-xs"
					onclick={loadAll}
					disabled={isLoading}
					aria-label="Refresh"
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						class="h-3.5 w-3.5 {isLoading ? 'animate-spin' : ''}"
						fill="none"
						viewBox="0 0 24 24"
						stroke="currentColor"
					>
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							stroke-width="2"
							d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
						/>
					</svg>
				</button>
			</div>
		</div>

		<!-- Services Status (compact) -->
		{#if Object.keys(metrics.component_status).length > 0}
			<div class="flex items-center gap-2 px-1">
				<span class="term-label">Services:</span>
				{#each Object.entries(metrics.component_status) as [name, status]}
					<div class="tooltip tooltip-bottom" data-tip="{name}: {status}">
						<span
							class="flex items-center gap-1 text-xs font-mono bg-base-200 border border-base-content/10 rounded-sm px-1.5 py-0.5"
						>
							<span
								class="h-1.5 w-1.5 rounded-full {status === 'healthy' ||
								status === 'available'
									? 'bg-success'
									: 'bg-error'}"
							></span>
							<span class="capitalize">{name}</span>
						</span>
					</div>
				{/each}
			</div>
		{/if}

		<!-- Tabbed Content -->
		<AnalyticsTabs {activeTab} onTabChange={handleTabChange} {tabs}>
			{#if activeTab === 'scorecard'}
				<ScorecardTab />
			{:else if activeTab === 'trends'}
				<TrendsTab />
			{:else if activeTab === 'compare'}
				<ComparisonTab onRefresh={loadAll} />
			{:else if activeTab === 'system'}
				<SystemHealthTab {metrics} />
			{/if}
		</AnalyticsTabs>
	</div>
{/if}
