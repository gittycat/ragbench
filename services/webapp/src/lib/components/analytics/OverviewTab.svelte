<script lang="ts">
	import { untrack } from 'svelte';
	import type { SystemMetrics } from '$lib/api';
	import { fetchEvalRuns, fetchEvalRun, type EvalRunDetail, type ScorecardMetric } from '$lib/api/evals';
	import { STAGE_INFO, type StageId, metricDescription, metricLabel } from '$lib/utils/metricInfo';
	import {
		computeBracketHealth,
		computeWeakestLink,
		weakestLinkVerdict,
		bracketForGroup,
		type Bracket,
		type Health
	} from '$lib/utils/stageHealth';
	import MetricValue from './MetricValue.svelte';
	import PipelineDiagram from './PipelineDiagram.svelte';
	import InfoTip from './InfoTip.svelte';

	interface Props {
		metrics: SystemMetrics;
		/** Bumped by the parent on each auto/manual refresh so eval data reloads too. */
		refreshTick?: number;
	}

	let { metrics, refreshTick = 0 }: Props = $props();

	let detail = $state<EvalRunDetail | null>(null);
	let hasRuns = $state<boolean | null>(null); // null = unknown yet
	let isLoading = $state(true);
	let error = $state<string | null>(null);
	let selectedStage = $state<StageId | null>(null);
	let userPickedStage = $state(false);

	// Which pipeline stages belong to which measured bracket, for the stage detail panel.
	const STAGE_BRACKET: Record<StageId, Bracket> = {
		documents: 'retrieval',
		chunking: 'retrieval',
		embedding: 'retrieval',
		hybrid_search: 'retrieval',
		rerank: 'retrieval',
		llm: 'generation'
	};

	// Initial load + reload whenever the parent's refresh cycle ticks.
	// untrack: load() reads/writes its own state, which must not become a dependency.
	$effect(() => {
		void refreshTick;
		untrack(() => load());
	});

	async function load() {
		// Only show the spinner on first load, not on background refreshes.
		if (hasRuns === null) isLoading = true;
		error = null;
		try {
			const res = await fetchEvalRuns(1);
			if (res.runs.length === 0) {
				hasRuns = false;
				detail = null;
			} else {
				hasRuns = true;
				detail = await fetchEvalRun(res.runs[0].id);
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load overview data';
		} finally {
			isLoading = false;
		}
	}

	// Computed once here and shared with the diagram/verdict (avoids recomputing per consumer).
	let bracketHealth = $derived(computeBracketHealth(detail));
	let weakest = $derived(computeWeakestLink(bracketHealth));
	let verdict = $derived(weakestLinkVerdict(detail, bracketHealth, weakest));

	let bannerBorderClass = $derived.by(() => {
		if (!detail?.scorecard?.metrics?.length) return 'border-l-base-content/20';
		if (!weakest) return 'border-l-success';
		return weakest.band === 'bad' ? 'border-l-error' : 'border-l-warning';
	});

	// Default selection once data loads: the weakest bracket's representative
	// node, or the retrieval hub when nothing is weak.
	$effect(() => {
		if (userPickedStage || selectedStage !== null) return;
		if (!detail) return;
		selectedStage = weakest?.bracket === 'generation' ? 'llm' : 'hybrid_search';
	});

	function handleSelectStage(stage: StageId) {
		userPickedStage = true;
		selectedStage = stage;
	}

	let selectedStageMetrics = $derived.by((): ScorecardMetric[] => {
		if (!selectedStage || !detail?.scorecard) return [];
		const bracket = STAGE_BRACKET[selectedStage];
		return detail.scorecard.metrics.filter((m) => bracketForGroup(m.group) === bracket);
	});

	let selectedBracketHealth = $derived.by((): Health => {
		if (!selectedStage) return 'unknown';
		return bracketHealth[STAGE_BRACKET[selectedStage]].health;
	});

	interface ConfigRow {
		label: string;
		value: string;
	}

	let selectedStageConfig = $derived.by((): ConfigRow[] => {
		if (!selectedStage) return [];
		switch (selectedStage) {
			case 'documents':
				return [
					{ label: 'Documents', value: String(metrics.document_count) },
					{ label: 'Chunks', value: metrics.chunk_count.toLocaleString() }
				];
			case 'chunking':
				return [
					{ label: 'Contextual retrieval', value: metrics.retrieval.contextual_retrieval.enabled ? 'on' : 'off' },
					{ label: 'Chunk size', value: String(metrics.retrieval.hybrid_search.vector.chunk_size) },
					{ label: 'Chunk overlap', value: String(metrics.retrieval.hybrid_search.vector.chunk_overlap) }
				];
			case 'embedding':
				return [
					{ label: 'Model', value: metrics.models.embedding?.name ?? '—' },
					{ label: 'Provider', value: metrics.models.embedding?.provider ?? '—' },
					{ label: 'Status', value: metrics.models.embedding?.status ?? '—' }
				];
			case 'hybrid_search':
				return metrics.retrieval.hybrid_search.enabled
					? [
							{ label: 'Fusion', value: metrics.retrieval.hybrid_search.fusion_method },
							{ label: 'RRF k', value: String(metrics.retrieval.hybrid_search.rrf_k) },
							{ label: 'Retrieval top-k', value: String(metrics.retrieval.retrieval_top_k) }
						]
					: [
							{ label: 'Hybrid search', value: 'disabled (vector-only)' },
							{ label: 'Retrieval top-k', value: String(metrics.retrieval.retrieval_top_k) }
						];
			case 'rerank':
				return metrics.retrieval.reranker.enabled
					? [
							{ label: 'Model', value: metrics.retrieval.reranker.model ?? '—' },
							{ label: 'Final top-n', value: String(metrics.retrieval.final_top_n) }
						]
					: [{ label: 'Reranker', value: 'disabled' }];
			case 'llm':
				return [
					{ label: 'Model', value: metrics.models.llm?.name ?? '—' },
					{ label: 'Provider', value: metrics.models.llm?.provider ?? '—' },
					{ label: 'Status', value: metrics.models.llm?.status ?? '—' }
				];
			default:
				return [];
		}
	});
</script>

<div class="flex flex-col gap-3">
	{#if isLoading}
		<div class="flex items-center justify-center h-64">
			<span class="loading loading-spinner loading-lg"></span>
		</div>
	{:else if error}
		<div class="alert alert-error">
			<span>{error}</span>
			<button class="btn btn-sm" onclick={load}>Retry</button>
		</div>
	{:else if hasRuns === false}
		<div class="text-center py-8 text-base-content/50 flex flex-col gap-2">
			<span>No eval runs yet — run one to see pipeline health.</span>
			<span class="font-mono text-xs">POST /api/eval/runs or the evals CLI to trigger one.</span>
		</div>
	{:else}
		<!-- Diagnosis banner -->
		<div class="term-panel border-l-4 {bannerBorderClass} flex flex-col gap-2">
			<p class="text-sm">{verdict}</p>
			{#if detail}
				<div class="flex flex-wrap items-center gap-4 text-xs">
					<div class="flex items-center gap-1.5">
						<span class="text-base-content/50">weighted score</span>
						<MetricValue metricName="weighted_score" value={detail.weighted_score?.score ?? null} />
					</div>
					<div class="flex items-center gap-1.5">
						<span class="text-base-content/50">cost / query</span>
						<MetricValue metricName="avg_cost_usd" value={detail.dashboard_metrics?.avg_cost_usd} format="usd" colored={false} />
					</div>
					<div class="flex items-center gap-1.5">
						<span class="text-base-content/50">latency p95</span>
						<MetricValue
							metricName="latency_p95_seconds"
							value={detail.dashboard_metrics?.latency_p95_seconds}
							format="seconds"
							colored={false}
						/>
					</div>
				</div>
			{/if}
		</div>

		<!-- Pipeline diagram -->
		<PipelineDiagram {metrics} {bracketHealth} {selectedStage} onSelectStage={handleSelectStage} />

		<!-- Stage detail panel -->
		{#if selectedStage}
			{@const info = STAGE_INFO[selectedStage]}
			<div class="term-panel flex flex-col gap-2">
				<div class="flex items-center justify-between">
					<span class="term-label">{info.title}</span>
					<span class="flex items-center gap-1 text-[10px] font-mono uppercase tracking-wider">
						<span
							class="h-1.5 w-1.5 rounded-full {selectedBracketHealth === 'good'
								? 'bg-success'
								: selectedBracketHealth === 'warn'
									? 'bg-warning'
									: selectedBracketHealth === 'bad'
										? 'bg-error'
										: 'bg-base-content/25'}"
						></span>
						<span class="text-base-content/50">{selectedBracketHealth}</span>
					</span>
				</div>
				<p class="text-xs text-base-content/70">{info.description}</p>
				<p class="text-xs text-base-content/50">{info.whatItDoes}</p>

				<!-- Config values -->
				<div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs font-mono">
					{#each selectedStageConfig as row}
						<div><span class="text-base-content/50">{row.label}:</span> {row.value}</div>
					{/each}
				</div>

				<!-- Bracket metrics -->
				{#if selectedStageMetrics.length > 0}
					<table class="table table-xs term-table">
						<tbody>
							{#each selectedStageMetrics as m}
								{@const desc = metricDescription(m.name)}
								<tr class="hover">
									<td class="capitalize text-xs">
										<span class="inline-flex items-center gap-1">
											{metricLabel(m.name)}
											{#if desc}
												<InfoTip text={desc} />
											{/if}
										</span>
									</td>
									<td class="text-right">
										<MetricValue metricName={m.name} value={m.value} />
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				{:else}
					<div class="text-center py-2 text-base-content/40 text-xs">no eval data for this stage yet</div>
				{/if}

				<div class="text-xs bg-base-100 border border-base-content/10 rounded-sm p-2">
					<span class="font-semibold text-base-content/70">If this is weak, try: </span>
					<span class="text-base-content/60">{info.ifWeak}</span>
				</div>
			</div>
		{/if}
	{/if}
</div>
