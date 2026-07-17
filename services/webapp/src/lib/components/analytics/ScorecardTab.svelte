<script lang="ts">
	import { onMount } from 'svelte';
	import {
		fetchEvalRuns,
		fetchEvalRun,
		type EvalRunSummary,
		type EvalRunDetail,
		type ScorecardMetric
	} from '$lib/api/evals';
	import MetricValue from './MetricValue.svelte';
	import InfoTip from './InfoTip.svelte';
	import { metricDescription, panelDescription, metricLabel } from '$lib/utils/metricInfo';

	let runs = $state<EvalRunSummary[]>([]);
	let selectedRunId = $state<string>('');
	let detail = $state<EvalRunDetail | null>(null);
	let isLoading = $state(true);
	let isLoadingDetail = $state(false);
	let error = $state<string | null>(null);

	const GROUP_ORDER = ['retrieval', 'generation', 'citation', 'abstention'];
	const GROUP_LABELS: Record<string, string> = {
		retrieval: 'Retrieval',
		generation: 'Generation',
		citation: 'Citation',
		abstention: 'Abstention'
	};

	function metricsByGroup(group: string): ScorecardMetric[] {
		if (!detail?.scorecard) return [];
		return detail.scorecard.metrics.filter((m) => m.group === group);
	}

	let performanceMetrics = $derived.by(() => metricsByGroup('performance'));

	function findMetric(name: string): ScorecardMetric | undefined {
		return performanceMetrics.find((m) => m.name === name);
	}

	function latencySeconds(name: string): string {
		const v = detail?.scorecard?.metrics.find((m) => m.name === name)?.value;
		return typeof v === 'number' ? (v / 1000).toFixed(2) : '—';
	}

	let costDetails = $derived.by(() => {
		const m = findMetric('cost_per_query');
		return (m?.details ?? {}) as Record<string, unknown>;
	});

	let latencyDetails = $derived.by(() => {
		const m = findMetric('latency_avg_ms');
		return (m?.details ?? {}) as Record<string, unknown>;
	});

	let sortedContributions = $derived.by(() => {
		const ws = detail?.weighted_score;
		if (!ws) return [];
		return Object.entries(ws.contributions).sort((a, b) => b[1] - a[1]);
	});

	onMount(() => {
		loadRuns();
	});

	async function loadRuns() {
		isLoading = true;
		error = null;
		try {
			const res = await fetchEvalRuns(50);
			runs = res.runs;
			if (runs.length > 0) {
				selectedRunId = runs[0].id;
				await loadDetail(selectedRunId);
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load runs';
		} finally {
			isLoading = false;
		}
	}

	async function loadDetail(runId: string) {
		if (!runId) {
			detail = null;
			return;
		}
		isLoadingDetail = true;
		try {
			detail = await fetchEvalRun(runId);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load run detail';
		} finally {
			isLoadingDetail = false;
		}
	}

	function handleRunChange() {
		loadDetail(selectedRunId);
	}

	function formatDateTime(timestamp: string): string {
		return new Date(timestamp).toLocaleString('en-US', {
			month: 'short',
			day: 'numeric',
			hour: '2-digit',
			minute: '2-digit',
			hour12: false
		});
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
		<div class="text-center py-8 text-base-content/50 flex flex-col gap-2">
			<span>No eval runs yet.</span>
			<span class="font-mono text-xs">POST /api/eval/runs or the evals CLI to trigger one.</span>
		</div>
	{:else}
		<!-- Run picker -->
		<div class="flex items-center gap-2">
			<label class="text-xs text-base-content/50" for="run-picker">Run</label>
			<select
				id="run-picker"
				class="select select-sm font-mono text-xs"
				bind:value={selectedRunId}
				onchange={handleRunChange}
			>
				{#each runs as run}
					<option value={run.id}>{run.name} — {formatDateTime(run.created_at)}</option>
				{/each}
			</select>
			{#if isLoadingDetail}
				<span class="loading loading-spinner loading-xs"></span>
			{/if}
		</div>

		{#if detail}
			<!-- Run meta line -->
			<div class="term-panel px-3 flex items-center gap-3 flex-wrap text-xs font-mono tabular-nums">
				<InfoTip text={panelDescription('run_meta')} />
				<span class="badge badge-ghost badge-sm">{detail.tier}</span>
				{#each detail.datasets as ds}
					<span class="badge badge-ghost badge-sm">{ds}</span>
				{/each}
				<span class="text-base-content/50">samples/dataset: {detail.metadata.samples_per_dataset ?? '—'}</span>
				<span class="text-base-content/50">seed: {detail.metadata.seed ?? '—'}</span>
				<span class="text-base-content/50">questions: {detail.question_count}</span>
				<span class={detail.error_count > 0 ? 'text-error font-semibold' : 'text-base-content/50'}>
					errors: {detail.error_count}
				</span>
				<span class="text-base-content/50">duration: {detail.duration_seconds?.toFixed(1) ?? '—'}s</span>
			</div>

			<!-- Weighted score + contributions -->
			{#if detail.weighted_score}
				<div class="term-panel p-3">
					<div class="flex items-center justify-between mb-2">
						<span class="term-label flex items-center gap-1">
							Weighted Score
							<InfoTip text={panelDescription('weighted_score')} />
						</span>
						<span class="text-2xl font-mono tabular-nums">
							{(detail.weighted_score.score * 100).toFixed(1)}%
						</span>
					</div>
					<div class="flex flex-col gap-1">
						{#each sortedContributions as [objective, contribution]}
							{@const weight = detail.weighted_score.weights[objective] ?? 0}
							<div class="flex items-center gap-2 text-xs">
								<span class="w-28 shrink-0 capitalize font-mono">{objective}</span>
								<div class="flex-1 h-3 bg-base-300 rounded-sm overflow-hidden">
									<div
										class="h-full bg-primary"
										style="width: {Math.min(100, Math.max(0, contribution * 100))}%"
									></div>
								</div>
								<span class="w-16 text-right font-mono tabular-nums text-base-content/70">
									{(contribution * 100).toFixed(1)}%
								</span>
								<span class="w-16 text-right font-mono tabular-nums text-base-content/40">
									w={weight.toFixed(2)}
								</span>
							</div>
						{/each}
					</div>
				</div>
			{/if}

			<!-- Metric group tables -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-3">
				{#each GROUP_ORDER as group}
					{@const groupMetrics = metricsByGroup(group)}
					<div class="term-panel">
						<div class="term-label mb-2">
							{GROUP_LABELS[group]}
						</div>
						{#if group === 'retrieval' && detail.tier === 'generation'}
							<div class="text-center py-3 text-base-content/40 text-xs font-mono">
								n/a (generation tier)
							</div>
						{:else if groupMetrics.length === 0}
							<div class="text-center py-3 text-base-content/40 text-xs">no data</div>
						{:else}
							<table class="table table-xs term-table">
								<tbody>
									{#each groupMetrics as m}
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
												{#if group === 'generation' && typeof m.details?.std_dev === 'number'}
													<span class="text-base-content/40 text-xs ml-1">
														±{((m.details.std_dev as number) * 100).toFixed(1)}%
													</span>
												{/if}
											</td>
											{#if group === 'generation'}
												<td class="text-right text-xs text-base-content/40">
													n={m.sample_size ?? '—'}
												</td>
											{/if}
										</tr>
									{/each}
								</tbody>
							</table>
						{/if}
					</div>
				{/each}
			</div>

			<!-- Cost & speed panel -->
			<div class="term-panel">
				<div class="term-label mb-2 flex items-center gap-1">
					Cost &amp; Speed
					<InfoTip text={panelDescription('cost_speed')} />
				</div>
				<div class="grid grid-cols-2 md:grid-cols-4 gap-3">
					<div class="term-tile">
						<div class="term-label">Cost / Query</div>
						<MetricValue metricName="avg_cost_usd" value={detail.dashboard_metrics?.avg_cost_usd} format="usd" />
					</div>
					<div class="term-tile">
						<div class="term-label">Total Cost</div>
						<MetricValue metricName="total_cost_usd" value={detail.dashboard_metrics?.total_cost_usd} format="usd" />
					</div>
					<div class="term-tile">
						<div class="term-label">Prompt Tokens</div>
						<MetricValue metricName="total_prompt_tokens" value={detail.dashboard_metrics?.total_prompt_tokens} format="int" />
					</div>
					<div class="term-tile">
						<div class="term-label">Completion Tokens</div>
						<MetricValue metricName="total_completion_tokens" value={detail.dashboard_metrics?.total_completion_tokens} format="int" />
					</div>
					<div class="term-tile" title="RAG inference model whose token rates price the cost figures">
						<div class="term-label">Model (pricing)</div>
						<span class="font-mono text-xs">{costDetails.model ?? detail.dashboard_metrics?.cost_model ?? '—'}</span>
					</div>
					<div class="term-tile">
						<div class="term-label">Latency p50/p95/avg</div>
						<span class="font-mono text-xs tabular-nums">
							{latencySeconds('latency_p50_ms')}/
							{latencySeconds('latency_p95_ms')}/
							{latencySeconds('latency_avg_ms')} s
						</span>
					</div>
					<div class="term-tile">
						<div class="term-label">Latency min/max</div>
						<span class="font-mono text-xs tabular-nums">
							{typeof latencyDetails.min_ms === 'number' ? ((latencyDetails.min_ms as number) / 1000).toFixed(2) : '—'}/
							{typeof latencyDetails.max_ms === 'number' ? ((latencyDetails.max_ms as number) / 1000).toFixed(2) : '—'} s
						</span>
					</div>
				</div>
			</div>

			<!-- Config snapshot -->
			<div class="term-panel">
				<div class="term-label mb-2 flex items-center gap-1">
					Config Snapshot
					<InfoTip text={panelDescription('config_snapshot')} />
				</div>
				<div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs font-mono">
					<div><span class="text-base-content/50">LLM:</span> {detail.config.llm_model ?? '—'}</div>
					<div><span class="text-base-content/50">Provider:</span> {detail.config.llm_provider ?? '—'}</div>
					<div><span class="text-base-content/50">Embedding:</span> {detail.config.embedding_model ?? '—'}</div>
					<div><span class="text-base-content/50">Reranker:</span> {detail.config.reranker_model ?? '—'}</div>
					<div><span class="text-base-content/50">Top-K:</span> {detail.config.retrieval_top_k ?? '—'}</div>
					<div>
						<span class="text-base-content/50">Hybrid:</span>
						{detail.config.hybrid_search_enabled ? 'on' : 'off'}
					</div>
					<div>
						<span class="text-base-content/50">Contextual:</span>
						{detail.config.contextual_retrieval_enabled ? 'on' : 'off'}
					</div>
				</div>
			</div>
		{/if}
	{/if}
</div>
