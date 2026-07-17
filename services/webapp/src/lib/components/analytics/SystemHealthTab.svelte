<script lang="ts">
	import type { SystemMetrics } from '$lib/api';

	interface Props {
		metrics: SystemMetrics | null;
	}

	let { metrics }: Props = $props();

	function getStatusBadge(status: string): string {
		const s = status?.toLowerCase() || '';
		if (['healthy', 'loaded', 'available'].includes(s)) return 'badge-success';
		if (['unavailable', 'unhealthy', 'error'].includes(s)) return 'badge-error';
		return 'badge-warning';
	}

	function formatSize(mb: number | undefined): string {
		if (!mb) return '-';
		if (mb >= 1024) return `${(mb / 1024).toFixed(1)}G`;
		return `${mb.toFixed(0)}M`;
	}
</script>

{#if !metrics}
	<div class="flex items-center justify-center h-64">
		<span class="loading loading-spinner loading-lg"></span>
	</div>
{:else}
	<div class="flex flex-col gap-3">
		<!-- Component Status -->
		<div class="term-panel">
			<div class="term-label mb-2">
				Component Status
			</div>
			<div class="flex flex-wrap gap-2">
				{#each Object.entries(metrics.component_status) as [component, status]}
					<div class="flex items-center gap-1.5 bg-base-100 border border-base-content/10 rounded-sm px-2 py-1">
						<span class="h-2 w-2 rounded-full {getStatusBadge(status).replace('badge-', 'bg-')}"></span>
						<span class="text-xs capitalize">{component}</span>
						<span class="badge badge-xs {getStatusBadge(status)}">{status}</span>
					</div>
				{/each}
			</div>
		</div>

		<!-- Models Table -->
		<div class="term-panel">
			<div class="term-label mb-2">
				Models Configuration
			</div>
			<div class="overflow-x-auto">
				<table class="table table-xs term-table">
					<thead>
						<tr>
							<th>Type</th>
							<th>Model</th>
							<th>Provider</th>
							<th class="text-right">Params</th>
							<th class="text-right">Size</th>
							<th>Status</th>
						</tr>
					</thead>
					<tbody>
						<tr class="hover">
							<td class="font-semibold">LLM</td>
							<td class="font-mono text-xs truncate max-w-32" title={metrics.models.llm.name}>
								{metrics.models.llm.name}
							</td>
							<td class="text-xs">{metrics.models.llm.provider}</td>
							<td class="text-right text-xs">{metrics.models.llm.size?.parameters || '-'}</td>
							<td class="text-right text-xs">{formatSize(metrics.models.llm.size?.disk_size_mb)}</td>
							<td>
								<span class="badge badge-xs {getStatusBadge(metrics.models.llm.status)}">
									{metrics.models.llm.status}
								</span>
							</td>
						</tr>
						<tr class="hover">
							<td class="font-semibold">Embedding</td>
							<td class="font-mono text-xs truncate max-w-32" title={metrics.models.embedding.name}>
								{metrics.models.embedding.name}
							</td>
							<td class="text-xs">{metrics.models.embedding.provider}</td>
							<td class="text-right text-xs">{metrics.models.embedding.size?.parameters || '-'}</td>
							<td class="text-right text-xs">{formatSize(metrics.models.embedding.size?.disk_size_mb)}</td>
							<td>
								<span class="badge badge-xs {getStatusBadge(metrics.models.embedding.status)}">
									{metrics.models.embedding.status}
								</span>
							</td>
						</tr>
						{#if metrics.models.reranker}
							<tr class="hover">
								<td class="font-semibold">Reranker</td>
								<td class="font-mono text-xs truncate max-w-32" title={metrics.models.reranker.name}>
									{metrics.models.reranker.name}
								</td>
								<td class="text-xs">{metrics.models.reranker.provider}</td>
								<td class="text-right text-xs">{metrics.models.reranker.size?.parameters || '-'}</td>
								<td class="text-right text-xs">{formatSize(metrics.models.reranker.size?.disk_size_mb)}</td>
								<td>
									<span class="badge badge-xs {getStatusBadge(metrics.models.reranker.status)}">
										{metrics.models.reranker.status}
									</span>
								</td>
							</tr>
						{/if}
						{#if metrics.models.eval}
							<tr class="hover">
								<td class="font-semibold">Eval</td>
								<td class="font-mono text-xs truncate max-w-32" title={metrics.models.eval.name}>
									{metrics.models.eval.name}
								</td>
								<td class="text-xs">{metrics.models.eval.provider}</td>
								<td class="text-right text-xs">{metrics.models.eval.size?.parameters || '-'}</td>
								<td class="text-right text-xs">{formatSize(metrics.models.eval.size?.disk_size_mb)}</td>
								<td>
									<span class="badge badge-xs {getStatusBadge(metrics.models.eval.status)}">
										{metrics.models.eval.status}
									</span>
								</td>
							</tr>
						{/if}
					</tbody>
				</table>
			</div>
		</div>

		<!-- Document Stats -->
		<div class="term-panel">
			<div class="term-label mb-2">
				Index Statistics
			</div>
			<div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
				<div class="term-tile">
					<div class="text-2xl font-semibold font-mono tabular-nums">{metrics.document_count}</div>
					<div class="term-label">Documents</div>
				</div>
				<div class="term-tile">
					<div class="text-2xl font-semibold font-mono tabular-nums">{metrics.chunk_count}</div>
					<div class="term-label">Chunks</div>
				</div>
				<div class="term-tile">
					<div class="text-2xl font-semibold font-mono tabular-nums">{metrics.retrieval.retrieval_top_k}</div>
					<div class="term-label">Top-K Retrieved</div>
				</div>
				<div class="term-tile">
					<div class="text-2xl font-semibold font-mono tabular-nums">{metrics.retrieval.final_top_n}</div>
					<div class="term-label">Final Top-N</div>
				</div>
			</div>
		</div>
	</div>
{/if}
