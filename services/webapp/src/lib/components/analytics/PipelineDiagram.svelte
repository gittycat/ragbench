<script lang="ts">
	import type { SystemMetrics } from '$lib/api';
	import type { StageId } from '$lib/utils/metricInfo';
	import type { Bracket, BracketHealth, Health } from '$lib/utils/stageHealth';
	import InfoTip from './InfoTip.svelte';

	interface Props {
		metrics: SystemMetrics;
		bracketHealth: Record<Bracket, BracketHealth>;
		selectedStage: StageId | null;
		onSelectStage: (stage: StageId) => void;
	}

	let { metrics, bracketHealth, selectedStage, onSelectStage }: Props = $props();

	interface StageNode {
		id: StageId;
		label: string;
		details: string[];
		dimmed?: boolean;
	}

	function shortModelName(name: string | undefined | null): string {
		if (!name) return '—';
		const parts = name.split('/');
		return parts[parts.length - 1];
	}

	let documentsNode = $derived.by(
		(): StageNode => ({
			id: 'documents',
			label: 'Documents',
			details: [`${metrics.document_count} docs`, `${metrics.chunk_count.toLocaleString()} chunks`]
		})
	);

	let chunkingNode = $derived.by((): StageNode => {
		const contextual = metrics.retrieval.contextual_retrieval.enabled;
		return {
			id: 'chunking',
			label: contextual ? 'Chunking + Contextual' : 'Chunking',
			details: contextual ? ['context prefix on'] : ['context prefix off']
		};
	});

	let embeddingNode = $derived.by(
		(): StageNode => ({
			id: 'embedding',
			label: 'Embedding',
			details: [shortModelName(metrics.models.embedding?.name)]
		})
	);

	let hybridNode = $derived.by((): StageNode => {
		const hs = metrics.retrieval.hybrid_search;
		if (hs.enabled) {
			return {
				id: 'hybrid_search',
				label: 'Hybrid Search',
				details: [`BM25+Vector RRF k=${hs.rrf_k}`, `top-k ${metrics.retrieval.retrieval_top_k}`]
			};
		}
		return {
			id: 'hybrid_search',
			label: 'Vector Search',
			details: [`top-k ${metrics.retrieval.retrieval_top_k}`]
		};
	});

	let rerankNode = $derived.by((): StageNode => {
		const rr = metrics.retrieval.reranker;
		if (!rr.enabled) {
			return { id: 'rerank', label: 'Rerank', details: ['disabled'], dimmed: true };
		}
		return {
			id: 'rerank',
			label: 'Rerank',
			details: [shortModelName(rr.model), `→ top-${metrics.retrieval.final_top_n}`]
		};
	});

	let llmNode = $derived.by(
		(): StageNode => ({
			id: 'llm',
			label: 'LLM Answer',
			details: [shortModelName(metrics.models.llm?.name)]
		})
	);

	let retrievalNodes = $derived([chunkingNode, embeddingNode, hybridNode, rerankNode]);

	function healthBorder(health: Health): string {
		switch (health) {
			case 'good':
				return 'border-success/50';
			case 'warn':
				return 'border-warning/50';
			case 'bad':
				return 'border-error/50';
			default:
				return 'border-base-content/15';
		}
	}

	function healthDot(health: Health): string {
		switch (health) {
			case 'good':
				return 'bg-success';
			case 'warn':
				return 'bg-warning';
			case 'bad':
				return 'bg-error';
			default:
				return 'bg-base-content/25';
		}
	}
</script>

{#snippet arrow()}
	<span class="text-base-content/25 font-mono px-0.5 shrink-0 self-center">→</span>
{/snippet}

{#snippet node(n: StageNode)}
	<button
		type="button"
		class="term-tile flex flex-col items-center gap-0.5 min-w-[7rem] text-center transition-colors {selectedStage ===
		n.id
			? 'border-primary ring-1 ring-primary/40'
			: 'hover:border-base-content/30'} {n.dimmed ? 'opacity-50' : ''}"
		onclick={() => onSelectStage(n.id)}
		aria-pressed={selectedStage === n.id}
	>
		<span class="text-xs font-semibold">{n.label}</span>
		{#each n.details as d}
			<span class="text-[10px] font-mono text-base-content/50 tabular-nums">{d}</span>
		{/each}
	</button>
{/snippet}

<div class="term-panel">
	<div class="flex flex-wrap items-center gap-1.5">
		{@render node(documentsNode)}
		{@render arrow()}

		<!-- Retrieval bracket: measured jointly by retrieval-group eval metrics -->
		<div class="flex flex-col gap-1 border rounded-sm p-1.5 {healthBorder(bracketHealth.retrieval.health)}">
			<div class="flex items-center gap-1 justify-center">
				<span class="h-1.5 w-1.5 rounded-full {healthDot(bracketHealth.retrieval.health)}"></span>
				<span class="term-label">Retrieval (measured together)</span>
				<InfoTip
					text="Eval metrics score chunking, embedding, hybrid search and reranking as one pipeline — a low score can't be pinned to a single sub-stage without manual digging."
				/>
			</div>
			<div class="flex flex-wrap items-center gap-1.5">
				{@render node(chunkingNode)}
				{@render arrow()}
				{@render node(embeddingNode)}
				{@render arrow()}
				{@render node(hybridNode)}
				{@render arrow()}
				{@render node(rerankNode)}
			</div>
		</div>

		{@render arrow()}

		<!-- Generation bracket: the LLM answer stage -->
		<div class="flex flex-col gap-1 border rounded-sm p-1.5 {healthBorder(bracketHealth.generation.health)}">
			<div class="flex items-center gap-1 justify-center">
				<span class="h-1.5 w-1.5 rounded-full {healthDot(bracketHealth.generation.health)}"></span>
				<span class="term-label">Generation</span>
				<InfoTip
					text="Faithfulness, correctness, citation and abstention metrics score the LLM's answer as a whole — prompt, model choice and context quality all factor in together."
				/>
			</div>
			{@render node(llmNode)}
		</div>
	</div>
</div>
