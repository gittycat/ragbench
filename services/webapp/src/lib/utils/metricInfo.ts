// One-line explanations for eval metrics (shown as tooltips) and signed delta formatting.

const METRIC_DESCRIPTIONS: Record<string, string> = {
	weighted_score:
		'Headline score combining quality, latency and cost objectives by their configured weights.',
	mrr: 'Mean reciprocal rank: how high the first relevant chunk ranks in the results (1.0 = always first).',
	ndcg_at_10: 'Ranking quality of the top 10 retrieved chunks (1.0 = ideal ordering).',
	faithfulness: 'How well the answer sticks to the retrieved context — higher means fewer hallucinations.',
	answer_correctness: 'Agreement between the generated answer and the reference answer.',
	answer_relevancy: 'How directly the answer addresses the question asked.',
	citation_precision: 'Share of cited sources that actually support the answer.',
	citation_recall: 'Share of supporting sources that the answer cites.',
	section_accuracy: 'Whether citations point at the correct document section.',
	unanswerable_accuracy: 'How often the system correctly declines questions the corpus cannot answer.',
	abstention_false_positive_rate:
		'How often the system refuses questions it could have answered (lower is better).',
	abstention_false_negative_rate:
		'How often the system answers questions it should have declined (lower is better).',
	latency_p50_ms: 'Median end-to-end query time.',
	latency_p95_ms: '95th-percentile end-to-end query time — the slow tail.',
	latency_avg_ms: 'Average end-to-end query time.',
	cost_per_query: "Average LLM cost per query, priced at the RAG model's token rates.",
	avg_cost_usd: "Average LLM cost per query, priced at the RAG model's token rates.",
	total_cost_usd: 'Total LLM cost across all queries in the run.',
	duration_seconds: 'Wall-clock time for the whole eval run.'
};

/** Human display label for a metric name (underscores → spaces). */
export function metricLabel(name: string): string {
	return name.replace(/_/g, ' ');
}

export function metricDescription(name: string): string | undefined {
	if (METRIC_DESCRIPTIONS[name]) return METRIC_DESCRIPTIONS[name];
	let m = name.match(/^precision_at_(\d+)$/);
	if (m) return `Share of the top ${m[1]} retrieved chunks that are relevant.`;
	m = name.match(/^recall_at_(\d+)$/);
	if (m) return `Share of all relevant chunks found within the top ${m[1]} results.`;
	return undefined;
}

// ============================================================================
// Pipeline stage guidance — for the Overview tab's diagram + stage detail panel.
// ============================================================================

export type StageId = 'documents' | 'chunking' | 'embedding' | 'hybrid_search' | 'rerank' | 'llm';

export interface StageInfo {
	title: string;
	description: string;
	whatItDoes: string;
	ifWeak: string;
}

export const STAGE_INFO: Record<StageId, StageInfo> = {
	documents: {
		title: 'Documents',
		description: 'The source corpus that gets parsed, chunked and indexed.',
		whatItDoes:
			'Uploaded files are parsed (Docling), split into chunks, and stored alongside their vector and BM25 index entries.',
		ifWeak:
			'Retrieval can only find what was ingested well — check for parsing failures, missing documents, or chunks that are too coarse/fine for your questions.'
	},
	chunking: {
		title: 'Chunking + Contextual Retrieval',
		description: 'How documents are split into chunks, optionally prefixed with LLM-generated context.',
		whatItDoes:
			'Each chunk can be prepended with a short LLM-written summary of where it sits in the document, so isolated chunks retain surrounding context before embedding.',
		ifWeak:
			'If retrieval misses relevant chunks, try enabling contextual retrieval, or tune chunk size/overlap so chunks are self-contained without being too large.'
	},
	embedding: {
		title: 'Embedding',
		description: 'Converts chunk text into vectors for semantic (similarity) search.',
		whatItDoes: 'An embedding model turns each chunk (and each query) into a dense vector stored in ChromaDB.',
		ifWeak:
			'Low retrieval recall/relevance can mean the embedding model is too weak for your domain — try a stronger or domain-tuned embedding model.'
	},
	hybrid_search: {
		title: 'Hybrid Search',
		description: 'Combines keyword search (BM25) with vector similarity search via reciprocal rank fusion.',
		whatItDoes:
			'BM25 catches exact keyword/term matches that embeddings can miss; RRF fuses the two ranked lists into one.',
		ifWeak:
			'If recall/precision is low, try enabling hybrid search (if off), raising retrieval top-k, or tuning the RRF k parameter.'
	},
	rerank: {
		title: 'Rerank',
		description: 'A cross-encoder re-scores the retrieved chunks for tighter relevance ordering before the LLM sees them.',
		whatItDoes:
			'Takes the top-k retrieved chunks and reorders/truncates them to a smaller, higher-precision final set.',
		ifWeak:
			'If precision is low despite good retrieval, try enabling reranking, a stronger reranker model, or reducing final top-n to cut noise.'
	},
	llm: {
		title: 'LLM Answer',
		description: 'The LLM reads the retrieved context and produces the final answer, with citations.',
		whatItDoes:
			'Generates the answer from the query plus retrieved chunks, deciding what to cite and when to abstain.',
		ifWeak:
			'Low faithfulness/correctness → try a stronger LLM, fewer but more relevant chunks, or prompt tuning. Low citation precision → prompt tuning or a stronger LLM. Abstention errors → tune the abstention prompt/thresholds.'
	}
};

// ============================================================================
// Panel-level help text — one or two sentences, written for someone new to RAG evals.
// ============================================================================

export const PANEL_DESCRIPTIONS = {
	weighted_score:
		'A single headline number combining quality, latency and cost objectives by their configured weights — higher is better.',
	cost_speed: 'What each query costs in LLM tokens and how long it takes end-to-end, including the slow-tail p95 latency.',
	config_snapshot: 'The exact models and retrieval settings used for this run, so results can be attributed to a configuration.',
	run_meta:
		'Tier is the eval depth (generation-only vs. full end-to-end), seed controls sample selection for reproducibility, and samples/dataset caps how many questions were drawn per dataset.',
	trend_weighted_score: 'Headline score over time — watch for regressions after config changes.',
	trend_faithfulness: 'How well answers stick to retrieved context over time — drops often mean a weaker LLM or noisier retrieval.',
	trend_answer_correctness: 'Agreement with reference answers over time.',
	trend_latency_p95: 'Slow-tail end-to-end latency over time, in seconds.',
	trend_cost: 'Average LLM cost per query over time, in USD.',
	compare_delta: 'Newer run (B) minus the baseline (A). Green means the change improved this metric; red means it regressed.',
	compare_headline: 'Per-run summary of quality score, cost and speed, with deltas versus the baseline run (A).',
	compare_quality: 'Full scorecard for each selected run, grouped by retrieval, generation, citation and abstention metrics.',
	compare_cost_speed: 'Per-query cost and latency for each selected run, with the delta versus the baseline run (A).'
};

export type PanelKey = keyof typeof PANEL_DESCRIPTIONS;

export function panelDescription(key: PanelKey): string {
	return PANEL_DESCRIPTIONS[key];
}

export type DeltaFormat = 'pts' | 'seconds' | 'usd' | 'int';

/**
 * Always-signed delta string: 'pts' renders a 0-1 score difference in
 * percentage points ("+5.0 pts"), the rest keep the metric's unit.
 */
export function formatDelta(value: number, format: DeltaFormat = 'pts', decimals?: number): string {
	const sign = value >= 0 ? '+' : '-';
	const abs = Math.abs(value);
	switch (format) {
		case 'pts':
			return `${sign}${(abs * 100).toFixed(decimals ?? 1)} pts`;
		case 'seconds':
			return `${sign}${abs.toFixed(decimals ?? 2)}s`;
		case 'usd':
			return `${sign}$${abs.toFixed(decimals ?? 4)}`;
		case 'int':
			return `${sign}${Math.round(abs).toLocaleString()}`;
	}
}
