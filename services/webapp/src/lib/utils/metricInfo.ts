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

export function metricDescription(name: string): string | undefined {
	if (METRIC_DESCRIPTIONS[name]) return METRIC_DESCRIPTIONS[name];
	let m = name.match(/^precision_at_(\d+)$/);
	if (m) return `Share of the top ${m[1]} retrieved chunks that are relevant.`;
	m = name.match(/^recall_at_(\d+)$/);
	if (m) return `Share of all relevant chunks found within the top ${m[1]} results.`;
	return undefined;
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
