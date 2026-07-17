// Rolls scorecard metrics up into two measurable "brackets" (retrieval, generation)
// and identifies the weakest one, so the Overview tab can point at a single
// highest-leverage change instead of dumping a wall of metrics.

import type { EvalRunDetail, ScorecardMetric } from '$lib/api/evals';
import { getMetricThreshold } from './thresholds';
import { metricLabel } from './metricInfo';

export type Health = 'good' | 'warn' | 'bad' | 'unknown';
export type Bracket = 'retrieval' | 'generation';

const RETRIEVAL_GROUPS = new Set(['retrieval']);
const GENERATION_GROUPS = new Set(['generation', 'citation', 'abstention']);

const HEALTH_RANK: Record<Health, number> = { unknown: 0, good: 1, warn: 2, bad: 3 };

export function bracketForGroup(group: string): Bracket | null {
	if (RETRIEVAL_GROUPS.has(group)) return 'retrieval';
	if (GENERATION_GROUPS.has(group)) return 'generation';
	return null;
}

/**
 * Band a single metric's value using thresholds.ts's good/warn cutoffs.
 * Returns undefined for metrics thresholds.ts can't meaningfully band
 * (e.g. raw latency-ms or cost-usd values, which use placeholder 0.2/0.4
 * bands intended for 0-1 rates, not real units).
 */
function bandMetric(name: string, value: number | null | undefined): Health | undefined {
	if (value === null || value === undefined || Number.isNaN(value)) return undefined;
	const t = getMetricThreshold(name);
	if (t.direction === 'higher') {
		if (value >= t.good) return 'good';
		if (value >= t.warn) return 'warn';
		return 'bad';
	}
	// lower-is-better: only bandable for known 0-1 rate metrics (matches thresholdColorClass)
	if (!name.includes('rate')) return undefined;
	if (value <= t.good) return 'good';
	if (value <= t.warn) return 'warn';
	return 'bad';
}

/** Normalized "how far below (or above, for rates) the good line" — bigger is worse. */
function deviation(name: string, value: number): number {
	const t = getMetricThreshold(name);
	if (t.direction === 'higher') return t.good - value;
	return value - t.good;
}

export interface BandedMetric {
	metric: ScorecardMetric;
	band: Health;
	deviation: number;
}

export interface BracketHealth {
	bracket: Bracket;
	health: Health;
	metrics: ScorecardMetric[];
	banded: BandedMetric[];
}

function worstHealth(bands: Health[]): Health {
	if (bands.length === 0) return 'unknown';
	return bands.reduce((worst, h) => (HEALTH_RANK[h] > HEALTH_RANK[worst] ? h : worst), 'good' as Health);
}

export function computeBracketHealth(detail: EvalRunDetail | null | undefined): Record<Bracket, BracketHealth> {
	const empty = (bracket: Bracket): BracketHealth => ({ bracket, health: 'unknown', metrics: [], banded: [] });
	const result: Record<Bracket, BracketHealth> = {
		retrieval: empty('retrieval'),
		generation: empty('generation')
	};

	const metrics = detail?.scorecard?.metrics ?? [];
	for (const bracket of ['retrieval', 'generation'] as Bracket[]) {
		const bracketMetrics = metrics.filter((m) => bracketForGroup(m.group) === bracket);
		const banded: BandedMetric[] = bracketMetrics
			.map((metric) => {
				const band = bandMetric(metric.name, metric.value);
				return band ? { metric, band, deviation: deviation(metric.name, metric.value) } : null;
			})
			.filter((b): b is BandedMetric => b !== null);
		result[bracket] = {
			bracket,
			health: worstHealth(banded.map((b) => b.band)),
			metrics: bracketMetrics,
			banded
		};
	}
	return result;
}

export interface WeakestLink {
	bracket: Bracket;
	metricName: string;
	value: number;
	band: Health;
}

/**
 * The single weakest stage + metric across both brackets, or null when
 * nothing is actually weak (all bandable metrics are good) or there's no
 * usable data at all (no scorecard, or nothing bandable).
 */
export function computeWeakestLink(health: Record<Bracket, BracketHealth>): WeakestLink | null {
	const brackets = (['retrieval', 'generation'] as Bracket[])
		.map((b) => health[b])
		.filter((h) => h.health === 'warn' || h.health === 'bad');

	if (brackets.length === 0) return null;

	// Pick the worse-health bracket; tie-break on the larger single-metric deviation.
	const worstRank = Math.max(...brackets.map((b) => HEALTH_RANK[b.health]));
	const candidates = brackets.filter((b) => HEALTH_RANK[b.health] === worstRank);

	let best: { bracket: Bracket; banded: BandedMetric } | null = null;
	for (const b of candidates) {
		const worstInBracket = b.banded.filter((m) => m.band === b.health);
		for (const m of worstInBracket) {
			if (!best || m.deviation > best.banded.deviation) {
				best = { bracket: b.bracket, banded: m };
			}
		}
	}
	if (!best) return null;

	return {
		bracket: best.bracket,
		metricName: best.banded.metric.name,
		value: best.banded.metric.value,
		band: best.banded.band
	};
}

const BRACKET_LABELS: Record<Bracket, string> = { retrieval: 'Retrieval', generation: 'Generation' };

/** One-sentence human verdict for the diagnosis banner. */
export function weakestLinkVerdict(
	detail: EvalRunDetail | null | undefined,
	health: Record<Bracket, BracketHealth>,
	weak: WeakestLink | null
): string {
	if (!detail?.scorecard?.metrics?.length) {
		return 'No eval data yet — run an evaluation to see pipeline health.';
	}

	if (!weak) {
		const anyBanded = (['retrieval', 'generation'] as Bracket[]).some(
			(b) => health[b].banded.length > 0
		);
		if (!anyBanded) return 'No bandable metrics in this run — check the scorecard for raw values.';
		return 'No weak stage detected — all metrics above thresholds.';
	}

	const otherBracket: Bracket = weak.bracket === 'retrieval' ? 'generation' : 'retrieval';
	const otherHealth = health[otherBracket].health;
	const otherLooksFine = otherHealth === 'good' || otherHealth === 'unknown';

	const metricLbl = metricLabel(weak.metricName);
	const valueStr = weak.value.toFixed(2);
	const bracketLbl = BRACKET_LABELS[weak.bracket];

	const advice =
		weak.bracket === 'retrieval'
			? 'improving retrieval (top-k, hybrid search, or the embedding model) is the highest-leverage change'
			: 'improving the LLM or prompt is the highest-leverage change';

	const context = otherLooksFine
		? `${BRACKET_LABELS[otherBracket]} looks healthy; `
		: '';

	return `Weakest link: ${bracketLbl} — ${metricLbl} ${valueStr}. ${context}${advice}.`;
}
