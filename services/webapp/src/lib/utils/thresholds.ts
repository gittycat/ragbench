// Metric threshold bands for scorecard/compare coloring.
// direction 'higher' = higher value is better (quality scores),
// direction 'lower' = lower value is better (latency, cost, false-rate metrics).

export interface MetricThreshold {
	good: number;
	warn: number;
	direction: 'higher' | 'lower';
}

const LOWER_IS_BETTER = new Set([
	'abstention_false_positive_rate',
	'abstention_false_negative_rate',
	'latency_p50_ms',
	'latency_p95_ms',
	'latency_avg_ms',
	'latency_p50_seconds',
	'latency_p95_seconds',
	'latency_avg_seconds',
	'cost_per_query',
	'avg_cost_usd',
	'total_cost_usd'
]);

const DEFAULT_HIGHER: MetricThreshold = { good: 0.8, warn: 0.6, direction: 'higher' };
const DEFAULT_LOWER: MetricThreshold = { good: 0.2, warn: 0.4, direction: 'lower' };

export function getMetricThreshold(metricName: string): MetricThreshold {
	if (LOWER_IS_BETTER.has(metricName)) return DEFAULT_LOWER;
	return DEFAULT_HIGHER;
}

/** Text color class for a metric value, respecting per-metric direction. Returns '' if value is null/undefined. */
export function thresholdColorClass(metricName: string, value: number | null | undefined): string {
	if (value === null || value === undefined) return '';
	const t = getMetricThreshold(metricName);

	if (t.direction === 'higher') {
		if (value >= t.good) return 'text-success';
		if (value >= t.warn) return 'text-warning';
		return 'text-error';
	}
	// lower-is-better: thresholds are normalized rates (0-1); latency/cost don't have
	// fixed bands here, so only apply coloring for known 0-1 rate metrics.
	if (metricName.includes('rate')) {
		if (value <= t.good) return 'text-success';
		if (value <= t.warn) return 'text-warning';
		return 'text-error';
	}
	return '';
}

/** Delta color class: green if the change is an improvement for this metric, red if a regression. */
export function deltaColorClass(metricName: string, delta: number | null | undefined): string {
	if (delta === null || delta === undefined || delta === 0) return '';
	const t = getMetricThreshold(metricName);
	const improved = t.direction === 'higher' ? delta > 0 : delta < 0;
	return improved ? 'text-success' : 'text-error';
}
