/**
 * Export utilities for eval run data
 * Client-side CSV and JSON generation
 */
import type { EvalRunSummary, EvalCompareResponse } from '$lib/api/evals';

export interface ExportData {
	runs: EvalRunSummary[];
	compare?: EvalCompareResponse;
	exportedAt: string;
}

/**
 * Export eval run summaries to CSV format
 */
export function exportToCSV(runs: EvalRunSummary[]): void {
	if (runs.length === 0) {
		console.warn('No runs to export');
		return;
	}

	// Get all unique metric names across all runs
	const allMetrics = new Set<string>();
	for (const run of runs) {
		for (const metric of Object.keys(run.metrics)) {
			allMetrics.add(metric);
		}
	}
	const metrics = Array.from(allMetrics).sort();

	const headers = [
		'run_id',
		'name',
		'created_at',
		'tier',
		'datasets',
		'question_count',
		'error_count',
		'duration_seconds',
		'weighted_score',
		'avg_cost_usd',
		'total_cost_usd',
		...metrics.map((m) => `metric_${m}`)
	];

	const rows = runs.map((run) => {
		const values = [
			run.id,
			run.name,
			run.created_at,
			run.tier,
			run.datasets.join('|'),
			run.question_count.toString(),
			run.error_count.toString(),
			run.duration_seconds?.toFixed(1) ?? '',
			run.weighted_score !== null ? (run.weighted_score * 100).toFixed(1) : '',
			run.dashboard_metrics?.avg_cost_usd?.toFixed(4) ?? '',
			run.dashboard_metrics?.total_cost_usd?.toFixed(4) ?? '',
			...metrics.map((m) => (run.metrics[m] !== undefined ? run.metrics[m].toFixed(4) : ''))
		];
		return values.map(escapeCSV).join(',');
	});

	const csv = [headers.join(','), ...rows].join('\n');
	downloadFile(csv, generateFilename('csv'), 'text/csv');
}

/**
 * Export eval run data (and optional compare result) to JSON format
 */
export function exportToJSON(data: ExportData): void {
	const json = JSON.stringify(data, null, 2);
	downloadFile(json, generateFilename('json'), 'application/json');
}

/**
 * Escape a value for CSV (handle commas, quotes, newlines)
 */
function escapeCSV(value: string): string {
	if (value.includes(',') || value.includes('"') || value.includes('\n')) {
		return `"${value.replace(/"/g, '""')}"`;
	}
	return value;
}

/**
 * Generate a filename with timestamp
 */
function generateFilename(extension: string): string {
	const date = new Date().toISOString().slice(0, 10);
	const time = new Date().toISOString().slice(11, 16).replace(':', '');
	return `eval-comparison-${date}-${time}.${extension}`;
}

/**
 * Trigger a file download in the browser
 */
function downloadFile(content: string, filename: string, mimeType: string): void {
	const blob = new Blob([content], { type: mimeType });
	const url = URL.createObjectURL(blob);

	const link = document.createElement('a');
	link.href = url;
	link.download = filename;
	link.style.display = 'none';

	document.body.appendChild(link);
	link.click();

	document.body.removeChild(link);
	URL.revokeObjectURL(url);
}
