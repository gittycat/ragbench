<script lang="ts">
	import type { EvalRunSummary } from '$lib/api/evals';

	interface Props {
		runs: EvalRunSummary[];
		selected: string[];
		onSelectionChange: (ids: string[]) => void;
		maxSelection?: number;
	}

	let { runs, selected, onSelectionChange, maxSelection = 4 }: Props = $props();

	function toggleRun(runId: string) {
		if (selected.includes(runId)) {
			onSelectionChange(selected.filter((id) => id !== runId));
		} else if (selected.length < maxSelection) {
			onSelectionChange([...selected, runId]);
		}
	}

	function selectAll() {
		const toSelect = runs.slice(0, maxSelection).map((r) => r.id);
		onSelectionChange(toSelect);
	}

	function clearAll() {
		onSelectionChange([]);
	}

	function getRunTime(run: EvalRunSummary): string {
		return new Date(run.created_at).toLocaleTimeString('en-US', {
			hour: '2-digit',
			minute: '2-digit',
			hour12: false
		});
	}

	function getRunDate(run: EvalRunSummary): string {
		return new Date(run.created_at).toLocaleDateString('en-US', {
			month: 'short',
			day: 'numeric'
		});
	}

	function timeAgo(timestamp: string): string {
		const diff = Date.now() - new Date(timestamp).getTime();
		const mins = Math.floor(diff / 60000);
		if (mins < 60) return `${mins}m ago`;
		const hours = Math.floor(mins / 60);
		if (hours < 24) return `${hours}h ago`;
		const days = Math.floor(hours / 24);
		return `${days}d ago`;
	}

	function getScoreColor(score: number | null): string {
		if (score === null) return 'badge-ghost';
		if (score >= 0.8) return 'badge-success';
		if (score >= 0.6) return 'badge-warning';
		return 'badge-error';
	}
</script>

<div class="term-panel">
	<div class="flex items-center justify-between mb-2">
		<span class="term-label">
			Select Runs ({selected.length}/{maxSelection})
		</span>
		<div class="flex gap-1">
			<button
				class="btn btn-xs btn-ghost"
				onclick={selectAll}
				disabled={runs.length === 0 || selected.length === Math.min(runs.length, maxSelection)}
			>
				All
			</button>
			<button class="btn btn-xs btn-ghost" onclick={clearAll} disabled={selected.length === 0}>
				Clear
			</button>
		</div>
	</div>

	{#if runs.length === 0}
		<div class="text-center py-4 text-base-content/50 text-xs">
			No evaluation runs available
		</div>
	{:else}
		<div class="max-h-56 overflow-y-auto space-y-1">
			{#each runs as run}
				{@const isSelected = selected.includes(run.id)}
				{@const isDisabled = !isSelected && selected.length >= maxSelection}
				<label
					class="flex items-center gap-2 p-1.5 rounded cursor-pointer text-xs
						{isSelected ? 'bg-primary/10' : 'hover:bg-base-300'}
						{isDisabled ? 'opacity-50 cursor-not-allowed' : ''}"
				>
					<input
						type="checkbox"
						class="checkbox checkbox-xs checkbox-primary"
						checked={isSelected}
						disabled={isDisabled}
						onchange={() => toggleRun(run.id)}
					/>
					<div class="flex-1 min-w-0">
						<div class="font-mono truncate" title={run.name}>
							{run.name}
						</div>
						<div class="text-base-content/50 flex items-center gap-2">
							<span>{getRunDate(run)} {getRunTime(run)}</span>
							<span class="badge badge-xs {getScoreColor(run.weighted_score)}">
								{run.weighted_score !== null ? `${(run.weighted_score * 100).toFixed(0)}%` : '—'}
							</span>
						</div>
					</div>
					<div class="text-base-content/40 text-right whitespace-nowrap">
						{timeAgo(run.created_at)}
					</div>
					{#if run.error_count > 0}
						<span class="badge badge-xs badge-error">{run.error_count} err</span>
					{/if}
				</label>
			{/each}
		</div>
	{/if}
</div>
