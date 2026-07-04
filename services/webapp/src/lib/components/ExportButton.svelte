<script lang="ts">
	import type { EvalRunSummary, EvalCompareResponse } from '$lib/api/evals';
	import { exportToCSV, exportToJSON } from '$lib/utils/export';

	interface Props {
		runs: EvalRunSummary[];
		compare?: EvalCompareResponse;
		disabled?: boolean;
	}

	let { runs, compare, disabled = false }: Props = $props();

	let isOpen = $state(false);

	function handleExport(format: 'csv' | 'json') {
		if (runs.length === 0) return;

		if (format === 'csv') {
			exportToCSV(runs);
		} else {
			exportToJSON({
				runs,
				compare,
				exportedAt: new Date().toISOString()
			});
		}

		isOpen = false;
	}

	function handleClickOutside(event: MouseEvent) {
		const target = event.target as HTMLElement;
		if (!target.closest('.dropdown')) {
			isOpen = false;
		}
	}
</script>

<svelte:window onclick={handleClickOutside} />

<div class="dropdown dropdown-end" class:dropdown-open={isOpen}>
	<button
		class="btn btn-sm btn-ghost gap-1"
		disabled={disabled || runs.length === 0}
		onclick={() => (isOpen = !isOpen)}
	>
		<svg
			xmlns="http://www.w3.org/2000/svg"
			class="h-4 w-4"
			fill="none"
			viewBox="0 0 24 24"
			stroke="currentColor"
		>
			<path
				stroke-linecap="round"
				stroke-linejoin="round"
				stroke-width="2"
				d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
			/>
		</svg>
		Export
	</button>
	{#if isOpen}
		<ul class="dropdown-content menu bg-base-200 rounded-box z-50 w-36 p-2 shadow-lg">
			<li>
				<button onclick={() => handleExport('csv')} class="text-sm">
					<svg
						xmlns="http://www.w3.org/2000/svg"
						class="h-4 w-4"
						fill="none"
						viewBox="0 0 24 24"
						stroke="currentColor"
					>
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							stroke-width="2"
							d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
						/>
					</svg>
					CSV
				</button>
			</li>
			<li>
				<button onclick={() => handleExport('json')} class="text-sm">
					<svg
						xmlns="http://www.w3.org/2000/svg"
						class="h-4 w-4"
						fill="none"
						viewBox="0 0 24 24"
						stroke="currentColor"
					>
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							stroke-width="2"
							d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
						/>
					</svg>
					JSON
				</button>
			</li>
		</ul>
	{/if}
</div>
