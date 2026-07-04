<script lang="ts">
	import type { EvalRunConfig } from '$lib/api/evals';
	import { diffConfigs, getDiffLineClasses, getDiffPrefix } from '$lib/utils/diff';

	interface Props {
		configA: EvalRunConfig | null | undefined;
		configB: EvalRunConfig | null | undefined;
		labelA?: string;
		labelB?: string;
		showUnchanged?: boolean;
	}

	let {
		configA,
		configB,
		labelA = 'Before',
		labelB = 'After',
		showUnchanged = true
	}: Props = $props();

	let diff = $derived(diffConfigs(configA, configB));

	let filteredDiff = $derived(
		showUnchanged ? diff : diff.filter((line) => line.type !== 'same')
	);

	let hasChanges = $derived(diff.some((line) => line.type !== 'same'));
</script>

<div class="term-panel">
	<div class="flex items-center gap-2 mb-2 text-xs text-base-content/70">
		<span class="badge badge-ghost badge-sm">{labelA}</span>
		<span>→</span>
		<span class="badge badge-ghost badge-sm">{labelB}</span>
		{#if !hasChanges}
			<span class="badge badge-success badge-sm ml-auto">No changes</span>
		{/if}
	</div>

	{#if !configA && !configB}
		<div class="text-center py-4 text-base-content/50 text-xs">
			Select two runs to compare configurations
		</div>
	{:else}
		<div class="font-mono text-xs overflow-x-auto">
			{#each filteredDiff as line}
				<div class="py-0.5 px-2 rounded-sm {getDiffLineClasses(line.type)}">
					<span class="opacity-50 mr-1">{getDiffPrefix(line.type)}</span>
					<span class="font-semibold">{line.key}:</span>
					<span class="ml-1">{line.value}</span>
					{#if line.type === 'changed' && line.oldValue}
						<span class="text-base-content/50 line-through ml-2">{line.oldValue}</span>
					{/if}
				</div>
			{/each}
		</div>

		{#if !showUnchanged && diff.length > filteredDiff.length}
			<div class="text-center py-1 text-base-content/50 text-xs mt-2">
				{diff.length - filteredDiff.length} unchanged fields hidden
			</div>
		{/if}
	{/if}
</div>
