<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Tab {
		id: string;
		label: string;
		icon?: string;
	}

	interface Props {
		activeTab: string;
		onTabChange: (tabId: string) => void;
		tabs: Tab[];
		children: Snippet;
	}

	let { activeTab, onTabChange, tabs, children }: Props = $props();
</script>

<div class="flex flex-col gap-3">
	<!-- Tab Navigation -->
	<div role="tablist" class="tabs tabs-box tabs-sm bg-base-200 border border-base-content/10 rounded-sm w-fit">
		{#each tabs as tab}
			<button
				role="tab"
				class="tab gap-1.5 font-mono text-xs uppercase tracking-wider"
				class:tab-active={activeTab === tab.id}
				onclick={() => onTabChange(tab.id)}
				aria-selected={activeTab === tab.id}
			>
				{#if tab.icon === 'scorecard'}
					<svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
					</svg>
				{:else if tab.icon === 'trends'}
					<svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 17l6-6 4 4 8-8m0 0h-5m5 0v5" />
					</svg>
				{:else if tab.icon === 'compare'}
					<svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
					</svg>
				{:else if tab.icon === 'system'}
					<svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
					</svg>
				{/if}
				{tab.label}
			</button>
		{/each}
	</div>

	<!-- Tab Content -->
	<div>
		{@render children()}
	</div>
</div>
