<script lang="ts">
	interface Props {
		text: string;
	}

	let { text }: Props = $props();

	let button = $state<HTMLButtonElement | null>(null);
	let visible = $state(false);
	let x = $state(0);
	let y = $state(0);

	// Bubble is position:fixed so it can't be clipped by overflow-x-auto panels;
	// clamp its center so a 16rem-wide bubble stays inside the viewport.
	const HALF_WIDTH = 128 + 8;

	function show() {
		if (!button) return;
		const r = button.getBoundingClientRect();
		x = Math.min(Math.max(r.left + r.width / 2, HALF_WIDTH), window.innerWidth - HALF_WIDTH);
		y = r.bottom + 6;
		visible = true;
	}

	function hide() {
		visible = false;
	}
</script>

<!-- Hide on any scroll/resize since the fixed bubble doesn't track its trigger. -->
<svelte:window onscrollcapture={() => visible && hide()} onresize={() => visible && hide()} />

<!--
	Small inline info tooltip. The trigger is a real <button> (not a span) so
	the tooltip — shown on both hover and focus — also works on tap/keyboard
	focus for touch and a11y.
-->
<span class="inline-flex">
	<button
		type="button"
		class="align-middle"
		aria-label={text}
		bind:this={button}
		onmouseenter={show}
		onmouseleave={hide}
		onfocus={show}
		onblur={hide}
	>
		<svg
			xmlns="http://www.w3.org/2000/svg"
			class="h-3.5 w-3.5 text-base-content/40 hover:text-base-content/70 transition-colors"
			fill="none"
			viewBox="0 0 24 24"
			stroke="currentColor"
		>
			<path
				stroke-linecap="round"
				stroke-linejoin="round"
				stroke-width="2"
				d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
			/>
		</svg>
	</button>
</span>

{#if visible}
	<!-- span, not div: InfoTip is used inside inline/table contexts where a div is invalid -->
	<span
		class="info-tip-bubble bg-neutral text-neutral-content text-xs rounded px-2 py-1"
		role="tooltip"
		style="left: {x}px; top: {y}px;"
	>
		{text}
	</span>
{/if}

<style>
	.info-tip-bubble {
		position: fixed;
		transform: translateX(-50%);
		max-width: 16rem;
		white-space: pre-wrap;
		text-align: left;
		z-index: 50;
		pointer-events: none;
	}
</style>
