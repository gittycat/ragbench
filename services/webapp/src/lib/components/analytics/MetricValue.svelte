<script lang="ts">
	import { thresholdColorClass } from '$lib/utils/thresholds';

	interface Props {
		metricName: string;
		value: number | null | undefined;
		format?: 'percent' | 'decimal' | 'ms' | 'seconds' | 'usd' | 'int' | 'raw';
		decimals?: number;
	}

	let { metricName, value, format = 'percent', decimals = 1 }: Props = $props();

	let colorClass = $derived(thresholdColorClass(metricName, value));

	let display = $derived.by(() => {
		if (value === null || value === undefined) return '—';
		switch (format) {
			case 'percent':
				return `${(value * 100).toFixed(decimals)}%`;
			case 'decimal':
				return value.toFixed(decimals);
			case 'ms':
				return `${value.toFixed(0)}ms`;
			case 'seconds':
				return `${value.toFixed(3)}s`;
			case 'usd':
				return `$${value.toFixed(4)}`;
			case 'int':
				return value.toLocaleString();
			default:
				return String(value);
		}
	});
</script>

<span class="font-mono tabular-nums {colorClass}">{display}</span>
