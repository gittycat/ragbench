/**
 * Svelte action for Chart.js integration + theme-derived chart colors.
 * Colors are resolved from DaisyUI CSS variables at config-build time; config
 * builders read themeSignal() (theme.svelte.ts) so charts rebuild on theme change.
 */
import {
	Chart,
	CategoryScale,
	LinearScale,
	BarElement,
	BarController,
	LineElement,
	LineController,
	PointElement,
	Title,
	Tooltip,
	Legend,
	type ChartConfiguration,
	type ScaleOptionsByType,
	type TooltipOptions
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';

// Register Chart.js components (bar + line charts)
Chart.register(
	CategoryScale,
	LinearScale,
	BarElement,
	BarController,
	LineElement,
	LineController,
	PointElement,
	Title,
	Tooltip,
	Legend,
	annotationPlugin
);

/**
 * Svelte action to create and manage a Chart.js instance
 * Usage: <canvas use:chartAction={chartConfig}></canvas>
 */
export function chartAction(
	canvas: HTMLCanvasElement,
	config: ChartConfiguration
): { update: (newConfig: ChartConfiguration) => void; destroy: () => void } {
	let chart = new Chart(canvas, config);

	return {
		update(newConfig: ChartConfiguration) {
			// Update data and options without recreating the chart
			if (newConfig.data) {
				chart.data = newConfig.data;
			}
			if (newConfig.options) {
				chart.options = newConfig.options;
			}
			chart.update('none'); // Skip animations on update for smoother experience
		},
		destroy() {
			chart.destroy();
		}
	};
}

export const CHART_FONT_FAMILY =
	'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace';

// 1x1 scratch canvas: lets the browser parse any CSS color (DaisyUI vars are
// oklch) and hands back concrete RGB we can attach an alpha to.
let scratch: CanvasRenderingContext2D | null = null;

/** Resolve a DaisyUI color token (e.g. 'primary', 'base-content') to an rgba() string. */
export function themeColor(token: string, alpha = 1): string {
	if (typeof document === 'undefined') return `rgba(128, 128, 128, ${alpha})`;
	const css = getComputedStyle(document.documentElement)
		.getPropertyValue(`--color-${token}`)
		.trim();
	if (!css) return `rgba(128, 128, 128, ${alpha})`;
	scratch ??= document.createElement('canvas').getContext('2d', { willReadFrequently: true });
	if (!scratch) return css;
	scratch.clearRect(0, 0, 1, 1);
	scratch.fillStyle = css;
	scratch.fillRect(0, 0, 1, 1);
	const [r, g, b] = scratch.getImageData(0, 0, 1, 1).data;
	return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** Recessive ink/grid colors for chart scaffolding, derived from the active theme. */
export function chartInk() {
	return {
		ink: themeColor('base-content', 0.7),
		inkMuted: themeColor('base-content', 0.45),
		grid: themeColor('base-content', 0.08)
	};
}

/**
 * Series colors for up to `count` datasets: an ordered lightness ramp of the
 * theme's primary hue (fill alpha steps). Single-hue + big lightness steps keeps
 * run identity legible under CVD and works in both nord/dim without a fixed palette.
 */
export function getChartColors(count: number): { background: string[]; border: string[] } {
	const fillAlphas = [0.9, 0.62, 0.4, 0.22];
	const background: string[] = [];
	const border: string[] = [];
	for (let i = 0; i < count; i++) {
		const a = fillAlphas[i % fillAlphas.length];
		background.push(themeColor('primary', a));
		border.push(themeColor('primary', Math.min(1, a + 0.1)));
	}
	return { background, border };
}

/** Dense monospace scale options (recessive grid, muted ticks) for a linear/category axis. */
export function denseScale(
	overrides: Record<string, unknown> = {}
): Partial<ScaleOptionsByType<'linear'>> {
	const { inkMuted, grid } = chartInk();
	return {
		grid: { color: grid },
		border: { color: grid },
		ticks: {
			color: inkMuted,
			font: { family: CHART_FONT_FAMILY, size: 10 }
		},
		...overrides
	} as Partial<ScaleOptionsByType<'linear'>>;
}

/** Theme-surface tooltip styling (Chart.js default is a hardcoded dark box). */
export function denseTooltip(): Partial<TooltipOptions> {
	const { ink, grid } = chartInk();
	return {
		backgroundColor: themeColor('base-100', 0.95),
		titleColor: ink,
		bodyColor: ink,
		borderColor: grid,
		borderWidth: 1,
		titleFont: { family: CHART_FONT_FAMILY, size: 10 },
		bodyFont: { family: CHART_FONT_FAMILY, size: 10 },
		padding: 6,
		cornerRadius: 2,
		displayColors: false
	} as Partial<TooltipOptions>;
}
