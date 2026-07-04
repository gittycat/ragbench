// Reactive DaisyUI theme signal. Chart config builders call themeSignal() so
// their $derived/template expressions re-run (and charts re-render with fresh
// theme-resolved colors) when ThemeToggle flips data-theme on <html>.
import { browser } from '$app/environment';

let current = $state('');

if (browser) {
	current = document.documentElement.getAttribute('data-theme') ?? '';
	new MutationObserver(() => {
		current = document.documentElement.getAttribute('data-theme') ?? '';
	}).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
}

export function themeSignal(): string {
	return current;
}
