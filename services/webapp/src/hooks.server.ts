import type { Handle } from '@sveltejs/kit';
import { isValidTheme } from '$lib/themes';
import { env } from '$env/dynamic/private';

const RAG_SERVER_URL = env.RAG_SERVER_URL || 'http://localhost:8001';
const EVALS_SERVICE_URL = env.EVALS_SERVICE_URL || 'http://localhost:8002';

function proxyRequest(event: Parameters<Handle>[0]['event'], targetBaseUrl: string, stripPrefix: string) {
	const targetPath = event.url.pathname.replace(new RegExp(`^${stripPrefix}`), '');
	const targetUrl = `${targetBaseUrl}${targetPath}${event.url.search}`;

	const headers = new Headers(event.request.headers);
	headers.delete('host');

	return fetch(targetUrl, {
		method: event.request.method,
		headers,
		body: event.request.method !== 'GET' && event.request.method !== 'HEAD'
			? event.request.body
			: undefined,
		// @ts-expect-error - duplex is needed for streaming request bodies
		duplex: 'half'
	});
}

export const handle: Handle = async ({ event, resolve }) => {
	// Proxy /api/eval/* to eval service
	if (event.url.pathname.startsWith('/api/eval/') || event.url.pathname === '/api/eval') {
		const response = await proxyRequest(event, EVALS_SERVICE_URL, '/api');
		return new Response(response.body, {
			status: response.status,
			statusText: response.statusText,
			headers: response.headers
		});
	}

	// Proxy all other /api/* to RAG server
	if (event.url.pathname.startsWith('/api/')) {
		const response = await proxyRequest(event, RAG_SERVER_URL, '/api');
		return new Response(response.body, {
			status: response.status,
			statusText: response.statusText,
			headers: response.headers
		});
	}

	const theme = event.cookies.get('theme');

	// If no valid theme cookie, render with default (let CSS handle it)
	if (!theme || !isValidTheme(theme)) {
		return await resolve(event);
	}

	// Inject theme into HTML before sending to client (prevents FOUC)
	return await resolve(event, {
		transformPageChunk: ({ html }) => {
			return html.replace('data-theme=""', `data-theme="${theme}"`);
		}
	});
};
