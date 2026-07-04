import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	server: {
		host: '0.0.0.0',
		port: 5173,
		proxy: {
			// More specific rule first: /api/eval/* goes to the evals service
			'/api/eval': {
				target: 'http://localhost:8002',
				changeOrigin: true,
				rewrite: (path) => path.replace(/^\/api/, '')
			},
			'/api': {
				target: 'http://localhost:8001',
				changeOrigin: true,
				rewrite: (path) => path.replace(/^\/api/, '')
			}
		}
	}
});
