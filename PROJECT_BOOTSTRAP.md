# Create Svelte 5 + Tailwind CSS 4 + DaisyUI Webapp

## CLI Commands

```bash
# 1. Create project (from /services directory)
cd /Users/bernard/dev/code/rag/ragbench/services
npx -y sv create webapp --template minimal --types ts --add tailwindcss
cd webapp

# 2. Install DaisyUI 5
npm install -D daisyui@latest

# 3. Install Tailwind Vite plugin (required for Tailwind v4)
npm install -D @tailwindcss/vite
```

---

## Configuration Files

### 1. `vite.config.ts`

**CRITICAL:** The `@tailwindcss/vite` plugin must be placed BEFORE `sveltekit()`.

```typescript
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [
    tailwindcss(),
    sveltekit()
  ],
  server: {
    host: '0.0.0.0',
    port: 5173
  }
});
```

### 2. `src/app.css`

```css
@import "tailwindcss";
@plugin "daisyui" {
  themes: light --default, dark --prefersdark;
}
```


## Key Implementation Notes

1. **Svelte 5 Runes**: Uses `$state()` for reactive state and `$props()` for component props
2. **SvelteKit Routing**: File-based routing with `+page.svelte` files
3. **Tailwind v4 + DaisyUI**: Uses new CSS-based plugin syntax (`@plugin "daisyui"`) instead of old JS config
4. **DaisyUI 5 Components**: For chat interface, use `chat`, `chat-start`, `chat-end`, `chat-bubble` classes
5. **Theme Support**: Manual light/dark toggle with persistence (see below)

---

## Theme Switching Implementation

The webapp supports light (`nord`) and dark (`dim`) themes with:
- **Cookie + localStorage persistence** - survives page refresh
- **SSR injection** - prevents flash of wrong theme (FOUC)
- **DaisyUI swap component** - animated sun/moon toggle

### Key Points

1. **Don't use DaisyUI's `theme-controller` class** - it conflicts with Svelte's reactive state when syncing SSR theme
2. **Manually set `data-theme`** in the persist function instead
3. **Dual storage** (localStorage + cookie) ensures both client navigation and SSR work
4. **Cookie read happens in `hooks.server.ts`** on every server request
