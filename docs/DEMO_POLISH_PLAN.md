# Demo Polish Plan

Turn RAGBench into a polished, deploy-ready showcase project. This plan is self-contained:
implement it top-to-bottom without needing prior conversation context.

## Context & Goals

RAGBench is a **showcase project** demonstrating the author's ability to build an AI product with a
clean, coherent architecture and an up-to-date stack. It is deliberately **not** enterprise
production software — no monitoring, alerting, HA, or DR. Reducing complexity is a goal in itself.

Decisions already made (do not re-litigate):

- **Deployment**: single Linux VM + Docker Compose, as the existing `docker-compose.server.yml`
  overlay is designed for. Actual cloud provisioning is handled by the owner and is **out of scope**.
- **Hosted demo models**: cloud LLM APIs for inference (exercising the PII-masking feature);
  embeddings stay local (Ollama `nomic-embed-text` on the VM). Local dev defaults stay fully local.
- **Access**: the deployed demo is public behind a **simple shared-password login** on the webapp.
- **Scope**: full sweep — dead-code purge, config/CI correctness, login gate, docs overhaul, demo UX.

## Ground Rules for the Implementer

1. Work on a branch: `git checkout -b demo-polish`.
2. One commit per numbered task (or per phase for small tasks). Commit messages: short one-liners.
3. After every phase, run `just test-unit` and confirm it passes before moving on.
4. **Before deleting any dependency or file, run the grep given in the task** and confirm zero hits
   in source (ignore `.venv/`, `__pycache__/`, `node_modules/`, lock files). If you get hits, stop
   and leave that item in place with a note in the commit message.
5. Do not modify RAG pipeline logic (`pipelines/`, `infrastructure/search/`, `infrastructure/pii/`)
   except where a task explicitly says so.
6. Paths are relative to the repo root. Python service work happens in `services/rag_server` and
   `services/evals` (both use `uv`); webapp work in `services/webapp` (npm).

---

## Phase 1 — Dead Code & Dependency Purge

### 1.1 Delete unused Anthropic client module

`services/rag_server/app/anthropic_client.py` is dead code (nothing imports it; it also references
the outdated model `claude-3-5-sonnet-latest`).

- Verify: `grep -rn "anthropic_client\|call_anthropic" services/rag_server --include='*.py'`
  → only hits inside the file itself.
- Delete the file. **Keep `services/rag_server/app/settings.py`** — it is imported everywhere.
- If `services/rag_server/app/__init__.py` re-exports anything from `anthropic_client`, remove that
  export.

### 1.2 Remove unused Python dependencies (rag_server)

Edit `services/rag_server/pyproject.toml`. For each dependency below, run the verification grep
(from `services/rag_server/`, source dirs only: `api app core infrastructure pipelines schemas
scripts services tests main.py`); remove the dependency only if there are zero hits:

| Dependency | Verification grep |
|---|---|
| `streamlit` | `grep -rn "streamlit" --include='*.py' .` |
| `pgvector` | `grep -rn "pgvector" --include='*.py' .` |
| `alembic` | `grep -rn "alembic" --include='*.py' .` (env.py hits are deleted in this task) |
| `llama-index-vector-stores-postgres` | `grep -rn "vector_stores.postgres\|PGVectorStore" --include='*.py' .` |
| `llama-index-retrievers-bm25` | `grep -rn "llama_index.retrievers.bm25" --include='*.py' .` |
| `sse-starlette` | `grep -rn "sse_starlette\|EventSourceResponse" --include='*.py' .` |

Also delete the unused Alembic scaffolding (schema is managed by `services/postgres/init.sql`;
there are zero migration versions):

- `services/rag_server/alembic.ini`
- `services/rag_server/infrastructure/database/migrations/` (whole directory)
- Verify nothing else imports it: `grep -rn "database.migrations\|migrations.env" --include='*.py' .`

Do **not** remove: `llama-index-readers-file`, `llama-index-llms-openai-like` (vLLM provider in
`infrastructure/llm/factory.py`), `fpdf2` (test PDF generation) — verify with a grep if unsure.

Finish with: `cd services/rag_server && uv sync --group dev` (regenerates `uv.lock`), then
`just test-unit`.

### 1.3 Delete the dead benchmark subsystem

The `bench` tooling targets `evaluation.benchmark_cli` / `evaluation.datasets`, modules that no
longer exist in `rag_server` (evals moved to `services/evals`). Everything below is dead:

- Delete `docker-compose.bench.yml`.
- Delete `services/rag_server/scripts/benchmark_pipeline.py`. Check
  `services/rag_server/scripts/README.md`: delete it too if it only documents the benchmark script,
  otherwise remove the benchmark section.
- `justfile`: delete the whole `# Benchmarking` section (recipes `bench-setup`, `bench-up`,
  `bench-down`, `bench-logs`, `bench-load`, `bench-run`, `bench-quick`, `bench-datasets`,
  `bench-history`, `bench-stats`, `bench-dashboard`).
- `services/rag_server/pyproject.toml`: delete the `bench` dependency group; run `uv sync` again.
- Verify no remaining references: `grep -rn "bench" justfile docker-compose*.yml README.md DEVELOPMENT.md docs/*.md`
  (docs hits are cleaned in Phase 5; justfile/compose must be clean now).

### 1.4 Delete the stale root Makefile

`Makefile` duplicates the justfile and references files that no longer exist
(`tests/test_rag_eval.py`, `tests/evaluation`).

- Verify no docs instruct `make`: `grep -rn "make \|Makefile" README.md DEVELOPMENT.md CLAUDE.md AGENTS.md .forgejo/ docs/dev/`
  (fix any stragglers to use `just`).
- Delete `Makefile`.

### 1.5 Fix stale justfile recipes

- Delete the `migrate-sessions` recipe — `scripts/migrate_sessions.py` no longer exists.
- Leave `deploy`, `deploy-local`, `deploy-cloud`, `deploy-down` alone (deployment is owner-handled).

### 1.6 Delete stray npm artifacts in `services/`

`services/package.json` and `services/package-lock.json` are accidental `npm install` leftovers
(they even pin different major versions than the real webapp). The webapp's own manifest lives at
`services/webapp/package.json`.

- Delete `services/package.json`, `services/package-lock.json`, and `services/node_modules/` if present.
- Verify nothing references them: `grep -rn "services/package.json" . --include='*.yml' --include='*.md' --include='Dockerfile*' -r --exclude-dir=node_modules --exclude-dir=.venv`

### 1.7 Fix inaccurate startup log

`services/rag_server/main.py` (~line 138) logs
`"Hybrid search enabled (pg_search BM25 + pgvector)"` — wrong on both counts. Replace the message
with: `"Hybrid search enabled (pg_textsearch BM25 + ChromaDB vectors, RRF fusion)"`.

**Phase acceptance**: `just test-unit` green; `docker compose build rag-server` succeeds;
`git grep -l "streamlit\|alembic\|benchmark_pipeline"` returns no source files.

---

## Phase 2 — Configuration Correctness

### 2.1 Fix duplicate `pii:` key in config.yml

`config.yml` has **two** top-level `pii:` blocks (a 4-line stub around line 196 under
"PII Masking Settings", and the full block around line 284). YAML last-key-wins, so the first is
silently ignored.

- Delete the short stub block (the one containing only `enabled: false` with the
  "opt-in cloud generation tier — Task 2.3 stub" header).
- Keep the full block. Merge the useful comment from the stub
  ("When true, embedding provider must be local (enforced at startup)") into the surviving block's
  `enabled:` line.
- Verify: `python3 -c "import yaml,sys; d=yaml.safe_load(open('config.yml')); print(d['pii']['enabled'], len(d['pii']))"`
  → prints `False` and a count > 1 (full block, not the stub).
- Verify `just show-config` still runs.

### 2.2 Fix stale config.yml header comment

Line ~3 says "API keys are stored separately in secrets/.env". Actual mechanism: individual files
under `secrets/` mounted as Docker secrets (see `secrets/README.md`). Update the comment to:
`# API keys are stored as individual files under secrets/ (one file per key, see secrets/README.md)`.

### 2.3 Make the default config local-first

`active.inference` is currently `gpt5-mini` (cloud, requires an OpenAI key) while the README
quick start promises local models work out of the box. Restore truthfulness:

```yaml
active:
  inference: gemma3-4b    # local default — cloud models available below
  embedding: nomic-embed
  eval: gpt5-2            # eval judge is cloud-only; only needed when running evals
  reranker: minilm-l6
```

Add a short comment block above `active:` describing the **hosted demo profile**: set `inference`
to a cloud model (e.g. `claude-sonnet`), keep `embedding: nomic-embed` local, and set
`pii.enabled: true` so outbound cloud requests are PII-masked (the startup guardrail
`validate_privacy_posture()` requires local embeddings when PII masking is on — that combination
is the intended demo configuration).

**Phase acceptance**: `just show-config` reflects the local defaults; unit tests green.

---

## Phase 3 — CI Repair (`.forgejo/workflows/ci.yml`)

The `test-eval` job is broken: it references `tests/test_rag_eval.py` and `--group eval`, which no
longer exist in `services/rag_server` (evals moved to `services/evals`).

### 3.1 Fix the core test job

Change the pytest invocation to match the justfile:
`uv run pytest tests/ --ignore=tests/integration -v --tb=short`
(the current `--ignore=tests/evaluation --ignore=tests/test_rag_eval.py` flags point at
non-existent paths).

### 3.2 Replace the broken eval job with an evals-service unit test job

Delete the `test-eval` job and the `workflow_dispatch.inputs.run_eval` input. Add:

```yaml
  test-evals-service:
    name: Evals Service Tests
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/astral-sh/uv:python3.13-bookworm-slim
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      - name: Install dependencies
        working-directory: services/evals
        run: uv sync --group dev
      - name: Run unit tests
        working-directory: services/evals
        run: uv run pytest tests/test_api.py -v --tb=short
```

(`tests/test_rag_eval.py` needs a live RAG server + API key; it stays a manual `just test-eval`.)
Update the comment header at the top of the workflow file to match the new job list.

### 3.3 Add a webapp check job

```yaml
  test-webapp:
    name: Webapp Check
    runs-on: ubuntu-latest
    container:
      image: node:22-alpine
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      - name: Install dependencies
        working-directory: services/webapp
        run: npm ci
      - name: Type check
        working-directory: services/webapp
        run: npm run check
      - name: Build
        working-directory: services/webapp
        run: npm run build
```

### 3.4 Add the evals image to docker-build

In the `docker-build` job, add a step building `services/evals/Dockerfile`
(build context `.`, matching `docker-compose.yml`).

**Phase acceptance**: `npm run check` and `npm run build` pass locally in `services/webapp`;
YAML is valid (`python3 -c "import yaml; yaml.safe_load(open('.forgejo/workflows/ci.yml'))"`).

---

## Phase 4 — Webapp Login Gate

Add a shared-password login to the SvelteKit webapp. Mirrors the existing rag-server pattern
(`services/rag_server/infrastructure/auth.py`): **the gate activates only when a secret is
configured; local dev without the secret has no login.**

### 4.1 Server-side auth helper — `services/webapp/src/lib/server/auth.ts` (new)

```ts
import { createHmac, timingSafeEqual } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { env } from '$env/dynamic/private';

const COOKIE_NAME = 'ragbench_auth';
const COOKIE_PAYLOAD = 'ragbench-session-v1';

function loadPassword(): string | undefined {
  if (env.WEBAPP_ACCESS_PASSWORD_FILE) {
    try {
      return readFileSync(env.WEBAPP_ACCESS_PASSWORD_FILE, 'utf-8').trim() || undefined;
    } catch {
      return undefined;
    }
  }
  return env.WEBAPP_ACCESS_PASSWORD?.trim() || undefined;
}

const PASSWORD = loadPassword();

export const authEnabled = PASSWORD !== undefined;
export const cookieName = COOKIE_NAME;

function expectedToken(): string {
  return createHmac('sha256', PASSWORD!).update(COOKIE_PAYLOAD).digest('hex');
}

export function isValidSession(cookieValue: string | undefined): boolean {
  if (!authEnabled) return true;
  if (!cookieValue) return false;
  const expected = Buffer.from(expectedToken());
  const provided = Buffer.from(cookieValue);
  return provided.length === expected.length && timingSafeEqual(provided, expected);
}

export function verifyPassword(attempt: string): boolean {
  if (!authEnabled) return true;
  const a = Buffer.from(createHmac('sha256', 'attempt').update(attempt).digest());
  const b = Buffer.from(createHmac('sha256', 'attempt').update(PASSWORD!).digest());
  return timingSafeEqual(a, b);
}

export function sessionCookieValue(): string {
  return expectedToken();
}
```

### 4.2 Enforce in `services/webapp/src/hooks.server.ts`

At the **top** of the `handle` function (before the existing `/api/eval` and `/api` proxy blocks):

- Import `authEnabled`, `isValidSession`, `cookieName` from `$lib/server/auth`.
- Allowlist (no auth required): `pathname === '/login'`, `pathname.startsWith('/_app/')`,
  `pathname.startsWith('/favicon')`.
- If `authEnabled` and the `ragbench_auth` cookie fails `isValidSession`:
  - `/api/*` paths → return `new Response(JSON.stringify({ detail: 'Unauthorized' }), { status: 401, headers: { 'content-type': 'application/json' } })`.
  - Everything else → `redirect(303, '/login')` (import `redirect` from `@sveltejs/kit`).

This single choke point covers both pages and the API proxy — nothing reaches rag-server or the
evals service unauthenticated.

### 4.3 Login page — `services/webapp/src/routes/login/` (new)

`+page.server.ts`:
- `load`: if auth is disabled or session already valid → `redirect(303, '/')`.
- `actions.default`: read `password` from form data; on `verifyPassword` success, set the cookie
  (`cookies.set(cookieName, sessionCookieValue(), { path: '/', httpOnly: true, sameSite: 'lax',
  secure: url.protocol === 'https:', maxAge: 60 * 60 * 24 * 7 })`) and `redirect(303, '/')`.
  On failure: `await new Promise(r => setTimeout(r, 500))` (brute-force damper), then
  `return fail(400, { incorrect: true })`.

`+page.svelte`:
- Centered DaisyUI card matching the existing theme (the app uses DaisyUI 5 components and the
  theme mechanism in `src/lib/themes.ts`): app name/logo (`static/` has the logo), one password
  input (`input input-bordered`), submit button (`btn btn-primary`), and an error alert
  (`alert alert-error`) when `form?.incorrect`. Use a plain `<form method="POST">` with
  `use:enhance`. Keep it minimal and clean — this page is a first impression.

### 4.4 Wire the secret in `docker-compose.server.yml`

In the `webapp` service of the overlay add:

```yaml
    secrets:
      - RAG_SERVER_AUTH_TOKEN
      - WEBAPP_ACCESS_PASSWORD
    environment:
      - RAG_SERVER_AUTH_TOKEN_FILE=/run/secrets/RAG_SERVER_AUTH_TOKEN
      - WEBAPP_ACCESS_PASSWORD_FILE=/run/secrets/WEBAPP_ACCESS_PASSWORD
```

and to the overlay's top-level `secrets:` block:

```yaml
  WEBAPP_ACCESS_PASSWORD:
    file: secrets/WEBAPP_ACCESS_PASSWORD
```

Update `secrets/README.md` to document the new file (one line: shared password for the demo login).

### 4.5 Verification

- `cd services/webapp && npm run check` — clean.
- `WEBAPP_ACCESS_PASSWORD=testpw npm run dev`:
  - `/` redirects to `/login`; wrong password shows the error; correct password lands on `/`.
  - `curl -i localhost:5173/api/documents` (no cookie) → 401 — this must not depend on rag-server
    being up (the 401 fires before the proxy).
- Without the env var, `npm run dev` → no login page, app behaves exactly as before.
- `docker compose -f docker-compose.yml -f docker-compose.server.yml config -q` — valid.

---

## Phase 5 — Documentation Overhaul

### 5.1 Rewrite README.md (merge OVERVIEW.md into it)

`OVERVIEW.md` (repo root, currently untracked) contains polished marketing-quality copy that
overlaps the README. Produce **one** outstanding README and delete `OVERVIEW.md`. Structure:

1. **Title + tagline** — reuse OVERVIEW.md's opening ("A private, self-hostable RAG assistant …").
2. **Screenshot placeholder** — `![RAGBench chat](docs/images/chat.png)` and
   `![Analytics dashboard](docs/images/analytics.png)`. Create `docs/images/.gitkeep`. The owner
   captures real screenshots later (manual step — leave an HTML comment noting this).
3. **Why RAGBench** — from OVERVIEW.md (privacy + observability gap).
4. **Key features** — OVERVIEW.md's four feature sections, lightly trimmed.
5. **Architecture** — new Mermaid diagram:

    ```mermaid
    graph LR
        U[Browser] --> C[Caddy · HTTPS]
        C --> W[SvelteKit webapp :8000]
        W -->|/api proxy + bearer token| R[rag-server · FastAPI :8001]
        W -->|/api/eval proxy| E[evals service · FastAPI :8002]
        R --> P[(PostgreSQL 17<br/>pg_textsearch BM25)]
        R --> V[(ChromaDB<br/>vectors)]
        R --> O[Ollama local /<br/>cloud LLM APIs]
        T[task-worker] --> P
        T --> V
        T --> O
        E --> R
    ```

    Caption: Caddy and the login gate are the server-tier overlay (`docker-compose.server.yml`);
    local dev exposes the webapp directly. Internal services live on an isolated Docker network.
6. **Tech stack table** — from OVERVIEW.md.
7. **Quick Start** — current README steps (Ollama, models, `just init`, `docker compose up -d`),
   plus the new `just secrets-init` (Phase 6) replacing the manual "create files under secrets/"
   instruction, and `just demo-seed` for instant content.
8. **Configuration** — `config.yml` `active:` snippet; a short "Hosted demo profile" subsection:
   cloud inference + local embeddings + `pii.enabled: true` + `docker-compose.server.yml` overlay
   (Caddy TLS, bearer token, login password).
9. **Evaluation** — the 5 headline metrics and `just test-eval` pointer to `docs/EVALS_README.md`.
10. **Project scope** — keep and sharpen the existing disclaimer: this is a showcase/reference
    project focused on clean architecture and RAG know-how; monitoring, alerting, HA, and DR are
    deliberately out of scope.
11. **Development** — link to `DEVELOPMENT.md`. Drop the current self-deprecating "dumping ground"
    sentence.
12. **License**.

Delete `OVERVIEW.md` after the merge and `git add README.md docs/images/`.

### 5.2 Move FRONT_END.md into docs/dev/

- `git mv FRONT_END.md docs/dev/frontend.md`.
- While editing: remove the `@fnando/sparkline` entry from "Key Libraries" (not in package.json,
  not used in src — charts use chart.js) and give chart.js + chartjs-plugin-annotation a row.
- Update every reference: `grep -rn "FRONT_END" README.md DEVELOPMENT.md CLAUDE.md AGENTS.md docs/ services/ --include='*.md'`
  → fix all hits (at minimum `DEVELOPMENT.md` line ~5 and its Documentation Index table).
- Add a `frontend.md` row to `docs/dev/INDEX.md`.

### 5.3 Delete stale docs/TODO.md

Its main item (parallelize contextual retrieval) is already implemented
(`contextual_concurrency: 8` in config.yml; `asyncio.gather` + semaphore in
`pipelines/ingestion.py` — verify with a quick grep). Move the one still-open idea (cache
contextual prefixes by chunk hash) into `docs/ROADMAP.md` under Future Considerations, then delete
`docs/TODO.md` and remove its row from DEVELOPMENT.md's Documentation Index.

### 5.4 Refresh docs/ROADMAP.md

- Delete the "Fix or Remove Broken test_hybrid_search.py" section — the file no longer exists
  (verify: `ls services/rag_server/tests/integration/ | grep hybrid` → nothing).
- Skim the remaining "Testing & CI Improvements" items and delete any others that reference
  files/paths that no longer exist.

### 5.5 Update CLAUDE.md (project instructions are stale)

- **Evaluation section**: `evaluation/` no longer exists under rag_server. Replace
  `cd services/rag_server && uv sync --group eval` / `python -m evaluation.cli eval` with the real
  commands: `just test-eval` (smoke) / `just test-eval-full`, CLI = `services/evals`
  (`docker compose exec evals .venv/bin/python -m evals.cli …`).
- **Key Files**: replace the `evaluation/ — DeepEval framework…` bullet with
  `services/evals/ — standalone eval service (API :8002 + CLI), datasets, LLM judges`.
- **Architecture list**: add `webapp` (port 8000) and `evals` (port 8002) service entries.
- **Testing section**: eval tests live in `services/evals/tests/`, not rag_server.
- Fix any other references failing this check:
  `grep -n "evaluation\.\|evaluation/" CLAUDE.md`.

### 5.6 Update DEVELOPMENT.md

Reconcile the Documentation Index table with all moves/deletions from this phase
(TODO.md gone, FRONT_END.md → docs/dev/frontend.md). Verify every link in the table resolves:
each `[x](path)` target must exist.

**Phase acceptance**: `grep -rn "OVERVIEW.md\|FRONT_END.md\|TODO.md" README.md DEVELOPMENT.md CLAUDE.md AGENTS.md docs/dev/` → no stale hits; every doc link in README/DEVELOPMENT resolves.

---

## Phase 6 — Demo Experience

### 6.1 `just secrets-init` recipe

New justfile recipe that makes first-run setup one command. Never overwrite existing files:

```just
# Generate required secret files (skips any that already exist)
secrets-init:
    @mkdir -p secrets
    @[ -f secrets/POSTGRES_SUPERUSER ] || printf 'postgres' > secrets/POSTGRES_SUPERUSER
    @[ -f secrets/POSTGRES_SUPERPASSWORD ] || openssl rand -hex 16 | tr -d '\n' > secrets/POSTGRES_SUPERPASSWORD
    @[ -f secrets/RAG_SERVER_DB_USER ] || printf 'rag_server' > secrets/RAG_SERVER_DB_USER
    @[ -f secrets/RAG_SERVER_DB_PASSWORD ] || openssl rand -hex 16 | tr -d '\n' > secrets/RAG_SERVER_DB_PASSWORD
    @[ -f secrets/RAG_SERVER_AUTH_TOKEN ] || openssl rand -hex 32 | tr -d '\n' > secrets/RAG_SERVER_AUTH_TOKEN
    @[ -f secrets/WEBAPP_ACCESS_PASSWORD ] || openssl rand -hex 8 | tr -d '\n' > secrets/WEBAPP_ACCESS_PASSWORD
    @[ -f secrets/OPENAI_API_KEY ] || touch secrets/OPENAI_API_KEY
    @[ -f secrets/ANTHROPIC_API_KEY ] || touch secrets/ANTHROPIC_API_KEY
    @echo "Secrets ready under secrets/ — add real API keys to use cloud models."
```

Empty API-key files are fine: startup treats empty as "no key" and only warns. Reference this
recipe from the README Quick Start (replaces manual secret-file creation) and from
`secrets/README.md`.

### 6.2 `just demo-seed` — self-referential demo corpus

Seed the system with the project's own documentation so a visitor can immediately ask the app
about itself ("How does hybrid search work?", "What is contextual retrieval?").

- First, confirm the multipart field name of `POST /upload` in
  `services/rag_server/api/routes/documents.py` (~line 138) — use exactly that name in the curl
  calls below (shown here as `files`).
- New justfile recipe:

```just
# Upload the project's own docs as a demo corpus (requires running stack)
demo-seed: docker-up
    @echo "Seeding demo corpus..."
    curl -sf -X POST http://localhost:8001/upload \
      -F "files=@README.md" \
      -F "files=@docs/dev/architecture.md" \
      -F "files=@docs/dev/pii-masking.md" \
      -F "files=@docs/dev/eval-framework.md" \
      -F "files=@sample_documents/docker_guide.md" \
      -F "files=@sample_documents/machine_learning.txt" \
      -F "files=@sample_documents/python_basics.txt" | python3 -m json.tool
    @echo "Ingestion runs async — watch progress in the webapp Documents page."
```

  If the server-tier bearer token is active, the recipe still works locally (token unset).
  Adjust flags to the actual API contract if the endpoint differs (e.g. returns `batch_id` — then
  print it and mention `GET /tasks/{batch_id}/status`).
- Add a "Try asking" list of 3–4 suggested questions to the README Quick Start, matching the
  seeded corpus.

### 6.3 Screenshots (manual — owner)

Leave the README image placeholders from 5.1. Add a checklist item at the bottom of this plan's PR
description: capture `docs/images/chat.png` (chat with a cited answer) and
`docs/images/analytics.png` (analytics dashboard) at ~1400px wide, light theme.

---

## Phase 7 — Final Verification

Run in order; all must pass:

1. `just test-unit`
2. `cd services/evals && uv run pytest tests/test_api.py -v`
3. `cd services/webapp && npm run check && npm run build`
4. Compose file validity:
   - `docker compose config -q`
   - `docker compose -f docker-compose.yml -f docker-compose.server.yml config -q`
   - `docker compose -f docker-compose.yml -f docker-compose.cloud.yml config -q`
5. Full stack smoke test: `just secrets-init && just init && docker compose up -d --build`, then
   - `curl -sf localhost:8001/health`
   - `curl -sf localhost:8002/health`
   - `curl -sfI localhost:8000` (webapp responds; no login locally)
   - `just demo-seed`, wait for ingestion, ask one question through the webapp chat.
6. `just test-integration` (requires the stack from step 5).
7. `git status` — clean tree, no stray untracked files.

---

## Out of Scope (do not do)

- Cloud provisioning / VM setup / DNS / TLS certificates — owner-handled.
- Monitoring, alerting, HA, DR — deliberately excluded from this project.
- Multi-user auth, accounts, roles — the shared password is the intended scope.
- Refactoring `pipelines/inference.py` (1,250 lines) — noted as a possible follow-up, too risky
  for this pass.
- De-duplicating `infrastructure/config/` between rag_server and evals — intentional
  service-isolation duplication.
- `docker-compose.cloud.yml` and `docker-compose.ci.yml` — leave as-is (deployment and CI hosting
  are owner-managed).
