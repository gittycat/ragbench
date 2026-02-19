# RAG Evaluation: Implementation Plan

## Prerequisites

- Read `docs/evals-03-codebase-state.md` before starting any step
- Each step is independently reviewable and includes at least one testable eval
- Order: Tier 2 first (proves pipeline works), then Tier 1 (adds generation isolation)
- Tier 3 (synthetic generation) is deferred — not in this plan

---

## Step 1 — Fix Retrieval Metric Bugs

**Scope:** `services/evals/evals/metrics/retrieval.py`, `services/evals/evals/metrics/performance.py`

**Changes:**

In `retrieval.py`, find the three early-return branches that fire when `question.gold_passages` is empty (one each in `RecallAtK.compute()`, `MRR.compute()`, `NDCG.compute()`). Change `return 1.0` to `return 0.0`.

In `performance.py`, guard latency/cost computation: if there are no successful queries, return `None` or skip adding the metric rather than computing a meaningless 0.0.

**Why first:** These bugs silently mask all retrieval failures. Without fixing them, metric output is untrustworthy and there is no way to tell if Tier 2 is actually working.

**How to test:**
```bash
cd services/evals
uv run pytest tests/test_rag_eval.py -v
```

Add unit tests that assert: a question with no gold passages and some retrieved chunks scores 0.0 on Recall@K (not 1.0).

**Acceptance:** All existing unit tests pass. New unit tests for the no-gold-passages case pass.

---

## Step 2 — Add EvalTier and DatasetConfig to Config

**Scope:** `services/evals/evals/config.py`

**Changes:**

Add `EvalTier` enum:
```python
class EvalTier(str, Enum):
    GENERATION = "generation"    # Tier 1: inject context, no ingestion
    END_TO_END  = "end_to_end"  # Tier 2: ingest docs, full pipeline
```

Add `tier` field to `EvalConfig` with default `EvalTier.END_TO_END`.

Add `cleanup_on_failure: bool = True` field to `EvalConfig`.

Add a mapping that declares which tiers each dataset supports:
```python
DATASET_TIER_SUPPORT: dict[DatasetName, list[EvalTier]] = {
    DatasetName.RAGBENCH:  [EvalTier.GENERATION, EvalTier.END_TO_END],
    DatasetName.SQUAD_V2:  [EvalTier.GENERATION],
    DatasetName.GOLDEN:    [EvalTier.GENERATION],
    DatasetName.QASPER:    [EvalTier.END_TO_END],
    DatasetName.HOTPOTQA:  [EvalTier.END_TO_END],
    DatasetName.MSMARCO:   [EvalTier.END_TO_END],
}
```

Add validation in `EvalConfig.__post_init__()` (or a `validate()` method): raise `ValueError` with a clear message if a dataset is configured for a tier it doesn't support.

**How to test:**
```bash
uv run pytest tests/ -v -k "config"
```

Add unit tests for the validation logic (bad tier + dataset combination raises ValueError).

**Acceptance:** Config loads correctly. Invalid tier/dataset combos raise clear errors.

---

## Step 3 — Add RAGClient Ingestion Methods

**Scope:** `services/evals/evals/runner.py` — `RAGClient` class only

**Changes:**

Add three methods to `RAGClient`:

```python
def upload_text_as_document(self, text: str, filename: str) -> str:
    """Upload plain text as a .txt file. Returns batch_id."""
    # Use httpx multipart: files=[("files", (filename, text.encode(), "text/plain"))]
    # POST /upload
    # Return response["batch_id"]

def wait_for_batch(self, batch_id: str, timeout: float = 300.0, poll_interval: float = 2.0) -> bool:
    """Poll GET /tasks/{batch_id}/status until all tasks complete or timeout.
    Returns True if all completed successfully, False if any failed or timed out."""
    # Poll every poll_interval seconds up to timeout
    # completed when total == completed
    # Return False if any task status == "failed" or timeout exceeded

def delete_document(self, document_id: str) -> bool:
    """DELETE /documents/{document_id}. Returns True on success."""
```

No changes to `EvaluationRunner` in this step.

Also update `client.query()` to accept an `include_chunks: bool = False` parameter, passing it in the POST body. This is needed by Step 4.

**How to test (integration):**
```bash
# Requires running RAG server (docker compose up rag-server task-worker)
# Run a quick smoke test from the host:
cd services/evals
uv run python -c "
from evals.runner import RAGClient
client = RAGClient('http://localhost:8001')
batch_id = client.upload_text_as_document('BM25 is a ranking function.', 'test_doc.txt')
print('batch_id:', batch_id)
ok = client.wait_for_batch(batch_id)
print('completed:', ok)
"
```

**Acceptance:** Can upload a text document, wait for it to be processed, and delete it via the client.

---

## Step 4 — Add Tier 2 Ingestion + Cleanup to EvaluationRunner

**Scope:** `services/evals/evals/runner.py` — `EvaluationRunner` class

**Changes:**

Add private method `_ingest_documents(questions: list[EvalQuestion]) -> dict[str, str]`:
1. Collect all unique `(doc_id, text)` pairs from `question.gold_passages` across all questions
2. For each unique doc_id: call `client.upload_text_as_document(text, f"{doc_id}.txt")`
3. Collect all batch_ids; call `client.wait_for_batch()` for each
4. Call `GET /documents` on the RAG server; find each uploaded doc by matching `metadata.filename` or similar field
5. Return `gold_doc_id → rag_doc_id` mapping

Add private method `_cleanup_documents(rag_doc_ids: list[str]) -> None`:
1. For each doc_id: call `client.delete_document(doc_id)`, log any failures
2. Never raise — cleanup must always complete

Modify `EvaluationRunner.run()`:
```
if config.tier == EvalTier.END_TO_END:
    doc_id_map = self._ingest_documents(all_questions)
    # patch gold_passages in all questions to use rag doc_ids
    for q in all_questions:
        for gp in q.gold_passages:
            gp.doc_id = doc_id_map.get(gp.doc_id, gp.doc_id)

try:
    [existing query loop — add include_chunks=True for END_TO_END]
finally:
    if config.tier == EvalTier.END_TO_END:
        self._cleanup_documents(list(doc_id_map.values()))
```

Pass `include_chunks=True` in the `client.query()` call when tier is `END_TO_END` (uses the parameter added in Step 3).

**How to test:**

```bash
# Run with small ragbench sample in Tier 2 mode
docker compose --profile eval run --rm evals eval \
  --tier end_to_end --datasets ragbench --samples 5 --no-judge
```

**Acceptance:** Retrieval metrics (Recall@K, MRR) are non-zero. Documents are deleted from the RAG system after the run. Cleanup runs even when the eval raises an exception mid-way.

---

## Step 5 — Add POST /query/with-context to rag-server

**Scope:** `services/rag_server/` — new endpoint, potentially a new route or added to existing `query.py`

**What the endpoint does:**
- Accepts `{ query: str, context_passages: [{ text: str, doc_id: str }], session_id?: str }`
- Builds a context string from the provided passages (same format the inference pipeline uses for retrieved chunks)
- Calls the LLM with the same system prompt and chat engine configuration as `POST /query`
- Bypasses the hybrid retrieval pipeline entirely
- Returns the same response shape as `POST /query` (answer, session_id, metrics; `sources` populated from the provided passages, not from retrieval)

**Implementation path:**

`query_rag()` in `pipelines/inference.py` uses `CondensePlusContextChatEngine.chat()`, which performs retrieval internally. It cannot be called with pre-injected context without significant coupling. Add a new standalone function `query_rag_with_context()` in the same file that:

1. Calls `reset_token_counter()`
2. Formats the provided passages into a context string using the same format `extract_sources()` would produce
3. Calls `Settings.llm.complete()` (or `chat()`) with the system prompt + formatted context + user query — use `get_system_prompt()` and `get_context_prompt()` from `infrastructure/llm/prompts.py`
4. Builds and returns the same dict shape as `query_rag()`: `{ answer, sources, session_id, citations, metrics }`
5. The `sources` list is built directly from the provided `context_passages`, not from retrieval

In `api/routes/query.py`, add a new `QueryWithContextRequest` Pydantic model and `POST /query/with-context` handler that calls `query_rag_with_context()` via `run_in_executor` (same pattern as the existing `/query` handler).

Add `QueryWithContextRequest` to `schemas/query.py`:
```python
class ContextPassage(BaseModel):
    text: str
    doc_id: str

class QueryWithContextRequest(BaseModel):
    query: str
    context_passages: list[ContextPassage]
    session_id: str | None = None
```

**How to test:**
```bash
curl -X POST http://localhost:8001/query/with-context \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is BM25?",
    "context_passages": [
      {"text": "BM25 is a ranking function used in information retrieval.", "doc_id": "test-doc-1"}
    ]
  }'
```

**Acceptance:** Returns a coherent answer derived from the provided context. The answer does not reference information outside the provided passages. Response time is significantly lower than `POST /query` (no retrieval latency).

---

## Step 6 — Add Tier 1 Path to Eval Service

**Scope:** `services/evals/evals/runner.py`

**Changes:**

Add to `RAGClient`:
```python
def query_with_context(self, question: str, passages: list[GoldPassage], session_id: str | None = None) -> dict:
    """POST /query/with-context with gold passages as context."""
    body = {
        "query": question,
        "context_passages": [{"text": p.text, "doc_id": p.doc_id} for p in passages],
        "session_id": session_id,
    }
    # POST to /query/with-context
```

Modify `EvaluationRunner.run()`: branch on tier:
- `EvalTier.GENERATION`: call `client.query_with_context(q.question, q.gold_passages)` — no ingestion, no cleanup
- `EvalTier.END_TO_END`: existing flow from Step 4

For Tier 1, retrieval metrics are not applicable (nothing was retrieved). In `_init_metrics()`, check `if self.config.tier == EvalTier.GENERATION: skip initializing retrieval metrics`. They will then be absent from the scorecard. The `_compute_weighted_score()` method should already handle missing groups by normalizing over only the groups that have results — verify this and add the normalization if not present.

**How to test:**
```bash
# Run ragbench in Tier 1 mode (fast — no ingestion)
docker compose --profile eval run --rm evals eval \
  --tier generation --datasets ragbench --samples 5 --no-judge
```

Then compare with Tier 2 run on the same 5 questions. Generation metrics should be meaningfully non-zero in both modes.

**Acceptance:** Tier 1 run completes without ingestion. Generation metrics (faithfulness, answer correctness, answer relevancy) are non-zero. Retrieval metrics are absent from the scorecard.

---

## Step 7 — CLI Tier Flag and Dataset Validation

**Scope:** `services/evals/evals/cli.py`

**Changes:**

Add `--tier` flag to the `eval` command:
```
--tier [generation|end_to_end]   # default: end_to_end
```

On startup, validate that all configured datasets support the selected tier (reuses the validation added in Step 2). If not, print a clear error listing which datasets are incompatible and which tiers they support.

For `EvalTier.END_TO_END`, add an explicit check that the RAG server's task-worker is reachable (or at least that the upload endpoint responds). If not, error early with a clear message instead of silently producing empty retrieval metrics.

Update `just test-eval` and `just test-eval-full` in the justfile to pass `--tier end_to_end`.

The justfile `test-eval` and `test-eval-full` targets currently hardcode `--datasets ragbench --samples 5` with no tier flag. Update them to add `--tier end_to_end`:
```
test-eval:
    docker compose --profile eval run --rm evals eval --tier end_to_end --datasets ragbench --samples 5
```

**How to test:**
```bash
# Should work
docker compose --profile eval run --rm evals eval --tier generation --datasets ragbench --samples 5

# Should fail clearly with incompatible dataset error
docker compose --profile eval run --rm evals eval --tier generation --datasets qasper --samples 5
```

**Acceptance:** `--tier` flag works. Incompatible dataset+tier combinations are caught before any network calls. `just test-eval` and `just test-eval-full` still pass.

---

## Step 8 — Add SQuAD v2 Abstention Eval (Tier 1)

**Scope:** Mostly configuration and validation — the squad_v2 loader already exists

**Changes:**

Verify the squad_v2 loader correctly sets `is_unanswerable=True` on unanswerable questions. If not, fix the loader.

Add an integration test that runs a small squad_v2 sample in Tier 1 mode and confirms:
- Abstention metrics are present in the scorecard
- `is_unanswerable=True` questions are correctly handled by the abstention metric

Update CLI help text and dataset listing to indicate squad_v2 is Tier 1 only and tests abstention.

**How to test:**
```bash
uv run python -m evals.cli eval --tier generation --datasets squad_v2 --samples 10 --no-judge
```

**Acceptance:** Abstention metrics (unanswerable accuracy, FPR, FNR) appear in the scorecard with non-trivial values (not all 0.0 or all 1.0).

---

## Implementation Notes for Sonnet

### Doc-ID Alignment (Tier 2 — Step 4)

When gold passages are uploaded as `.txt` files and processed by the task-worker, the RAG server assigns its own UUID as the document ID. The gold passages in `EvalQuestion` use dataset-specific IDs like `"covidqa:doc:0"`.

Recommended approach:
1. Sanitize the gold doc_id for use as a filename: replace `:` and spaces with `_`, append `.txt`
   e.g. `covidqa:doc:0` → `covidqa_doc_0.txt`
2. Upload each gold passage using that filename
3. After batch completes, call `GET /documents` — it returns `[{ "id": "<uuid>", "file_name": "...", ... }]`
4. Match each uploaded filename against the `file_name` field to find the rag_doc_id (the `id` field)
5. Build the `gold_doc_id → rag_doc_id` mapping

Note: `GET /documents` returns `file_name` at the top level of each document object, not inside a metadata subobject.

### Upload Format (Tier 2 — Step 3)

The RAG server's `/upload` endpoint accepts `multipart/form-data` with a `files[]` field. To upload a text string as a file using httpx:

```python
import io
files = [("files", (filename, io.BytesIO(text.encode()), "text/plain"))]
response = self._client.post("/upload", files=files)
```

### Context Injection Format (Tier 1 — Step 5)

When building the injected context for `POST /query/with-context`, format passages the same way the inference pipeline formats retrieved chunks. Check `pipelines/inference.py` for the exact format used when building the LLM context string.

### Cleanup Strategy (Tier 2 — Step 4)

The `finally` block in `run()` must catch and log all delete failures without re-raising. A single failed delete should not hide the eval results. Log `WARNING` with the doc_id for any document that could not be deleted. Consider writing failed-to-delete doc_ids to the run metadata for manual cleanup.

### Tier 1 Weight Redistribution (Step 6)

For Tier 1 runs, `retrieval` weight (default 15%) should be distributed proportionally to the remaining dimensions. Simplest approach: compute the weighted score only over the dimensions that have results, then normalize. Do not hardcode separate weight sets for each tier.
