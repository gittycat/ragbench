# Integration Tests — Status

## Completed

✅ **Removed test-runner service** — docker-compose.yml no longer has duplicated config
✅ **Updated justfile** — `test-integration` uses `run` (CI), `test-integration-local` uses `exec` (fast)
✅ **Added OLLAMA_URL env var** — rag-server and task-worker now use `http://host.docker.internal:11434`
✅ **Fixed conftest.py** — Added DB pool cleanup to avoid event loop conflicts
✅ **Updated infrastructure tests** — Changed HNSW vector index check to BM25 index (post-ChromaDB migration)

## Test Status (10/32 passing)

**Passing (10):**
- ✅ test_infrastructure.py — All infrastructure/schema tests pass

**Failing (12):**
- ❌ test_api_documents.py — test_check_duplicates_after_upload
- ❌ test_async_upload.py — 5 upload/task tests
- ❌ test_full_upload_pipeline.py — 3 pipeline tests
- ❌ test_pipeline.py — 3 deletion tests

**Errors (10):**
- ⚠️ test_async_upload.py — test_progress_tracking_accuracy
- ⚠️ test_chat_sessions.py — 4 chat session tests
- ⚠️ test_pipeline.py — 5 text file pipeline tests

## Remaining Issues

1. **Upload/task tests** — Likely timing/async issues with task worker
2. **Chat session tests** — Possible state pollution or RAG server connection issues
3. **Pipeline tests** — File processing or cleanup problems

## Next Steps

- Investigate upload task timing issues (worker claiming tasks via SKIP LOCKED)
- Debug chat session test failures (ERROR at setup suggests fixture issues)
- Fix pipeline test cleanup/state pollution
- Delete this document once all tests pass
