# RAG Server API Reference

Base URL: `http://localhost:8001`

## Health & Configuration

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/config` | System configuration (max_upload_size_mb) |
| GET | `/models/info` | Model names and settings |

## Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/documents` | List documents (supports sorting) |
| POST | `/documents/check-duplicates` | Check file hashes for duplicates |
| POST | `/upload` | Upload files (returns batch_id) |
| GET | `/tasks/{batch_id}/status` | Upload progress |
| DELETE | `/documents/{document_id}` | Delete document |
| GET | `/documents/{document_id}/download` | Download original file |

**Sorting Parameters:** `sort_by` (name, chunks, uploaded_at), `sort_order` (asc, desc)

## Query & Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | RAG query (non-streaming) |
| POST | `/query/stream` | RAG query (SSE streaming) |
| GET | `/chat/history/{session_id}` | Get conversation history |
| POST | `/chat/clear` | Clear session history |

**Query Request:**
```json
{
  "query": "What is...",
  "session_id": "uuid-optional",
  "is_temporary": false
}
```

**Streaming Events:** `token`, `sources`, `done`, `error`

## Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/chat/sessions` | List sessions (paginated) |
| GET | `/chat/sessions/{session_id}` | Get session metadata |
| POST | `/chat/sessions/new` | Create new session |
| DELETE | `/chat/sessions/{session_id}` | Delete session |
| POST | `/chat/sessions/{session_id}/archive` | Archive session |

## Metrics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/metrics/system` | Complete system overview |
| GET | `/metrics/models` | Detailed model info |
| GET | `/metrics/retrieval` | Retrieval pipeline config |
| GET | `/metrics/evaluation/definitions` | Metric definitions |
| GET | `/metrics/evaluation/history` | Past evaluation runs |
| GET | `/metrics/evaluation/summary` | Latest evaluation with trends |
| GET | `/metrics/evaluation/{run_id}` | Get specific evaluation run |
| DELETE | `/metrics/evaluation/{run_id}` | Delete evaluation run |
| GET | `/metrics/baseline` | Get golden baseline |
| POST | `/metrics/baseline/{run_id}` | Set golden baseline |
| DELETE | `/metrics/baseline` | Clear golden baseline |
| GET | `/metrics/compare/{run_a}/{run_b}` | Compare two evaluation runs |
| GET | `/metrics/compare-to-baseline/{run_id}` | Compare run to baseline |
| POST | `/metrics/recommend` | Get config recommendation |
