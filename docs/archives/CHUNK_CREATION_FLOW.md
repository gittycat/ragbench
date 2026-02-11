# Document → Chunks → PostgreSQL Flow

Complete trace of how documents are processed and chunks are stored in PostgreSQL.

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. UPLOAD ENDPOINT                                                          │
│    api/routes/documents.py:upload_documents()                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │ Save files to /tmp/shared                             │
        │ Generate batch_id and task_id (UUIDs)                 │
        │ Enqueue tasks via pgmq.send('documents', task_payload) │
        │ Create batch and task records in PostgreSQL:          │
        │   - job_batches table                                 │
        │   - job_tasks table                                   │
        └───────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. PGMQ WORKER                                                              │
│    infrastructure/tasks/pgmq_worker.py (separate process)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │ Poll pgmq queue: pgmq.read('documents', ...)          │
        │ Call process_document() from worker.py                │
        └───────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. WORKER PROCESSING                                                        │
│    infrastructure/tasks/worker.py:process_document_async()                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │ A. Extract metadata (hash, size, type)                │
        │    pipelines/ingestion.py:extract_file_metadata()     │
        └───────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │ B. Create Document record FIRST (foreign key needed)  │
        │    infrastructure/database/documents.py:               │
        │      create_document()                                │
        │                                                        │
        │    INSERT INTO documents (                            │
        │      id, file_name, file_type, file_path,             │
        │      file_size_bytes, file_hash, metadata             │
        │    ) VALUES (...)                                     │
        └───────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │ C. Get vector index                                   │
        │    infrastructure/search/vector_store.py:             │
        │      get_vector_index()                               │
        │                                                        │
        │    Returns VectorStoreIndex wrapping PGVectorStore    │
        │    (configured for table "document_chunks")           │
        └───────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │ D. Run ingestion pipeline                             │
        │    pipelines/ingestion.py:ingest_document()           │
        └───────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. INGESTION PIPELINE                                                       │
│    pipelines/ingestion.py:ingest_document()                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │                                                        │
        ▼                                                        ▼
┌──────────────────────┐                              ┌──────────────────────┐
│ STEP 1: Metadata     │                              │ STEP 2: Chunking     │
│ extract_file_metadata│                              │ chunk_document()     │
│ - Compute file hash  │                              │                      │
│ - Get file size      │                              │ Route by extension:  │
│ - Extract file type  │                              │                      │
└──────────────────────┘                              │ .txt/.md →           │
                                                       │   SentenceSplitter   │
                                                       │   (500 chunk_size,   │
                                                       │    50 overlap)       │
                                                       │                      │
                                                       │ .pdf/.docx/etc →     │
                                                       │   DoclingReader +    │
                                                       │   DoclingNodeParser  │
                                                       │                      │
                                                       │ Returns: List[       │
                                                       │   TextNode]          │
                                                       └──────────────────────┘
                                                                 │
                                                                 ▼
                                              ┌──────────────────────────────┐
                                              │ STEP 3: Contextual Retrieval │
                                              │ add_contextual_retrieval()   │
                                              │                              │
                                              │ If enabled:                  │
                                              │   For each chunk:            │
                                              │     - Take first 400 chars   │
                                              │     - Send to LLM for        │
                                              │       1-2 sentence context   │
                                              │     - Prepend context to     │
                                              │       chunk text             │
                                              │                              │
                                              │ Time: ~85% of processing     │
                                              └──────────────────────────────┘
                                                                 │
                                                                 ▼
                                              ┌──────────────────────────────┐
                                              │ STEP 4: Add Metadata         │
                                              │ add_document_metadata_to_    │
                                              │   chunks()                   │
                                              │                              │
                                              │ For each chunk:              │
                                              │   node.metadata.update({     │
                                              │     "document_id": doc_id,   │
                                              │     "chunk_index": i,        │
                                              │     "file_name": "...",      │
                                              │     "file_type": "...",      │
                                              │     "file_size": ...,        │
                                              │     "file_hash": "...",      │
                                              │     "uploaded_at": "..."     │
                                              │   })                         │
                                              │   node.id_ = "{doc_id}-      │
                                              │     chunk-{i}"               │
                                              └──────────────────────────────┘
                                                                 │
                                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: EMBEDDING & STORAGE                                                │
│ pipelines/ingestion.py:embed_and_index_chunks()                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │ For each chunk (TextNode):                            │
        │                                                        │
        │   index.insert_nodes([node])                          │
        │     │                                                  │
        │     ▼                                                  │
        │   VectorStoreIndex.insert_nodes()                     │
        │     │                                                  │
        │     ▼                                                  │
        │   PGVectorStore.add()                                 │
        │     │                                                  │
        │     ├─> Generate embedding via OllamaEmbedding        │
        │     │   (nomic-embed-text:latest, 768 dims)           │
        │     │                                                  │
        │     └─> Store in PostgreSQL                           │
        └───────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. POSTGRESQL STORAGE                                                       │
│    LlamaIndex PGVectorStore writes to data_document_chunks table            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │ INSERT INTO public.data_document_chunks (             │
        │   id,              -- UUID (node.id_)                 │
        │   embedding,       -- vector(768) from Ollama         │
        │   text,            -- TEXT (chunk content)            │
        │   metadata_,       -- JSONB (all metadata)            │
        │   node_id,         -- TEXT (same as id)               │
        │   ref_doc_id       -- TEXT (document_id)              │
        │ )                                                      │
        │                                                        │
        │ NOTE: LlamaIndex creates this table automatically     │
        │       with "data_" prefix even though we specify      │
        │       table_name="document_chunks"                    │
        └───────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │ BM25 Index Auto-Update                                │
        │                                                        │
        │ pg_search maintains idx_chunks_bm25 automatically     │
        │ when rows are inserted/updated/deleted                │
        │                                                        │
        │ No manual refresh needed (unlike old Tantivy BM25)    │
        └───────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │ HNSW Index Auto-Update                                │
        │                                                        │
        │ PostgreSQL HNSW index (idx_chunks_embedding)          │
        │ automatically updates on insert                       │
        │ Parameters: m=16, ef_construction=64                  │
        └───────────────────────────────────────────────────────┘

```

---

## Database Schema

### Tables Created

1. **`documents`** (init.sql) - Source file metadata
   - Created BEFORE chunk processing (foreign key constraint)
   - Stores: file_name, file_type, file_hash, uploaded_at, metadata

2. **`data_document_chunks`** (LlamaIndex auto-created) - Chunk storage
   - Created automatically by PGVectorStore
   - Stores: id, embedding (vector), text, metadata (JSONB), node_id, ref_doc_id
   - Has HNSW index for vector similarity
   - Has BM25 index (pg_search) for full-text search

3. **`document_chunks`** (init.sql) - Alternative schema (unused?)
   - Defined in models.py and init.sql
   - **NOTE**: This table exists but LlamaIndex uses `data_document_chunks` instead!

### Why Two Tables?

The codebase has some confusion:
- `init.sql` creates `document_chunks` table
- PGVectorStore is configured with `table_name="document_chunks"`
- **BUT** LlamaIndex adds `data_` prefix, creating `data_document_chunks`
- Query functions reference `data_document_chunks` (documents.py:79, 123)

---

## Key Files Reference

### Ingestion Pipeline
- `pipelines/ingestion.py` - Main ingestion logic
  - `chunk_document()` - Routes to DoclingReader or SentenceSplitter
  - `add_contextual_retrieval()` - LLM-based context generation
  - `embed_and_index_chunks()` - Embedding + PostgreSQL insert

### Infrastructure
- `infrastructure/search/vector_store.py` - PGVectorStore singleton
- `infrastructure/database/documents.py` - Document CRUD operations
- `infrastructure/tasks/worker.py` - Async document processing
- `infrastructure/tasks/pgmq_worker.py` - Queue consumer

### Storage
- `services/postgres/init.sql` - Database schema
- `infrastructure/database/models.py` - SQLAlchemy ORM models

---

## Performance Breakdown (Typical Document)

Based on logs in ingestion.py:

1. **Chunking**: ~5-15% of time
   - DoclingReader: slower but preserves structure
   - SentenceSplitter: faster for plain text

2. **Contextual Retrieval**: ~85% of time
   - 1 LLM call per chunk
   - ~2s per chunk average
   - Can be disabled via `ENABLE_CONTEXTUAL_RETRIEVAL=false`

3. **Embedding**: ~10-15% of time
   - Ollama nomic-embed-text
   - ~0.2-0.5s per chunk
   - Includes PostgreSQL insert

---

## Important Notes

1. **Foreign Key Constraint**: Document record MUST be created before chunks
   - See worker.py:87-98 (creates document first)
   - Chunks reference documents(id) via document_id

2. **BM25 Auto-Refresh**: pg_search handles index updates automatically
   - No manual refresh needed
   - Old Tantivy implementation required manual refresh

3. **Chunk IDs**: Format is `{document_id}-chunk-{index}`
   - Ensures uniqueness across all documents
   - Used for deduplication and updates

4. **Metadata Storage**: All metadata goes into JSONB field
   - Includes: file metadata + chunk metadata
   - Supports arbitrary nesting (no flattening needed)

5. **Table Name Confusion**: Watch out for `document_chunks` vs `data_document_chunks`
   - LlamaIndex uses `data_document_chunks`
   - Some code still references the unused `document_chunks` table
