-- PostgreSQL initialization script for RAG system
-- Run by docker-entrypoint-initdb.d on first container start

-- Extensions: pg_textsearch for BM25 search, pgmq for message queue
CREATE EXTENSION IF NOT EXISTS pg_textsearch;
CREATE EXTENSION IF NOT EXISTS pgmq;

-- Documents table (source files)
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_path TEXT,
    file_size_bytes BIGINT,
    file_hash VARCHAR(64) UNIQUE,
    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Document chunks (text content for BM25 search; vectors stored in ChromaDB)
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_with_context TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);

-- BM25 index via pg_textsearch (Timescale) for ranked full-text search
CREATE INDEX idx_chunks_bm25 ON document_chunks
USING bm25 (content) WITH (text_config='english');

-- Chat sessions
CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY,
    title VARCHAR(255) DEFAULT 'New Chat',
    llm_model VARCHAR(100),
    search_type VARCHAR(20),
    is_archived BOOLEAN DEFAULT FALSE,
    is_temporary BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chat messages
CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);

-- Job batches (progress tracking)
CREATE TABLE IF NOT EXISTS job_batches (
    id UUID PRIMARY KEY,
    total_tasks INTEGER NOT NULL,
    completed_tasks INTEGER DEFAULT 0,
    status VARCHAR(30) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Job tasks
CREATE TABLE IF NOT EXISTS job_tasks (
    id UUID PRIMARY KEY,
    batch_id UUID NOT NULL REFERENCES job_batches(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    status VARCHAR(30) DEFAULT 'pending',
    total_chunks INTEGER DEFAULT 0,
    completed_chunks INTEGER DEFAULT 0,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_batch ON job_tasks(batch_id);

-- Create pgmq queue for document processing
SELECT pgmq.create('documents');

-- Grants are handled in 02-grants.sh using secrets-backed roles.
