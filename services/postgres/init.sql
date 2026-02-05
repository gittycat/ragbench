-- PostgreSQL initialization script for RAG system
-- Run by docker-entrypoint-initdb.d on first container start

-- Enable extensions (ParadeDB image includes pgvector and pg_search)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;
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

-- Document chunks with vectors
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_with_context TEXT,
    embedding vector(768),  -- nomic-embed-text dimension
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);

-- HNSW index for vector similarity search (cosine)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks
USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- BM25 index via pg_search (ParadeDB v2 API)
-- Uses English stemmer for better search relevance
CREATE INDEX idx_chunks_bm25 ON document_chunks
USING bm25 (id, content)
WITH (key_field='id');

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

-- Grant permissions to raguser (paranoid, usually owner has all)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO raguser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO raguser;
