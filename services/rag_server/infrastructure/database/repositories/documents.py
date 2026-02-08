"""Document repository for PostgreSQL storage."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select, func, literal_column, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import Document, DocumentChunk
from infrastructure.database.repositories.base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    """Repository for document and chunk operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Document)

    async def get_by_hash(self, file_hash: str) -> Document | None:
        """Get document by file hash (for duplicate detection)."""
        result = await self.session.execute(
            select(Document).where(Document.file_hash == file_hash)
        )
        return result.scalar_one_or_none()

    async def check_duplicates(self, file_hashes: list[str]) -> dict[str, str]:
        """
        Check which files already exist by hash.
        Returns dict of {hash: document_id} for existing documents.
        """
        if not file_hashes:
            return {}

        result = await self.session.execute(
            select(Document.file_hash, Document.id).where(
                Document.file_hash.in_(file_hashes)
            )
        )
        return {row.file_hash: str(row.id) for row in result.all()}

    async def create_document(
        self,
        file_name: str,
        file_type: str,
        file_path: str | None = None,
        file_size_bytes: int | None = None,
        file_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
        document_id: UUID | None = None,
    ) -> Document:
        """Create a new document record with optional custom ID."""
        doc = Document(
            file_name=file_name,
            file_type=file_type,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            file_hash=file_hash,
            metadata_=metadata or {},
        )
        if document_id:
            doc.id = document_id
        return await self.add(doc)

    async def list_documents(
        self,
        sort_by: str = "uploaded_at",
        sort_order: str = "desc",
    ) -> list[dict[str, Any]]:
        """
        List all documents with chunk counts.
        Returns list of dicts with document info.
        """
        # Count chunks from LlamaIndex's data_document_chunks table
        # (PGVectorStore uses "data_" prefix: table_name → data_{table_name})
        subquery = (
            select(
                literal_column("(metadata_->>'document_id')::uuid").label("doc_id"),
                func.count().label("chunk_count"),
            )
            .select_from(text("public.data_document_chunks"))
            .group_by(literal_column("(metadata_->>'document_id')::uuid"))
            .subquery()
        )

        query = select(
            Document,
            func.coalesce(subquery.c.chunk_count, 0).label("chunks"),
        ).outerjoin(subquery, Document.id == subquery.c.doc_id)

        # Apply sorting
        sort_column = getattr(Document, sort_by, Document.uploaded_at)
        if sort_order == "desc":
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())

        result = await self.session.execute(query)
        documents = []
        for row in result.all():
            doc = row[0]
            chunks = row[1]
            documents.append({
                "document_id": str(doc.id),
                "file_name": doc.file_name,
                "file_type": doc.file_type,
                "file_path": doc.file_path,
                "file_size_bytes": doc.file_size_bytes,
                "file_hash": doc.file_hash,
                "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
                "chunks": chunks,
                "metadata": doc.metadata_,
            })
        return documents

    async def get_document_info(self, document_id: UUID) -> dict[str, Any] | None:
        """Get detailed document info including chunk count."""
        doc = await self.get_by_id(document_id)
        if not doc:
            return None

        chunk_count = await self.session.execute(
            text("SELECT COUNT(*) FROM public.data_document_chunks WHERE (metadata_->>'document_id')::uuid = :doc_id"),
            {"doc_id": str(document_id)},
        )
        chunks = chunk_count.scalar() or 0

        return {
            "document_id": str(doc.id),
            "file_name": doc.file_name,
            "file_type": doc.file_type,
            "file_path": doc.file_path,
            "file_size_bytes": doc.file_size_bytes,
            "file_hash": doc.file_hash,
            "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
            "chunks": chunks,
            "metadata": doc.metadata_,
        }

    async def add_chunks(
        self,
        document_id: UUID,
        chunks: list[dict[str, Any]],
    ) -> list[DocumentChunk]:
        """
        Add chunks for a document.

        Each chunk dict should have:
        - chunk_index: int
        - content: str
        - content_with_context: str (optional)
        - embedding: list[float] (optional, can be added later)
        - metadata: dict (optional)
        """
        chunk_models = []
        for chunk in chunks:
            chunk_model = DocumentChunk(
                document_id=document_id,
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                content_with_context=chunk.get("content_with_context"),
                embedding=chunk.get("embedding"),
                metadata_=chunk.get("metadata", {}),
            )
            chunk_models.append(chunk_model)

        return await self.add_all(chunk_models)

    async def update_chunk_embedding(
        self,
        chunk_id: UUID,
        embedding: list[float],
    ) -> None:
        """Update embedding for a chunk."""
        chunk = await self.session.get(DocumentChunk, chunk_id)
        if chunk:
            chunk.embedding = embedding
            await self.session.flush()

    async def get_all_chunks(self) -> list[DocumentChunk]:
        """Get all chunks (for BM25 indexing)."""
        result = await self.session.execute(
            select(DocumentChunk).order_by(
                DocumentChunk.document_id, DocumentChunk.chunk_index
            )
        )
        return list(result.scalars().all())

    async def get_chunks_for_document(self, document_id: UUID) -> list[DocumentChunk]:
        """Get all chunks for a specific document."""
        result = await self.session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        return list(result.scalars().all())

    async def delete_document_with_chunks(self, document_id: UUID) -> bool:
        """Delete document and all its chunks (CASCADE)."""
        return await self.delete_by_id(document_id)

    async def search_chunks_by_content(
        self, query: str, limit: int = 10
    ) -> list[tuple[DocumentChunk, float]]:
        """
        BM25 full-text search using pg_search.
        Returns list of (chunk, score) tuples.
        """
        # Use pg_search BM25 index
        sql = text("""
            SELECT dc.*, paradedb.score(dc.id) as bm25_score
            FROM document_chunks dc
            WHERE dc.id @@@ paradedb.match('content', :query)
            ORDER BY bm25_score DESC
            LIMIT :limit
        """)
        result = await self.session.execute(sql, {"query": query, "limit": limit})
        rows = result.fetchall()

        chunks_with_scores = []
        for row in rows:
            # Reconstruct chunk from row data
            chunk = DocumentChunk(
                id=row.id,
                document_id=row.document_id,
                chunk_index=row.chunk_index,
                content=row.content,
                content_with_context=row.content_with_context,
                embedding=row.embedding,
                metadata_=row.metadata,
                created_at=row.created_at,
            )
            chunks_with_scores.append((chunk, row.bm25_score))
        return chunks_with_scores
