"""Document database operations."""

from typing import Any
from uuid import UUID

from sqlalchemy import select, delete, func, literal_column, text
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import Document, DocumentChunk

SORT_COLUMNS = {
    "file_name": Document.file_name,
    "uploaded_at": Document.uploaded_at,
}


async def get_document(session: AsyncSession, document_id: UUID) -> Document | None:
    return await session.get(Document, document_id)


async def get_document_by_hash(session: AsyncSession, file_hash: str) -> Document | None:
    result = await session.execute(
        select(Document).where(Document.file_hash == file_hash)
    )
    return result.scalar_one_or_none()


async def check_duplicates(session: AsyncSession, file_hashes: list[str]) -> dict[str, str]:
    """Return {hash: document_id} for existing documents."""
    if not file_hashes:
        return {}
    result = await session.execute(
        select(Document.file_hash, Document.id).where(
            Document.file_hash.in_(file_hashes)
        )
    )
    return {row.file_hash: str(row.id) for row in result.all()}


async def create_document(
    session: AsyncSession,
    file_name: str,
    file_type: str,
    file_path: str | None = None,
    file_size_bytes: int | None = None,
    file_hash: str | None = None,
    metadata: dict[str, Any] | None = None,
    document_id: UUID | None = None,
) -> Document:
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
    session.add(doc)
    await session.flush()
    return doc


async def list_documents(
    session: AsyncSession,
    sort_by: str = "uploaded_at",
    sort_order: str = "desc",
) -> list[dict[str, Any]]:
    """List all documents with chunk counts."""
    # Count chunks from LlamaIndex's data_document_chunks table
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
    if sort_by == "chunks":
        sort_expr = func.coalesce(subquery.c.chunk_count, 0)
    else:
        sort_expr = SORT_COLUMNS.get(sort_by, Document.uploaded_at)

    if sort_order == "desc":
        query = query.order_by(sort_expr.desc())
    else:
        query = query.order_by(sort_expr.asc())

    result = await session.execute(query)
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


async def get_document_info(session: AsyncSession, document_id: UUID) -> dict[str, Any] | None:
    doc = await session.get(Document, document_id)
    if not doc:
        return None

    chunk_count = await session.execute(
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
    session: AsyncSession,
    document_id: UUID,
    chunks: list[dict[str, Any]],
) -> list[DocumentChunk]:
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

    session.add_all(chunk_models)
    await session.flush()
    return chunk_models


async def update_chunk_embedding(
    session: AsyncSession,
    chunk_id: UUID,
    embedding: list[float],
) -> None:
    chunk = await session.get(DocumentChunk, chunk_id)
    if chunk:
        chunk.embedding = embedding
        await session.flush()


async def get_all_chunks(session: AsyncSession) -> list[DocumentChunk]:
    result = await session.execute(
        select(DocumentChunk).order_by(
            DocumentChunk.document_id, DocumentChunk.chunk_index
        )
    )
    return list(result.scalars().all())


async def get_chunks_for_document(session: AsyncSession, document_id: UUID) -> list[DocumentChunk]:
    result = await session.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    return list(result.scalars().all())


async def delete_document(session: AsyncSession, document_id: UUID) -> bool:
    """Delete document and all its chunks (CASCADE)."""
    result = await session.execute(
        delete(Document).where(Document.id == document_id)
    )
    await session.flush()
    return result.rowcount > 0


async def search_chunks_bm25(
    session: AsyncSession, query: str, limit: int = 10
) -> list[tuple[DocumentChunk, float]]:
    """BM25 full-text search using pg_search. Returns (chunk, score) tuples."""
    sql = text("""
        SELECT dc.*, paradedb.score(dc.id) as bm25_score
        FROM document_chunks dc
        WHERE dc.id @@@ paradedb.match('content', :query)
        ORDER BY bm25_score DESC
        LIMIT :limit
    """)
    result = await session.execute(sql, {"query": query, "limit": limit})
    rows = result.fetchall()

    chunks_with_scores = []
    for row in rows:
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
