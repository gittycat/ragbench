"""BM25 retriever using pg_search (ParadeDB) for PostgreSQL full-text search."""

import logging
from typing import Any

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.postgres import get_session

logger = logging.getLogger(__name__)


class PgSearchBM25Retriever(BaseRetriever):
    """
    BM25 retriever using pg_search (ParadeDB) for true BM25 full-text search.

    Unlike in-memory BM25, this uses PostgreSQL's pg_search extension which:
    - Scales to millions of documents
    - Persists across restarts
    - Uses optimized inverted indexes
    - Supports stemming and tokenization
    """

    def __init__(self, similarity_top_k: int = 10):
        super().__init__()
        self._similarity_top_k = similarity_top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Synchronous retrieve - calls async version."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're in an async context, use nest_asyncio or run in executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, self._aretrieve(query_bundle)
                )
                return future.result()
        else:
            return asyncio.run(self._aretrieve(query_bundle))

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """
        Retrieve documents using pg_search BM25.

        Uses ParadeDB's BM25 index on document_chunks.content field.
        """
        query_str = query_bundle.query_str
        if not query_str.strip():
            return []

        logger.debug(f"[BM25] Searching for: {query_str[:100]}...")

        async with get_session() as session:
            return await self._search_bm25(session, query_str)

    async def _search_bm25(
        self, session: AsyncSession, query_str: str
    ) -> list[NodeWithScore]:
        """Execute BM25 search using pg_search."""
        # Escape single quotes in query
        safe_query = query_str.replace("'", "''")

        # Use pg_search BM25 index with paradedb.parse for better tokenization
        sql = text("""
            SELECT
                dc.id,
                dc.document_id,
                dc.chunk_index,
                dc.content,
                dc.content_with_context,
                dc.metadata,
                dc.created_at,
                d.file_name,
                d.file_type,
                d.file_path,
                d.file_size_bytes,
                d.file_hash,
                d.uploaded_at,
                paradedb.score(dc.id) as bm25_score
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE dc.content @@@ paradedb.parse(:query)
            ORDER BY bm25_score DESC
            LIMIT :limit
        """)

        try:
            result = await session.execute(
                sql, {"query": safe_query, "limit": self._similarity_top_k}
            )
            rows = result.fetchall()
        except Exception as e:
            logger.warning(f"[BM25] Search failed: {e}")
            return []

        nodes_with_scores = []
        for row in rows:
            # Build metadata dict
            metadata = dict(row.metadata) if row.metadata else {}
            metadata.update({
                "document_id": str(row.document_id),
                "chunk_index": row.chunk_index,
                "file_name": row.file_name,
                "file_type": row.file_type,
                "path": row.file_path,
                "file_size_bytes": row.file_size_bytes,
                "file_hash": row.file_hash,
                "uploaded_at": row.uploaded_at.isoformat() if row.uploaded_at else None,
            })

            # Create TextNode
            node = TextNode(
                id_=f"{row.document_id}-chunk-{row.chunk_index}",
                text=row.content_with_context or row.content,
                metadata=metadata,
            )

            nodes_with_scores.append(
                NodeWithScore(node=node, score=float(row.bm25_score))
            )

        logger.debug(f"[BM25] Found {len(nodes_with_scores)} results")
        return nodes_with_scores
