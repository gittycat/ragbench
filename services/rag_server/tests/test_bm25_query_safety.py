from pathlib import Path
import sys
from unittest.mock import AsyncMock, Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.asyncio
async def test_pgsearch_retriever_uses_match_for_raw_user_queries():
    """BM25 retriever should use match() and preserve raw user text."""
    from infrastructure.search.bm25_retriever import PgSearchBM25Retriever

    query = "what's an LLM"
    retriever = PgSearchBM25Retriever(similarity_top_k=10)

    result_proxy = Mock()
    result_proxy.fetchall.return_value = []

    session = AsyncMock()
    session.execute.return_value = result_proxy

    results = await retriever._search_bm25(session, query)

    assert results == []
    sql, params = session.execute.await_args.args
    assert "paradedb.match('content', :query)" in str(sql)
    assert params["query"] == query


@pytest.mark.asyncio
async def test_document_bm25_search_uses_match():
    """BM25 search should route user text through match()."""
    from infrastructure.database.documents import search_chunks_bm25

    query = "what's an LLM"

    result_proxy = Mock()
    result_proxy.fetchall.return_value = []

    session = AsyncMock()
    session.execute.return_value = result_proxy

    results = await search_chunks_bm25(session, query=query, limit=5)

    assert results == []
    sql, params = session.execute.await_args.args
    assert "paradedb.match('content', :query)" in str(sql)
    assert params == {"query": query, "limit": 5}
