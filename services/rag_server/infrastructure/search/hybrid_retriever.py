"""Hybrid retriever combining BM25 and vector search with RRF fusion."""

import logging
from collections import defaultdict
from typing import Any

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

logger = logging.getLogger(__name__)


class HybridRRFRetriever(BaseRetriever):
    """
    Hybrid retriever using Reciprocal Rank Fusion (RRF) to combine
    BM25 (sparse) and vector (dense) search results.

    RRF Formula: score = sum(1 / (k + rank)) for each result list
    where k is a constant (default 60) that controls rank sensitivity.

    Research shows hybrid search improves retrieval by ~48% vs vector-only,
    and combined with reranking achieves 67% improvement.
    """

    def __init__(
        self,
        bm25_retriever: BaseRetriever,
        vector_retriever: BaseRetriever,
        rrf_k: int = 60,
        similarity_top_k: int = 10,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
    ):
        super().__init__()
        self._bm25_retriever = bm25_retriever
        self._vector_retriever = vector_retriever
        self._rrf_k = rrf_k
        self._similarity_top_k = similarity_top_k
        self._bm25_weight = bm25_weight
        self._vector_weight = vector_weight

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Synchronous retrieve using RRF fusion."""
        # Get results from both retrievers
        bm25_results = self._bm25_retriever.retrieve(query_bundle)
        vector_results = self._vector_retriever.retrieve(query_bundle)

        return self._fuse_results(bm25_results, vector_results)

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Async retrieve using RRF fusion."""
        import asyncio

        # Run both retrievers in parallel
        bm25_task = asyncio.create_task(
            self._bm25_retriever._aretrieve(query_bundle)
        )
        vector_task = asyncio.create_task(
            self._vector_retriever._aretrieve(query_bundle)
        )

        bm25_results, vector_results = await asyncio.gather(bm25_task, vector_task)

        return self._fuse_results(bm25_results, vector_results)

    def _fuse_results(
        self,
        bm25_results: list[NodeWithScore],
        vector_results: list[NodeWithScore],
    ) -> list[NodeWithScore]:
        """
        Fuse results using Reciprocal Rank Fusion.

        RRF score = bm25_weight * (1 / (k + bm25_rank)) + vector_weight * (1 / (k + vector_rank))
        """
        logger.debug(
            f"[HYBRID] Fusing {len(bm25_results)} BM25 + {len(vector_results)} vector results"
        )

        # Track scores and nodes by ID
        rrf_scores: dict[str, float] = defaultdict(float)
        node_map: dict[str, NodeWithScore] = {}

        # Score BM25 results
        for rank, node_with_score in enumerate(bm25_results, start=1):
            node_id = node_with_score.node.node_id
            rrf_scores[node_id] += self._bm25_weight * (1.0 / (self._rrf_k + rank))
            if node_id not in node_map:
                node_map[node_id] = node_with_score

        # Score vector results
        for rank, node_with_score in enumerate(vector_results, start=1):
            node_id = node_with_score.node.node_id
            rrf_scores[node_id] += self._vector_weight * (1.0 / (self._rrf_k + rank))
            if node_id not in node_map:
                node_map[node_id] = node_with_score

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build result list with RRF scores
        fused_results = []
        for node_id in sorted_ids[: self._similarity_top_k]:
            original = node_map[node_id]
            fused_results.append(
                NodeWithScore(node=original.node, score=rrf_scores[node_id])
            )

        logger.debug(f"[HYBRID] Fused to {len(fused_results)} results")
        return fused_results


def create_hybrid_retriever(
    vector_index,
    similarity_top_k: int = 10,
    rrf_k: int = 60,
) -> HybridRRFRetriever:
    """
    Create a hybrid retriever combining BM25 (pg_textsearch) and vector search (ChromaDB).

    Args:
        vector_index: VectorStoreIndex for vector similarity search
        similarity_top_k: Number of results to return
        rrf_k: RRF constant (default 60)

    Returns:
        HybridRRFRetriever instance
    """
    from infrastructure.search.bm25_retriever import PgSearchBM25Retriever

    # Create BM25 retriever using pg_textsearch
    bm25_retriever = PgSearchBM25Retriever(similarity_top_k=similarity_top_k)

    # Create vector retriever from ChromaDB index
    vector_retriever = vector_index.as_retriever(similarity_top_k=similarity_top_k)

    return HybridRRFRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        rrf_k=rrf_k,
        similarity_top_k=similarity_top_k,
    )
