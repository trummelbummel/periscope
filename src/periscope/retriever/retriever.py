"""Hybrid retrieval: combine keyword search (BM25) and vector similarity search.

Per PRD: BM25 for keyword search, vector similarity, TOP_K = 10. Use Llama-index.
"""

import logging
from collections import defaultdict

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever

from periscope.config import RRF_K, TOP_K
from periscope.embedder import set_global_embed_model

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines BM25 and vector retrieval with RRF fusion."""

    def __init__(
        self,
        vector_index: VectorStoreIndex,
        bm25_nodes: list[BaseNode],
        top_k: int | None = None,
    ) -> None:
        """Initialize with vector index and nodes for BM25; default top_k from config.

        :param vector_index: Vector store index (Chroma).
        :param bm25_nodes: Nodes for BM25 keyword search.
        :param top_k: Final number of results; default from config.
        """
        self._vector_index = vector_index
        self._bm25_nodes = bm25_nodes
        self._top_k = HybridRetriever._resolve_top_k(top_k)

    def retrieve(self, query: str) -> list[NodeWithScore]:
        """Run hybrid retrieval: BM25 + vector, then RRF merge.

        :param query: User query.
        :return: Merged and re-ranked NodeWithScore list.
        """
        set_global_embed_model()
        vector_retriever = self._vector_index.as_retriever(
            similarity_top_k=self._top_k
        )
        bm25_retriever = BM25Retriever(
            nodes=self._bm25_nodes,
            similarity_top_k=self._top_k,
        )
        vector_results = vector_retriever.retrieve(query)
        bm25_results = bm25_retriever.retrieve(query)
        fused = HybridRetriever._reciprocal_rank_fusion([vector_results, bm25_results])
        return fused[: self._top_k]

    @staticmethod
    def _resolve_top_k(top_k: int | None) -> int:
        """Return top_k or config default."""
        return top_k if top_k is not None else TOP_K

    @staticmethod
    def _reciprocal_rank_fusion(
        results_list: list[list[NodeWithScore]], k: int = RRF_K
    ) -> list[NodeWithScore]:
        """Merge multiple ranked lists using reciprocal rank fusion (RRF).

        RRF score = sum(1 / (k + rank_i)) across retrievers. Deduplicates by node_id.

        :param results_list: List of ranked lists from different retrievers.
        :param k: RRF constant; default 60.
        :return: Merged and re-ranked NodeWithScore list.
        """
        scores: dict[str, float] = defaultdict(float)
        node_map: dict[str, NodeWithScore] = {}
        for results in results_list:
            for rank, nws in enumerate(results, start=1):
                node_id = nws.node.node_id
                rrf = 1.0 / (k + rank)
                scores[node_id] += rrf
                if node_id not in node_map:
                    node_map[node_id] = NodeWithScore(node=nws.node, score=0.0)
        for node_id, total in scores.items():
            node_map[node_id].score = total
        return sorted(node_map.values(), key=lambda x: x.score, reverse=True)

    @staticmethod
    def get_vector_retriever(
        index: VectorStoreIndex,
        top_k: int | None = None,
    ) -> VectorIndexRetriever:
        """Return vector similarity retriever from index.

        :param index: Vector store index.
        :param top_k: Number of results; default from config.
        :return: VectorIndexRetriever instance.
        """
        return index.as_retriever(
            similarity_top_k=HybridRetriever._resolve_top_k(top_k)
        )

    @staticmethod
    def get_bm25_retriever(
        nodes: list[BaseNode],
        top_k: int | None = None,
    ) -> BM25Retriever:
        """Return BM25 retriever over nodes.

        :param nodes: Nodes to index for BM25.
        :param top_k: Number of results; default from config.
        :return: BM25Retriever instance.
        """
        return BM25Retriever(
            nodes=nodes,
            similarity_top_k=HybridRetriever._resolve_top_k(top_k),
        )

    @staticmethod
    def hybrid_retrieve(
        query: str,
        vector_index: VectorStoreIndex,
        bm25_nodes: list[BaseNode],
        top_k: int | None = None,
    ) -> list[NodeWithScore]:
        """Run hybrid retrieval: BM25 + vector, then RRF merge.

        :param query: User query.
        :param vector_index: Vector store index (Chroma).
        :param bm25_nodes: Same nodes used for BM25 index (for keyword search).
        :param top_k: Final number of results; default from config.
        :return: Merged and re-ranked NodeWithScore list.
        """
        retriever = HybridRetriever(
            vector_index=vector_index,
            bm25_nodes=bm25_nodes,
            top_k=top_k,
        )
        return retriever.retrieve(query)


def _resolve_top_k(top_k: int | None) -> int:
    """Return top_k or config default. Delegates to HybridRetriever."""
    return HybridRetriever._resolve_top_k(top_k)


def _reciprocal_rank_fusion(
    results_list: list[list[NodeWithScore]], k: int = RRF_K
) -> list[NodeWithScore]:
    """Merge ranked lists with RRF. Delegates to HybridRetriever."""
    return HybridRetriever._reciprocal_rank_fusion(results_list, k=k)


def get_vector_retriever(
    index: VectorStoreIndex,
    top_k: int | None = None,
) -> VectorIndexRetriever:
    """Return vector similarity retriever from index."""
    return HybridRetriever.get_vector_retriever(index, top_k=top_k)


def get_bm25_retriever(
    nodes: list[BaseNode],
    top_k: int | None = None,
) -> BM25Retriever:
    """Return BM25 retriever over nodes."""
    return HybridRetriever.get_bm25_retriever(nodes, top_k=top_k)


def hybrid_retrieve(
    query: str,
    vector_index: VectorStoreIndex,
    bm25_nodes: list[BaseNode],
    top_k: int | None = None,
) -> list[NodeWithScore]:
    """Run hybrid retrieval. Delegates to HybridRetriever."""
    return HybridRetriever.hybrid_retrieve(
        query, vector_index, bm25_nodes, top_k=top_k
    )
