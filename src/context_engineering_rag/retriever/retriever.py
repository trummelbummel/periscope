"""Hybrid retrieval: combine keyword search (BM25) and vector similarity search.

Per PRD: BM25 for keyword search, vector similarity, TOP_K = 10. Use Llama-index.
"""

import logging
from collections import defaultdict

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from context_engineering_rag.config import TOP_K
from context_engineering_rag.embedder import set_global_embed_model

logger = logging.getLogger(__name__)


def _reciprocal_rank_fusion(
    results_list: list[list[NodeWithScore]], k: int = 60
) -> list[NodeWithScore]:
    """Merge multiple ranked lists using reciprocal rank fusion (RRF).

    RRF score = sum(1 / (k + rank_i)) across retrievers. Deduplicates by node_id.
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
    sorted_nodes = sorted(
        node_map.values(), key=lambda x: x.score, reverse=True
    )
    return sorted_nodes


def get_vector_retriever(
    index: VectorStoreIndex,
    top_k: int | None = None,
) -> VectorIndexRetriever:
    """Return vector similarity retriever from index."""
    k = top_k if top_k is not None else TOP_K
    return index.as_retriever(similarity_top_k=k)


def get_bm25_retriever(
    nodes: list[BaseNode],
    top_k: int | None = None,
) -> BM25Retriever:
    """Return BM25 retriever over nodes."""
    k = top_k if top_k is not None else TOP_K
    return BM25Retriever(nodes=nodes, similarity_top_k=k)


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
    set_global_embed_model()
    k = top_k if top_k is not None else TOP_K
    vector_retriever = get_vector_retriever(vector_index, top_k=k)
    bm25_retriever = get_bm25_retriever(bm25_nodes, top_k=k)
    vector_results = vector_retriever.retrieve(query)
    bm25_results = bm25_retriever.retrieve(query)
    fused = _reciprocal_rank_fusion([vector_results, bm25_results])
    return fused[:k]
