"""Tests for retriever (context_engineering_rag.retriever.retriever)."""

from unittest.mock import MagicMock, patch

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode

from context_engineering_rag.retriever.retriever import (
    HybridRetriever,
    _reciprocal_rank_fusion,
    _resolve_top_k,
    get_bm25_retriever,
    get_vector_retriever,
    hybrid_retrieve,
)


def test_resolve_top_k_uses_value_when_provided() -> None:
    """_resolve_top_k returns provided value when not None."""
    assert _resolve_top_k(5) == 5


def test_resolve_top_k_uses_config_when_none() -> None:
    """_resolve_top_k returns config TOP_K when None."""
    result = _resolve_top_k(None)
    assert isinstance(result, int)
    assert result >= 1


def test_reciprocal_rank_fusion_deduplicates_by_node_id() -> None:
    """_reciprocal_rank_fusion merges lists and deduplicates by node_id."""
    node1 = TextNode(text="a", id_="id1")
    node2 = TextNode(text="b", id_="id2")
    nws1 = NodeWithScore(node=node1, score=0.9)
    nws2 = NodeWithScore(node=node2, score=0.8)
    list_a = [nws1, nws2]
    list_b = [NodeWithScore(node=node2, score=0.7), NodeWithScore(node=node1, score=0.6)]
    fused = _reciprocal_rank_fusion([list_a, list_b], k=60)
    assert len(fused) == 2
    node_ids = {f.node.node_id for f in fused}
    assert node_ids == {"id1", "id2"}


def test_reciprocal_rank_fusion_returns_sorted_by_score() -> None:
    """_reciprocal_rank_fusion returns list sorted by RRF score descending."""
    node1 = TextNode(text="a", id_="id1")
    node2 = TextNode(text="b", id_="id2")
    list_a = [NodeWithScore(node=node1, score=0.9), NodeWithScore(node=node2, score=0.8)]
    list_b = [NodeWithScore(node=node2, score=0.7), NodeWithScore(node=node1, score=0.6)]
    fused = _reciprocal_rank_fusion([list_a, list_b], k=60)
    assert fused[0].score >= fused[1].score


def test_get_vector_retriever_returns_retriever() -> None:
    """get_vector_retriever returns index.as_retriever with top_k."""
    mock_index = MagicMock(spec=VectorStoreIndex)
    retriever = get_vector_retriever(mock_index, top_k=5)
    mock_index.as_retriever.assert_called_once_with(similarity_top_k=5)
    assert retriever is mock_index.as_retriever.return_value


def test_get_bm25_retriever_returns_bm25_retriever() -> None:
    """get_bm25_retriever returns BM25Retriever with nodes and top_k."""
    from llama_index.retrievers.bm25 import BM25Retriever

    nodes = [
        TextNode(text="chunk one", id_="n1"),
        TextNode(text="chunk two", id_="n2"),
        TextNode(text="chunk three", id_="n3"),
    ]
    retriever = get_bm25_retriever(nodes, top_k=3)
    assert isinstance(retriever, BM25Retriever)
    assert retriever.similarity_top_k == 3


def test_hybrid_retriever_init_uses_resolve_top_k() -> None:
    """HybridRetriever uses _resolve_top_k for top_k."""
    mock_index = MagicMock(spec=VectorStoreIndex)
    retriever = HybridRetriever(mock_index, [], top_k=7)
    assert retriever._top_k == 7


def test_hybrid_retrieve_returns_list() -> None:
    """hybrid_retrieve returns list of NodeWithScore."""
    mock_index = MagicMock(spec=VectorStoreIndex)
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_index.as_retriever.return_value = mock_retriever

    with patch("context_engineering_rag.retriever.retriever.set_global_embed_model"):
        with patch(
            "context_engineering_rag.retriever.retriever.BM25Retriever"
        ) as mock_bm25_class:
            mock_bm25 = MagicMock()
            mock_bm25.retrieve.return_value = []
            mock_bm25_class.return_value = mock_bm25
            result = hybrid_retrieve("query", mock_index, [])

    assert isinstance(result, list)
