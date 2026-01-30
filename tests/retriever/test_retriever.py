"""Tests for retriever (periscope.retriever.retriever)."""

from unittest.mock import MagicMock, patch

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode

from periscope.retriever.retriever import (
    HybridRetriever,
    _reciprocal_rank_fusion,
    hybrid_retrieve,
)


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
    assert {f.node.node_id for f in fused} == {"id1", "id2"}


def test_hybrid_retrieve_returns_list() -> None:
    """hybrid_retrieve returns list of NodeWithScore (retrieval + RRF)."""
    mock_index = MagicMock(spec=VectorStoreIndex)
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_index.as_retriever.return_value = mock_retriever

    with patch("periscope.retriever.retriever.set_global_embed_model"):
        with patch("periscope.retriever.retriever.BM25Retriever") as mock_bm25_class:
            mock_bm25 = MagicMock()
            mock_bm25.retrieve.return_value = []
            mock_bm25_class.return_value = mock_bm25
            result = hybrid_retrieve("query", mock_index, [], top_k=5)
    assert isinstance(result, list)
