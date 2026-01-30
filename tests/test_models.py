"""Tests for Pydantic models."""

import pytest

from periscope.models import (
    IngestionStats,
    QueryRequest,
    QueryResponse,
    RetrievedNode,
)


def test_query_request_valid() -> None:
    """QueryRequest accepts valid query and optional top_k."""
    r = QueryRequest(query="What is RAG?")
    assert r.query == "What is RAG?"
    assert r.top_k is None
    r2 = QueryRequest(query="test", top_k=5)
    assert r2.top_k == 5


def test_query_request_min_length() -> None:
    """QueryRequest rejects empty query."""
    with pytest.raises(ValueError):
        QueryRequest(query="")


def test_retrieved_node() -> None:
    """RetrievedNode has text, score, optional node_id and metadata."""
    n = RetrievedNode(text="chunk", score=0.9)
    assert n.text == "chunk"
    assert n.score == 0.9
    assert n.node_id is None
    assert n.metadata == {}


def test_query_response() -> None:
    """QueryResponse has answer, sources, metadata, abstained."""
    r = QueryResponse(answer="Yes", sources=[], metadata={}, abstained=False)
    assert r.answer == "Yes"
    assert r.abstained is False


def test_ingestion_stats() -> None:
    """IngestionStats validates document_count, chunk_count, total_chars."""
    s = IngestionStats(
        document_count=2,
        chunk_count=10,
        total_chars=5000,
        avg_chunk_size=500.0,
        paths=[],
    )
    assert s.document_count == 2
    assert s.chunk_count == 10
    assert s.avg_chunk_size == 500.0
