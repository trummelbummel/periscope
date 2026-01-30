"""Tests for Pydantic models (core contracts)."""

import pytest

from periscope.models import QueryRequest, RetrievedNode


def test_query_request_rejects_empty_query() -> None:
    """QueryRequest rejects empty query (API contract)."""
    with pytest.raises(ValueError):
        QueryRequest(query="")


def test_retrieved_node_has_text_and_score() -> None:
    """RetrievedNode has text and score for pipeline."""
    n = RetrievedNode(text="chunk", score=0.9)
    assert n.text == "chunk"
    assert n.score == 0.9
