"""Tests for pipeline (periscope.app.pipeline)."""

from unittest.mock import MagicMock, patch

from periscope.app.pipeline import run_query
from periscope.models import QueryResponse, RetrievedNode


def test_run_query_returns_abstained_when_guardrails_trigger() -> None:
    """run_query returns abstained=True when should_abstain is True (edge case)."""
    mock_index = MagicMock()
    mock_nodes: list = []
    with (
        patch(
            "periscope.app.pipeline.HybridRetriever.hybrid_retrieve"
        ) as mock_retrieve,
        patch(
            "periscope.app.pipeline.should_abstain"
        ) as mock_abstain,
    ):
        mock_retrieve.return_value = []
        mock_abstain.return_value = True
        result = run_query("query", mock_index, mock_nodes)
    assert isinstance(result, QueryResponse)
    assert result.abstained is True
    assert result.answer == ""
    assert "abstained_reason" in result.metadata


def test_run_query_returns_answer_when_generation_succeeds() -> None:
    """run_query returns answer and abstained=False when generation succeeds (core)."""
    mock_index = MagicMock()
    mock_nodes: list = []
    sources = [RetrievedNode(text="ctx", score=0.9)]
    with (
        patch(
            "periscope.app.pipeline.HybridRetriever.hybrid_retrieve"
        ) as mock_retrieve,
        patch(
            "periscope.app.pipeline.should_abstain"
        ) as mock_abstain,
        patch(
            "periscope.app.pipeline.AnswerGenerator.generate_answer"
        ) as mock_gen,
    ):
        # hybrid_retrieve returns NodeWithScore-like; pipeline converts to RetrievedNode
        mock_nws = MagicMock()
        mock_nws.node.get_content.return_value = "ctx"
        mock_nws.node.node_id = "id1"
        mock_nws.node.metadata = {}
        mock_nws.score = 0.9
        mock_retrieve.return_value = [mock_nws]
        mock_abstain.return_value = False
        mock_gen.return_value = "Generated answer"
        result = run_query("query", mock_index, mock_nodes)
    assert isinstance(result, QueryResponse)
    assert result.abstained is False
    assert result.answer == "Generated answer"
    assert len(result.sources) == 1
    assert "retrieval_time_seconds" in result.metadata
    assert "generation_time_seconds" in result.metadata


def test_run_query_returns_generation_error_metadata_on_exception() -> None:
    """run_query captures generation_error in metadata when generate_answer raises (edge case)."""
    mock_index = MagicMock()
    mock_nodes: list = []
    with (
        patch(
            "periscope.app.pipeline.HybridRetriever.hybrid_retrieve"
        ) as mock_retrieve,
        patch(
            "periscope.app.pipeline.should_abstain"
        ) as mock_abstain,
        patch(
            "periscope.app.pipeline.AnswerGenerator.generate_answer"
        ) as mock_gen,
    ):
        mock_nws = MagicMock()
        mock_nws.node.get_content.return_value = "ctx"
        mock_nws.node.node_id = "id1"
        mock_nws.node.metadata = {}
        mock_nws.score = 0.9
        mock_retrieve.return_value = [mock_nws]
        mock_abstain.return_value = False
        mock_gen.side_effect = RuntimeError("API error")
        result = run_query("query", mock_index, mock_nodes)
    assert result.abstained is False
    assert result.answer == ""
    assert "generation_error" in result.metadata
    assert "API error" in result.metadata["generation_error"]
