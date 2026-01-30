"""Tests for pipeline (periscope.app.pipeline)."""

from unittest.mock import MagicMock, patch

from periscope.app.pipeline import run_query
from periscope.models import QueryResponse, RetrievedNode


def test_run_query_returns_abstained_when_guardrails_trigger() -> None:
    """run_query returns abstained=True when guardrails enabled and should_abstain is True."""
    mock_index = MagicMock()
    mock_nodes: list = []
    with (
        patch("periscope.app.pipeline.ENABLE_GUARDRAILS", True),
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


def test_run_query_does_not_abstain_when_guardrails_disabled() -> None:
    """run_query proceeds to generation when ENABLE_GUARDRAILS is False even if scores are low."""
    mock_index = MagicMock()
    mock_nodes: list = []
    sources = [RetrievedNode(text="ctx", score=0.001)]
    with (
        patch("periscope.app.pipeline.ENABLE_GUARDRAILS", False),
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
        mock_nws.score = 0.001
        mock_retrieve.return_value = [mock_nws]
        mock_abstain.return_value = True  # would abstain if guardrails on
        mock_gen.return_value = "Answer anyway"
        result = run_query("query", mock_index, mock_nodes)
    assert result.abstained is False
    assert result.answer == "Answer anyway"
    mock_gen.assert_called_once()


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


def test_run_query_filters_by_min_perf_improvement() -> None:
    """run_query drops sources below min_perf_improvement based on metadata."""
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
        # Two nodes: only one meets the threshold.
        high = MagicMock()
        high.node.get_content.return_value = "high perf"
        high.node.node_id = "id-high"
        high.node.metadata = {"perf_improvement_value": 10.0}
        high.score = 0.9

        low = MagicMock()
        low.node.get_content.return_value = "low perf"
        low.node.node_id = "id-low"
        low.node.metadata = {"perf_improvement_value": 1.0}
        low.score = 0.8

        mock_retrieve.return_value = [high, low]
        mock_abstain.return_value = False
        mock_gen.return_value = "Answer"

        result = run_query(
            "query",
            mock_index,
            mock_nodes,
            top_k=None,
            min_perf_improvement=5.0,
        )

    assert isinstance(result, QueryResponse)
    assert result.abstained is False
    # Only the high-perf source should remain.
    assert len(result.sources) == 1
    assert result.sources[0].metadata["perf_improvement_value"] == 10.0


