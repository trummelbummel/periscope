"""Tests for guardrails (context_engineering_rag.monitoring.guardrails)."""

from context_engineering_rag.models import RetrievedNode
from context_engineering_rag.monitoring import should_abstain


def test_should_abstain_empty_nodes() -> None:
    """Empty nodes -> abstain."""
    assert should_abstain([]) is True


def test_should_abstain_below_threshold() -> None:
    """Best score below threshold -> abstain."""
    nodes = [
        RetrievedNode(text="a", score=0.3),
        RetrievedNode(text="b", score=0.4),
    ]
    assert should_abstain(nodes, threshold=0.5) is True


def test_should_not_abstain_above_threshold() -> None:
    """Best score at or above threshold -> do not abstain."""
    nodes = [
        RetrievedNode(text="a", score=0.6),
        RetrievedNode(text="b", score=0.4),
    ]
    assert should_abstain(nodes, threshold=0.5) is False


def test_should_not_abstain_at_threshold() -> None:
    """Best score equal to threshold -> do not abstain."""
    nodes = [RetrievedNode(text="a", score=0.5)]
    assert should_abstain(nodes, threshold=0.5) is False
