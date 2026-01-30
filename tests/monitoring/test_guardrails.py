"""Tests for guardrails (periscope.monitoring.guardrails)."""

from periscope.models import RetrievedNode
from periscope.monitoring import should_abstain


def test_should_abstain_empty_nodes() -> None:
    """should_abstain returns True when no nodes (abstain)."""
    assert should_abstain([]) is True


def test_should_abstain_below_threshold() -> None:
    """should_abstain returns True when best score below threshold."""
    nodes = [
        RetrievedNode(text="a", score=0.1),
        RetrievedNode(text="b", score=0.2),
    ]
    assert should_abstain(nodes, threshold=0.5) is True


def test_should_not_abstain_above_threshold() -> None:
    """should_abstain returns False when best score above threshold."""
    nodes = [
        RetrievedNode(text="a", score=0.9),
        RetrievedNode(text="b", score=0.2),
    ]
    assert should_abstain(nodes, threshold=0.5) is False
