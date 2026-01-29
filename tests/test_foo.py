"""Tests for foo (context_engineering_rag.foo)."""

from context_engineering_rag.foo import foo


def test_foo_returns_input() -> None:
    """foo returns the input string."""
    assert foo("foo") == "foo"


def test_foo_empty_string() -> None:
    """foo returns empty string when given empty string (edge case)."""
    assert foo("") == ""
