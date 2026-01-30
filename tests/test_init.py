"""Tests for package __init__ (periscope)."""

from periscope import (
    API_HOST,
    CHROMA_PERSIST_DIR,
    DATA_DIR,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    PORT,
    TOP_K,
    QueryRequest,
    QueryResponse,
    app,
)


def test_app_is_fastapi() -> None:
    """Package exports FastAPI app."""
    from fastapi import FastAPI

    assert isinstance(app, FastAPI)


def test_exports_config_values() -> None:
    """Package exports expected config names."""
    assert PORT is not None
    assert API_HOST is not None
    assert DATA_DIR is not None
    assert CHROMA_PERSIST_DIR is not None
    assert EMBEDDING_MODEL is not None
    assert GENERATION_MODEL is not None
    assert TOP_K is not None


def test_exports_models() -> None:
    """Package exports QueryRequest and QueryResponse."""
    assert QueryRequest is not None
    assert QueryResponse is not None
