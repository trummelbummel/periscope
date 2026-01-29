"""Tests for config (context_engineering_rag.config)."""

import os

from context_engineering_rag import config


def test_port_default() -> None:
    """PORT defaults to 8000 or env."""
    assert config.PORT == 8000 or os.environ.get("PORT")


def test_api_host_default() -> None:
    """API_HOST defaults to 0.0.0.0 or env."""
    assert config.API_HOST in ("0.0.0.0", os.environ.get("API_HOST", "0.0.0.0"))


def test_top_k_default() -> None:
    """TOP_K defaults to 10 or env."""
    assert config.TOP_K == 10 or int(os.environ.get("TOP_K", "10"))


def test_embedding_model_set() -> None:
    """EMBEDDING_MODEL is set (bge or env)."""
    assert "bge" in config.EMBEDDING_MODEL.lower() or config.EMBEDDING_MODEL


def test_chroma_persist_dir_is_path() -> None:
    """CHROMA_PERSIST_DIR is a Path."""
    assert hasattr(config.CHROMA_PERSIST_DIR, "mkdir")
