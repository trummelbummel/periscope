"""Tests for config (periscope.config)."""

import os

from periscope import config


def test_port_default() -> None:
    """PORT is int and defaults to 8000 or env."""
    assert isinstance(config.PORT, int)
    assert config.PORT == 8000 or os.environ.get("PORT")


def test_api_host_default() -> None:
    """API_HOST is set (0.0.0.0 or env)."""
    assert config.API_HOST in ("0.0.0.0", os.environ.get("API_HOST", "0.0.0.0"))


def test_data_dir_is_path() -> None:
    """DATA_DIR is a Path."""
    assert hasattr(config.DATA_DIR, "mkdir")


def test_arxiv_data_dir_is_path() -> None:
    """ARXIV_DATA_DIR is a Path."""
    assert hasattr(config.ARXIV_DATA_DIR, "mkdir")


def test_top_k_default() -> None:
    """TOP_K is int (default 10 or env)."""
    assert isinstance(config.TOP_K, int)
    assert config.TOP_K >= 1


def test_embedding_model_set() -> None:
    """EMBEDDING_MODEL is set (bge or env)."""
    assert len(config.EMBEDDING_MODEL) > 0


def test_chroma_persist_dir_is_path() -> None:
    """CHROMA_PERSIST_DIR is a Path."""
    assert hasattr(config.CHROMA_PERSIST_DIR, "mkdir")


def test_generation_model_set() -> None:
    """GENERATION_MODEL is set."""
    assert len(config.GENERATION_MODEL) > 0


def test_generation_prompt_has_placeholders() -> None:
    """GENERATION_PROMPT contains context_str and query_str."""
    assert "{context_str}" in config.GENERATION_PROMPT
    assert "{query_str}" in config.GENERATION_PROMPT


def test_chunk_size_overlap_positive() -> None:
    """CHUNK_SIZE and CHUNK_OVERLAP are positive ints."""
    assert config.CHUNK_SIZE > 0
    assert config.CHUNK_OVERLAP >= 0


def test_similarity_threshold_in_range() -> None:
    """SIMILARITY_THRESHOLD is a float in [0, 1]."""
    assert 0 <= config.SIMILARITY_THRESHOLD <= 1.0


def test_ingestion_stats_path_is_path() -> None:
    """INGESTION_STATS_PATH is a Path."""
    assert hasattr(config.INGESTION_STATS_PATH, "parent")
