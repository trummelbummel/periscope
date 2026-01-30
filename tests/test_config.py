"""Tests for config (periscope.config)."""

from periscope import config


def test_generation_prompt_has_placeholders() -> None:
    """GENERATION_PROMPT contains context_str and query_str for answer generation."""
    assert "{context_str}" in config.GENERATION_PROMPT
    assert "{query_str}" in config.GENERATION_PROMPT


def test_data_dir_is_path() -> None:
    """DATA_DIR is a Path for document loading."""
    assert hasattr(config.DATA_DIR, "mkdir")
