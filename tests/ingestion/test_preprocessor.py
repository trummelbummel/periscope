"""Tests for preprocessor (periscope.ingestion.preprocessor)."""

from llama_index.core import Document

from periscope.ingestion.preprocessor import (
    PreprocessingConfig,
    clean_text,
    preprocess_documents,
)


def test_clean_text_strips_reference_section() -> None:
    """clean_text strips content after References when enabled."""
    config = PreprocessingConfig(remove_reference_section=True)
    text = "Main content here.\n\nReferences\n\n[1] Author. Title. 2020."
    result = clean_text(text, config)
    assert "References" not in result
    assert "Main content here." in result


def test_preprocess_documents_preserves_metadata() -> None:
    """preprocess_documents returns new documents with metadata preserved."""
    config = PreprocessingConfig(remove_inline_citations=True)
    doc = Document(
        text="Some text [1] here.",
        metadata={"file_path": "/foo.pdf", "page_number": 1},
    )
    result = preprocess_documents([doc], config)
    assert len(result) == 1
    assert result[0].metadata.get("file_path") == "/foo.pdf"
    assert "[1]" not in result[0].text
