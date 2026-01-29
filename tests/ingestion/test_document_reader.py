"""Tests for document_reader (context_engineering_rag.ingestion.document_reader)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from context_engineering_rag.ingestion.document_reader import (
    DocumentReader,
    load_documents_from_directory,
    read_pdf_path,
)


def test_document_reader_init_uses_defaults() -> None:
    """DocumentReader() uses config DATA_DIR and DEFAULT_DOCUMENT_EXTENSIONS."""
    reader = DocumentReader()
    assert reader.directory is not None
    assert reader.required_extensions == [".pdf"]


def test_document_reader_init_accepts_overrides() -> None:
    """DocumentReader accepts directory and required_extensions."""
    reader = DocumentReader(
        directory=Path("/tmp"),
        required_extensions=[".pdf", ".txt"],
    )
    assert reader.directory == Path("/tmp")
    assert reader.required_extensions == [".pdf", ".txt"]


def test_read_pdf_path_raises_when_file_missing() -> None:
    """read_pdf_path raises FileNotFoundError when path does not exist."""
    with pytest.raises(FileNotFoundError):
        read_pdf_path(Path("/nonexistent/file.pdf"))


def test_read_pdf_path_extracts_text(tmp_path: Path) -> None:
    """read_pdf_path returns extracted text from a valid PDF."""
    try:
        from pypdf import PdfReader
    except ImportError:
        pytest.skip("pypdf not available")
    pdf_path = tmp_path / "test.pdf"
    # Create minimal PDF with pypdf (write a simple file pypdf can read)
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_metadata({"/Title": "Test"})
    with open(pdf_path, "wb") as f:
        writer.write(f)
    reader = DocumentReader()
    text = reader.read_pdf_path(pdf_path)
    assert isinstance(text, str)


def test_load_documents_from_directory_empty_dir(tmp_path: Path) -> None:
    """load_documents_from_directory returns a list when given a directory."""
    with patch(
        "context_engineering_rag.ingestion.document_reader.SimpleDirectoryReader"
    ) as mock_reader:
        mock_reader.return_value.load_data.return_value = []
        result = load_documents_from_directory(directory=tmp_path)
    assert result == []


def test_load_documents_from_directory_nonexistent_returns_empty() -> None:
    """load_documents_from_directory returns [] when directory does not exist."""
    result = load_documents_from_directory(directory=Path("/nonexistent/dir"))
    assert result == []


def test_load_documents_from_directory_accepts_extensions(tmp_path: Path) -> None:
    """load_documents_from_directory accepts required_extensions and returns list."""
    with patch(
        "context_engineering_rag.ingestion.document_reader.SimpleDirectoryReader"
    ) as mock_reader:
        mock_reader.return_value.load_data.return_value = []
        result = load_documents_from_directory(
            directory=tmp_path,
            required_extensions=[".pdf"],
        )
    assert isinstance(result, list)
