"""Tests for document_reader (periscope.ingestion.document_reader)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llama_index.core import Document

from periscope.ingestion.document_reader import (
    DocumentReader,
    load_documents_from_directory,
    read_pdf_path,
)


def test_read_pdf_path_raises_for_missing_file(tmp_path: Path) -> None:
    """read_pdf_path raises FileNotFoundError when path does not exist."""
    missing = tmp_path / "missing.pdf"
    with pytest.raises(FileNotFoundError):
        read_pdf_path(missing)


def test_read_pdf_path_returns_text_when_conversion_succeeds(tmp_path: Path) -> None:
    """read_pdf_path returns extracted text when PyMuPDF conversion succeeds."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")
    mock_text = "# Title\n\nSome body text."

    with patch("periscope.ingestion.document_reader._open_pdf") as m_open:
        m_open.return_value = MagicMock()
        with patch("periscope.ingestion.document_reader._extract_text_from_pdf", return_value=mock_text):
            result = read_pdf_path(pdf)
        assert result == mock_text
        m_open.assert_called_once_with(pdf)


def test_read_pdf_path_propagates_conversion_error(tmp_path: Path) -> None:
    """read_pdf_path re-raises when PyMuPDF conversion fails."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    with patch("periscope.ingestion.document_reader._open_pdf") as m_open:
        m_open.side_effect = RuntimeError("conversion failed")
        with pytest.raises(RuntimeError, match="conversion failed"):
            read_pdf_path(pdf)


def test_load_documents_returns_empty_when_directory_missing(tmp_path: Path) -> None:
    """load_documents returns empty list when configured directory does not exist."""
    missing_dir = tmp_path / "nonexistent"
    reader = DocumentReader(directory=missing_dir)
    assert reader.load_documents() == []


def test_load_documents_returns_empty_when_no_matching_files(tmp_path: Path) -> None:
    """load_documents returns empty list when directory has no matching extensions."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "readme.txt").write_text("hello")
    reader = DocumentReader(directory=tmp_path, required_extensions=[".pdf"])
    assert reader.load_documents() == []


def test_load_documents_returns_documents_with_metadata(tmp_path: Path) -> None:
    """load_documents returns LlamaIndex Documents with file_path, headers, tables in metadata."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    mock_text = "# Introduction\n\nContent."
    mock_headers = ["Introduction"]

    with patch("periscope.ingestion.document_reader._open_pdf") as m_open:
        m_open.return_value = MagicMock()
        with patch("periscope.ingestion.document_reader._extract_text_from_pdf", return_value=mock_text):
            with patch("periscope.ingestion.document_reader._extract_headers_from_pdf", return_value=mock_headers):
                with patch("periscope.ingestion.document_reader._extract_tables_from_pdf", return_value=[]):
                    reader = DocumentReader(directory=tmp_path, required_extensions=[".pdf"])
                    docs = reader.load_documents()

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].text == mock_text
    assert docs[0].metadata["file_path"] == str(pdf.resolve())
    assert docs[0].metadata["headers"] == ["Introduction"]


def test_load_documents_includes_tables_in_metadata(tmp_path: Path) -> None:
    """load_documents adds table Markdown to metadata when PyMuPDF extracts tables."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    mock_text = "# Paper\n\nTable below."
    mock_tables = ["| col |\n|---|\n| 1 |"]

    with patch("periscope.ingestion.document_reader._open_pdf") as m_open:
        m_open.return_value = MagicMock()
        with patch("periscope.ingestion.document_reader._extract_text_from_pdf", return_value=mock_text):
            with patch("periscope.ingestion.document_reader._extract_headers_from_pdf", return_value=[]):
                with patch("periscope.ingestion.document_reader._extract_tables_from_pdf", return_value=mock_tables):
                    reader = DocumentReader(directory=tmp_path, required_extensions=[".pdf"])
                    docs = reader.load_documents()

    assert len(docs) == 1
    assert docs[0].metadata["tables"] == ["| col |\n|---|\n| 1 |"]


def test_load_documents_from_directory_default_delegates(tmp_path: Path) -> None:
    """load_documents_from_directory uses DocumentReader and returns its result."""
    with patch.object(
        DocumentReader,
        "load_documents",
        return_value=[Document(text="x", metadata={"file_path": "y"})],
    ) as m_load:
        result = load_documents_from_directory(
            directory=tmp_path, required_extensions=[".pdf"]
        )
    assert len(result) == 1
    assert result[0].text == "x"
    m_load.assert_called_once()
