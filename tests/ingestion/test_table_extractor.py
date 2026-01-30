"""Tests for table extractor (periscope.ingestion.table_extractor)."""

from pathlib import Path

import pytest

from periscope.ingestion.table_extractor import PdfTableExtractor
from llama_index.core import Document


def test_extract_tables_from_pdf_raises_when_file_missing() -> None:
    """extract_tables_from_pdf raises FileNotFoundError when path does not exist."""
    extractor = PdfTableExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract_tables_from_pdf(Path("/nonexistent/file.pdf"))


def test_extract_tables_from_pdf_returns_dict_by_page(tmp_path: Path) -> None:
    """extract_tables_from_pdf returns a dict mapping page number to list of tables."""
    try:
        from pypdf import PdfWriter
    except ImportError:
        pytest.skip("pypdf not available")
    pdf_path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    extractor = PdfTableExtractor()
    tables_by_page = extractor.extract_tables_from_pdf(pdf_path)
    assert isinstance(tables_by_page, dict)
    assert 1 in tables_by_page
    assert tables_by_page[1] == []


def test_documents_from_pdf_with_tables_returns_one_doc_per_page(tmp_path: Path) -> None:
    """documents_from_pdf_with_tables returns one Document per page with metadata."""
    try:
        from pypdf import PdfWriter
    except ImportError:
        pytest.skip("pypdf not available")
    pdf_path = tmp_path / "two_pages.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    extractor = PdfTableExtractor()
    docs = extractor.documents_from_pdf_with_tables(pdf_path)
    assert len(docs) == 2
    for i, doc in enumerate(docs, start=1):
        assert isinstance(doc, Document)
        assert doc.metadata.get("page_number") == i
        assert "file_path" in doc.metadata
        assert doc.metadata.get("tables") == []


def test_documents_from_pdf_with_tables_raises_when_file_missing() -> None:
    """documents_from_pdf_with_tables raises FileNotFoundError when path does not exist."""
    extractor = PdfTableExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.documents_from_pdf_with_tables(Path("/nonexistent/file.pdf"))


def test_enrich_documents_with_tables_leaves_non_pdf_unchanged() -> None:
    """enrich_documents_with_tables leaves documents without PDF file_path unchanged."""
    extractor = PdfTableExtractor()
    doc = Document(text="Hello", metadata={"file_path": "/tmp/note.txt"})
    result = extractor.enrich_documents_with_tables([doc])
    assert len(result) == 1
    assert result[0].text == "Hello"
    assert "tables" not in (result[0].metadata or {})


def test_enrich_documents_with_tables_leaves_doc_without_file_path_unchanged() -> None:
    """enrich_documents_with_tables leaves documents without file_path unchanged."""
    extractor = PdfTableExtractor()
    doc = Document(text="Hello", metadata={})
    result = extractor.enrich_documents_with_tables([doc])
    assert len(result) == 1
    assert result[0].text == "Hello"
