"""Tests for chunker (periscope.ingestion.chunker)."""

from llama_index.core import Document

from periscope.ingestion import chunk_documents
from periscope.ingestion.chunker import _metadata_byte_size, _trim_document_metadata


def test_chunk_documents_produces_nodes() -> None:
    """chunk_documents splits documents into nodes for indexing."""
    docs = [
        Document(text="First sentence. Second sentence. Third sentence. " * 30),
    ]
    chunk_size = 100
    chunk_overlap = 10
    nodes = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert len(nodes) >= 1
    # MarkdownNodeParser splits by headers; plain text becomes one node, so no chunk size bound
    for node in nodes:
        assert len(node.get_content()) > 0


def test_chunk_documents_handles_large_metadata() -> None:
    """chunk_documents trims metadata so the parser does not raise (metadata > chunk_size)."""
    huge_tables = ["x" * 2000]
    doc = Document(
        text="Short text.",
        metadata={"file_path": "/a.pdf", "headers": ["H1"], "tables": huge_tables},
    )
    assert _metadata_byte_size(doc.metadata) > 512
    nodes = chunk_documents([doc], chunk_size=512, chunk_overlap=10)
    assert len(nodes) >= 1


def test_trim_document_metadata_caps_size() -> None:
    """_trim_document_metadata returns metadata under max size."""
    doc = Document(
        text="x",
        metadata={"file_path": "/a.pdf", "headers": ["H"], "tables": ["t" * 3000]},
    )
    trimmed = _trim_document_metadata(doc, max_metadata_size=400)
    assert _metadata_byte_size(trimmed.metadata) <= 400
    assert trimmed.metadata.get("file_path") == "/a.pdf"
