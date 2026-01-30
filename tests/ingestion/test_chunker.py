"""Tests for chunker (periscope.ingestion.chunker)."""

from llama_index.core import Document

from periscope.ingestion import chunk_documents


def test_chunk_documents_produces_nodes() -> None:
    """chunk_documents splits documents into nodes for indexing."""
    docs = [
        Document(text="First sentence. Second sentence. Third sentence. " * 30),
    ]
    nodes = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
    assert len(nodes) >= 1
    for node in nodes:
        assert len(node.get_content()) > 0
