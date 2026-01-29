"""Tests for chunker (context_engineering_rag.ingestion.chunker)."""

from llama_index.core import Document

from context_engineering_rag.ingestion import chunk_documents, get_header_aware_chunker


def test_get_header_aware_chunker() -> None:
    """get_header_aware_chunker returns SentenceSplitter with expected params."""
    parser = get_header_aware_chunker(chunk_size=100, chunk_overlap=20)
    assert parser.chunk_size == 100
    assert parser.chunk_overlap == 20


def test_chunk_documents_produces_nodes() -> None:
    """chunk_documents splits documents into nodes."""
    docs = [
        Document(text="First sentence. Second sentence. Third sentence. " * 30),
    ]
    nodes = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
    assert len(nodes) >= 1
    for node in nodes:
        assert len(node.get_content()) > 0
