"""Tests for chunker (periscope.ingestion.chunker)."""

from llama_index.core import Document
from llama_index.core.utils import get_tokenizer

from periscope.ingestion import chunk_documents


def test_chunk_documents_produces_nodes() -> None:
    """chunk_documents splits documents into nodes and respects chunk_size (in tokens)."""
    docs = [
        Document(text="First sentence. Second sentence. Third sentence. " * 30),
    ]
    chunk_size = 100
    chunk_overlap = 10
    nodes = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert len(nodes) >= 1
    tokenizer = get_tokenizer()
    for node in nodes:
        text = node.get_content()
        assert len(text) > 0
        token_count = len(tokenizer(text))
        assert token_count <= chunk_size, (
            f"Chunk has {token_count} tokens, exceeds chunk_size={chunk_size}"
        )


def test_markdown_headers_are_parsed() -> None:
    """chunk_documents uses MarkdownNodeParser to respect multiple markdown header levels."""
    docs = [
        Document(
            text=(
                "# H1 Title\n"
                "Intro text.\n\n"
                "## H2 Section\n"
                "More text.\n\n"
                "### H3 Subsection\n"
                "Even more text.\n\n"
                "#### H4 Deeper\n"
                "Deep text."
            ),
        ),
    ]
    nodes = chunk_documents(docs, chunk_size=100, chunk_overlap=0)
    assert len(nodes) >= 4
    contents = [n.get_content() for n in nodes]
    # Ensure each header level appears in at least one node's text
    assert any("# H1 Title" in c for c in contents)
    assert any("## H2 Section" in c for c in contents)
    assert any("### H3 Subsection" in c for c in contents)
    assert any("#### H4 Deeper" in c for c in contents)
