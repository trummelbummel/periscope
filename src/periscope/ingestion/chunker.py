"""Header-aware chunking: chunking that understands research paper structure.

Per PRD: Execute chunking that understands research paper structure. Use Llama-index.
"""

import json
import logging

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from periscope.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)

PARAGRAPH_SEPARATOR = "\n"

# Leave room so metadata + text per chunk stays under chunk_size in SentenceSplitter
METADATA_SIZE_MARGIN = 128


def _metadata_byte_size(metadata: dict) -> int:
    """Approximate size of metadata when serialized (e.g. for chunker limit)."""
    return len(json.dumps(metadata, ensure_ascii=False))


def _trim_document_metadata(doc: Document, max_metadata_size: int) -> Document:
    """Return a document with metadata capped so serialized size <= max_metadata_size.

    Keeps file_path; truncates or drops headers/tables if needed so metadata fits.
    Avoids SentenceSplitter ValueError when metadata length > chunk_size.
    """
    if not doc.metadata:
        return doc
    if _metadata_byte_size(doc.metadata) <= max_metadata_size:
        return doc
    out: dict = {"file_path": doc.metadata.get("file_path", "")}
    if _metadata_byte_size(out) > max_metadata_size:
        return Document(text=doc.text, metadata=out)
    # Add headers (list of strings) - take prefix that fits
    headers = doc.metadata.get("headers") or []
    if headers and isinstance(headers, list):
        trimmed_headers: list[str] = []
        for h in headers:
            candidate = trimmed_headers + [h]
            if _metadata_byte_size(out | {"headers": candidate}) <= max_metadata_size:
                trimmed_headers = candidate
            else:
                break
        if trimmed_headers:
            out["headers"] = trimmed_headers
    # Add tables (list of strings) - truncate each string so total fits
    tables = doc.metadata.get("tables") or []
    remaining = max_metadata_size - _metadata_byte_size(out)
    if tables and isinstance(tables, list) and remaining > 100:
        max_per_table = max(200, (remaining - 50) // max(len(tables), 1))
        trimmed_tables: list[str] = []
        for t in tables:
            s = t if isinstance(t, str) else str(t)
            if len(s) > max_per_table:
                s = s[: max_per_table - 3] + "..."
            trial = out | {"tables": trimmed_tables + [s]}
            if _metadata_byte_size(trial) <= max_metadata_size:
                trimmed_tables.append(s)
            else:
                break
        if trimmed_tables:
            out["tables"] = trimmed_tables
    logger.debug(
        "Trimmed document metadata from %d to %d bytes",
        _metadata_byte_size(doc.metadata),
        _metadata_byte_size(out),
    )
    return Document(text=doc.text, metadata=out)


class HeaderAwareChunker:
    """Chunks documents using sentence and paragraph boundaries (research papers)."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Build a chunker with optional size and overlap; defaults from config.

        :param chunk_size: Target characters per chunk; default from config.
        :param chunk_overlap: Overlap between consecutive chunks; default from config.
        """
        self._chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
        self._chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP
        self._parser = self._create_parser()

    def _create_parser(self) -> SentenceSplitter:
        """Create the SentenceSplitter used for chunking.

        :return: Configured SentenceSplitter.
        """
        return SentenceSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            paragraph_separator=PARAGRAPH_SEPARATOR,
        )

    @property
    def parser(self) -> SentenceSplitter:
        """Underlying node parser (e.g. for callers that need the parser)."""
        return self._parser

    def chunk_documents(self, documents: list[Document]) -> list[BaseNode]:
        """Split documents into nodes for embedding and retrieval.

        Trims document metadata so it does not exceed chunk_size (avoids SentenceSplitter
        ValueError when metadata length > chunk size).

        :param documents: LlamaIndex Documents to chunk.
        :return: List of nodes (chunks).
        """
        if not documents:
            logger.warning("chunk_documents called with empty documents")
            return []
        max_meta = max(0, self._chunk_size - METADATA_SIZE_MARGIN)
        trimmed = [_trim_document_metadata(d, max_meta) for d in documents]
        nodes = self._parser.get_nodes_from_documents(trimmed)
        logger.info("Chunked %d documents into %d nodes", len(documents), len(nodes))
        return nodes

    @staticmethod
    def get_header_aware_chunker(
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> SentenceSplitter:
        """Return a node parser that respects sentence and paragraph boundaries.

        Research papers often have clear sentence boundaries; SentenceSplitter
        avoids splitting mid-sentence. Chunk size is chosen to align with
        typical paragraph/section lengths.

        :param chunk_size: Target characters per chunk; default from config.
        :param chunk_overlap: Overlap between consecutive chunks; default from config.
        :return: Configured SentenceSplitter.
        """
        chunker = HeaderAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return chunker.parser

    @staticmethod
    def chunk_documents_with_options(
        documents: list[Document],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[BaseNode]:
        """Split documents into nodes (convenience: create chunker and run).

        :param documents: LlamaIndex Documents to chunk.
        :param chunk_size: Characters per chunk; default from config.
        :param chunk_overlap: Overlap between chunks; default from config.
        :return: List of nodes (chunks).
        """
        chunker = HeaderAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return chunker.chunk_documents(documents)


def get_header_aware_chunker(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> SentenceSplitter:
    """Return a node parser that respects sentence and paragraph boundaries."""
    return HeaderAwareChunker.get_header_aware_chunker(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[BaseNode]:
    """Split documents into nodes for embedding and retrieval."""
    return HeaderAwareChunker.chunk_documents_with_options(
        documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
