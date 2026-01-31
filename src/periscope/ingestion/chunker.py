"""Header-aware chunking: chunking that understands research paper structure.

Per PRD: Execute chunking that understands research paper structure. Use Llama-index.
Uses LlamaIndex IngestionPipeline: MarkdownNodeParser (split by headers) then SentenceSplitter
(chunk_size/chunk_overlap). Document text is markdown from PyMuPDF.
"""

import json
import logging

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode

from periscope.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MARKDOWN_INCLUDE_METADATA,
    METADATA_SIZE_MARGIN,
)

logger = logging.getLogger(__name__)


def _metadata_byte_size(metadata: dict) -> int:
    """Approximate size of metadata when serialized (e.g. for chunker limit)."""
    return len(json.dumps(metadata, ensure_ascii=False))


def _trim_document_metadata(doc: Document, max_metadata_size: int) -> Document:
    """Return a document with metadata capped so serialized size <= max_metadata_size.

    Keeps file_path; truncates or drops headers/tables if needed so metadata fits.
    Avoids parser ValueError when metadata length > chunk_size.
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


def _make_chunking_transformations(
    chunk_size: int,
    chunk_overlap: int,
) -> list[NodeParser]:
    """Build MarkdownNodeParser then SentenceSplitter for IngestionPipeline.

    Markdown splits by headers (config: MARKDOWN_INCLUDE_METADATA).
    SentenceSplitter uses chunk_size/chunk_overlap in tokens (config: CHUNK_SIZE, CHUNK_OVERLAP);
    chunker output is at most chunk_size tokens per node.
    """
    markdown_parser = MarkdownNodeParser.from_defaults(
        include_metadata=MARKDOWN_INCLUDE_METADATA,
    )
    sentence_splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return [markdown_parser, sentence_splitter]


class HeaderAwareChunker:
    """Chunks documents via IngestionPipeline: MarkdownNodeParser then SentenceSplitter."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Build a chunker with config from config.py.

        :param chunk_size: Max tokens per chunk (SentenceSplitter) and metadata cap; default CHUNK_SIZE (285).
        :param chunk_overlap: SentenceSplitter overlap in tokens; default CHUNK_OVERLAP from config.
        """
        self._chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
        self._chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP
        transformations = _make_chunking_transformations(
            self._chunk_size, self._chunk_overlap
        )
        self._pipeline = IngestionPipeline(transformations=transformations)

    @property
    def parser(self) -> NodeParser:
        """First node parser in the pipeline (MarkdownNodeParser)."""
        return self._pipeline.transformations[0]

    @property
    def pipeline(self) -> IngestionPipeline:
        """LlamaIndex IngestionPipeline (MarkdownNodeParser then SentenceSplitter)."""
        return self._pipeline

    def chunk_documents(self, documents: list[Document]) -> list[BaseNode]:
        """Split documents into nodes: markdown sections then sentence-sized chunks.

        Trims document metadata so it does not exceed chunk_size (avoids parser
        errors when metadata length > chunk size).

        :param documents: LlamaIndex Documents to chunk (text should be markdown).
        :return: List of nodes (markdown sections split by SentenceSplitter).
        """
        if not documents:
            logger.warning("chunk_documents called with empty documents")
            return []
        max_meta = max(0, self._chunk_size - METADATA_SIZE_MARGIN)
        trimmed = [_trim_document_metadata(d, max_meta) for d in documents]
        nodes = self._pipeline.run(documents=trimmed)
        logger.info("Chunked %d documents into %d nodes", len(documents), len(nodes))
        return nodes

    @staticmethod
    def get_header_aware_chunker(
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> NodeParser:
        """Return the MarkdownNodeParser (first stage of the chunking pipeline).

        Full chunking is done by IngestionPipeline (Markdown then SentenceSplitter).
        Use HeaderAwareChunker.chunk_documents or .pipeline for the full pipeline.

        :param chunk_size: Passed to SentenceSplitter and metadata cap; default from config.
        :param chunk_overlap: Passed to SentenceSplitter; default from config.
        :return: MarkdownNodeParser (first transformation).
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
) -> NodeParser:
    """Return a node parser that splits by markdown headers."""
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
