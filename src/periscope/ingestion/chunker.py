"""Header-aware chunking: chunking that understands research paper structure.

Per PRD: Execute chunking that understands research paper structure. Use Llama-index.
Uses LlamaIndex IngestionPipeline: MarkdownNodeParser (split by headers) then SentenceSplitter
(chunk_size/chunk_overlap). Document text is markdown from PyMuPDF.
"""

import logging

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode

from periscope.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MARKDOWN_HEADER_PATH_SEPARATOR,
    MARKDOWN_INCLUDE_METADATA,
    MARKDOWN_INCLUDE_PREV_NEXT_REL,
)

logger = logging.getLogger(__name__)


def _make_chunking_transformations(
    chunk_size: int,
    chunk_overlap: int,
) -> list[NodeParser]:
    """Build MarkdownNodeParser then SentenceSplitter for IngestionPipeline.

    Markdown splits by headers; MarkdownNodeParser header options are driven by config:
    - MARKDOWN_INCLUDE_METADATA toggles header_path metadata on nodes.
    - MARKDOWN_INCLUDE_PREV_NEXT_REL toggles prev/next header relationships.
    - MARKDOWN_HEADER_PATH_SEPARATOR controls the header_path separator (default '/').
    SentenceSplitter uses chunk_size/chunk_overlap in tokens (config: CHUNK_SIZE, CHUNK_OVERLAP);
    chunker output is at most chunk_size tokens per node.
    """
    markdown_parser = MarkdownNodeParser.from_defaults(
        include_metadata=MARKDOWN_INCLUDE_METADATA,
        include_prev_next_rel=MARKDOWN_INCLUDE_PREV_NEXT_REL,
        header_path_separator=MARKDOWN_HEADER_PATH_SEPARATOR,
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

        :param documents: LlamaIndex Documents to chunk (text should be markdown).
        :return: List of nodes (markdown sections split by SentenceSplitter).
        """
        if not documents:
            logger.warning("chunk_documents called with empty documents")
            return []
        nodes = self._pipeline.run(documents=documents)
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
        :param chunk_size: Max tokens per chunk (SentenceSplitter); default from config.
        :param chunk_overlap: Overlap between chunks in tokens; default from config.
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
