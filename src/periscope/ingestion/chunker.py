"""Header-aware chunking: chunking that understands research paper structure.

Per PRD: Execute chunking that understands research paper structure. Use Llama-index.
"""

import logging

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from periscope.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)

PARAGRAPH_SEPARATOR = "\n\n"


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

        :param documents: LlamaIndex Documents to chunk.
        :return: List of nodes (chunks).
        """
        if not documents:
            logger.warning("chunk_documents called with empty documents")
            return []
        nodes = self._parser.get_nodes_from_documents(documents)
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
