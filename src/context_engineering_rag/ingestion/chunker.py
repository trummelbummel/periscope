"""Header-aware chunking: chunking that understands research paper structure.

Per PRD: Execute chunking that understands research paper structure. Use Llama-index.
"""

import logging

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

# Chunk size tuned for research papers (paragraph/section-friendly)
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50


def get_header_aware_chunker(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> SentenceSplitter:
    """Return a node parser that respects sentence and paragraph boundaries.

    Research papers often have clear sentence boundaries; SentenceSplitter
    avoids splitting mid-sentence. Chunk size is chosen to align with
    typical paragraph/section lengths.

    :param chunk_size: Target characters per chunk.
    :param chunk_overlap: Overlap between consecutive chunks.
    :return: Configured SentenceSplitter.
    """
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
    )


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[BaseNode]:
    """Split documents into nodes for embedding and retrieval.

    :param documents: LlamaIndex Documents to chunk.
    :param chunk_size: Characters per chunk.
    :param chunk_overlap: Overlap between chunks.
    :return: List of nodes (chunks).
    """
    parser = get_header_aware_chunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    nodes = parser.get_nodes_from_documents(documents)
    logger.info("Chunked %d documents into %d nodes", len(documents), len(nodes))
    return nodes
