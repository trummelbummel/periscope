"""Ingestion pipeline: run load → preprocess → chunk → index → persist → stats in sequence.

Index build stores embeddings in Chroma at first creation; BM25 nodes are persisted to disk.
Ingestion stats are computed and written only after index and BM25 persist so they reflect
the persisted index and embedding data. Uses configuration from config.py.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode

from periscope.models import IngestionStats

from periscope.config import (
    ARXIV_DATA_DIR,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    DEFAULT_DOCUMENT_EXTENSIONS,
    EMBEDDING_MODEL,
    INDEX_NODES_PATH,
    INDEX_VERSION,
    INGESTION_STATS_PATH,
    PREPROCESS_REMOVE_FOOTNOTES,
    PREPROCESS_REMOVE_INLINE_CITATIONS,
    PREPROCESS_REMOVE_REFERENCE_SECTION,
    PREPROCESS_REMOVE_TABLES,
)
from periscope.ingestion.chunker import chunk_documents
from periscope.ingestion.document_reader import load_documents_from_directory
from periscope.ingestion.preprocessor import PreprocessingConfig, preprocess_documents
from periscope.retriever.embedder import set_global_embed_model
from periscope.monitoring import compute_ingestion_stats, write_ingestion_stats
from periscope.vector_store import build_index_from_nodes, persist_bm25_nodes

logger = logging.getLogger(__name__)


class NoDocumentsError(ValueError):
    """Raised when the pipeline finds no documents in the configured data directories."""


@dataclass
class IngestionResult:
    """Result of a successful ingestion run."""

    index: VectorStoreIndex
    nodes: list[BaseNode]
    stats: IngestionStats


class IngestionPipeline:
    """Runs ingestion steps sequentially using config.py settings."""

    def __init__(
        self,
        *,
        data_dir: Path | None = None,
        arxiv_data_dir: Path | None = None,
        required_extensions: list[str] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        embedding_model: str | None = None,
        chroma_persist_dir: Path | None = None,
        index_nodes_path: Path | None = None,
        ingestion_stats_path: Path | None = None,
    ) -> None:
        """Initialize pipeline with optional overrides; omitted values use config.py."""
        self._data_dir = data_dir if data_dir is not None else DATA_DIR
        self._arxiv_data_dir = arxiv_data_dir if arxiv_data_dir is not None else ARXIV_DATA_DIR
        self._required_extensions = required_extensions or DEFAULT_DOCUMENT_EXTENSIONS
        self._chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
        self._chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP
        self._preprocessing_config = preprocessing_config or PreprocessingConfig(
            remove_tables=PREPROCESS_REMOVE_TABLES,
            remove_footnotes=PREPROCESS_REMOVE_FOOTNOTES,
            remove_inline_citations=PREPROCESS_REMOVE_INLINE_CITATIONS,
            remove_reference_section=PREPROCESS_REMOVE_REFERENCE_SECTION,
        )
        self._embedding_model = embedding_model or EMBEDDING_MODEL
        self._chroma_persist_dir = chroma_persist_dir if chroma_persist_dir is not None else CHROMA_PERSIST_DIR
        self._index_nodes_path = index_nodes_path if index_nodes_path is not None else INDEX_NODES_PATH
        self._ingestion_stats_path = ingestion_stats_path if ingestion_stats_path is not None else INGESTION_STATS_PATH

    def run(self) -> IngestionResult:
        """Execute ingestion steps: load → preprocess → chunk → index (store embeddings) → persist BM25 → stats.

        Embeddings are stored in Chroma at first index creation. Stats are written after persist
        so they reflect the persisted index and embedding data.

        :return: IngestionResult with index, nodes, and stats.
        :raises NoDocumentsError: If no documents are found in data_dir or arxiv_data_dir.
        """
        logger.info(
            "Starting ingestion pipeline data_dir=%s embedding_model=%s",
            self._data_dir,
            self._embedding_model,
        )
        set_global_embed_model()

        docs = load_documents_from_directory(
            directory=self._data_dir,
            required_extensions=self._required_extensions,
        )
        logger.info("Loaded %d documents from %s", len(docs), self._data_dir)
        if not docs:
            raise NoDocumentsError(
                "No documents in data directory. Add PDFs to data/ or data/arxiv/."
            )

        docs = preprocess_documents(docs, self._preprocessing_config)

        # Build mapping from document id to file_path so we can re-attach it
        # to all derived chunk nodes (for UI source display) even if intermediate
        # transforms drop document-level metadata.
        doc_file_paths: dict[str, str] = {}
        for doc in docs:
            file_path = (doc.metadata or {}).get("file_path")
            if not file_path:
                continue
            doc_id = getattr(doc, "doc_id", getattr(doc, "id_", None))
            if doc_id is not None:
                doc_file_paths[str(doc_id)] = str(file_path)

        nodes = chunk_documents(
            docs,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

        # Re-attach file_path metadata to chunk nodes based on their ref_doc_id
        # so that retrieval responses can surface document sources in the UI.
        for node in nodes:
            try:
                # Some node types expose ref_doc_id; fall back to existing metadata.
                ref_id = getattr(node, "ref_doc_id", None)
                if not ref_id:
                    continue
                file_path = doc_file_paths.get(str(ref_id))
                if not file_path:
                    continue
                meta = dict(getattr(node, "metadata", {}) or {})
                # Do not overwrite an existing file_path if already present.
                if "file_path" not in meta:
                    meta["file_path"] = file_path
                    node.metadata = meta  # type: ignore[attr-defined]
            except Exception:
                # Best-effort; metadata attachment should never break ingestion.
                logger.debug(
                    "Could not attach file_path metadata to node %s",
                    getattr(node, "node_id", None),
                )

        # For monitoring, record unique source document paths (if any).
        paths: list[str] = sorted(set(doc_file_paths.values()))

        logger.info(
            "Building index from %d nodes persist_dir=%s",
            len(nodes),
            self._chroma_persist_dir,
        )
        # Build index: only nodes with valid text are embedded (avoids embedder TypeError).
        index, successful_nodes = build_index_from_nodes(
            nodes, persist_dir=self._chroma_persist_dir
        )
        logger.info(
            "Index built: %d nodes embedded (Chroma), persisting BM25 nodes to %s",
            len(successful_nodes),
            self._index_nodes_path,
        )
        persist_bm25_nodes(successful_nodes, path=self._index_nodes_path)

        total_chars = sum(len(n.get_content()) for n in successful_nodes)
        stats = compute_ingestion_stats(
            document_count=len(docs),
            chunk_count=len(successful_nodes),
            total_chars=total_chars,
            paths=paths,
            embedding_model=self._embedding_model,
            preprocessing_config=self._preprocessing_config.to_dict(),
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            index_version=INDEX_VERSION,
        )
        write_ingestion_stats(stats, output_path=self._ingestion_stats_path)
        logger.info("Wrote ingestion stats to %s", self._ingestion_stats_path)

        skipped = len(nodes) - len(successful_nodes)
        logger.info(
            "Ingestion complete: %d documents, %d chunks indexed%s",
            len(docs),
            len(successful_nodes),
            f" (%d skipped)" % skipped if skipped else "",
        )
        return IngestionResult(index=index, nodes=successful_nodes, stats=stats)


def run_ingestion(
    data_dir: Path | None = None,
    arxiv_data_dir: Path | None = None,
    **kwargs: object,
) -> IngestionResult:
    """Run the ingestion pipeline with config defaults and optional overrides.

    :param data_dir: Override DATA_DIR.
    :param arxiv_data_dir: Override ARXIV_DATA_DIR.
    :param kwargs: Passed to IngestionPipeline constructor for other overrides.
    :return: IngestionResult with index, nodes, and stats.
    """
    pipeline = IngestionPipeline(
        data_dir=data_dir,
        arxiv_data_dir=arxiv_data_dir,
        **kwargs,
    )
    return pipeline.run()
