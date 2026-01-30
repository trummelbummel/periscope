"""Vector database storage: store embedded chunks for retrieval.

Per PRD: ChromaDB for PoC. CHROMA_PERSIST_DIR = project_root/chroma_db.
Index is persisted in Chroma; BM25 nodes are persisted to INDEX_NODES_PATH (pickle).
"""

import logging
import pickle
from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from periscope.config import CHROMA_PERSIST_DIR, INDEX_NODES_PATH, INDEX_VERSION
from periscope.embedder import set_global_embed_model

logger = logging.getLogger(__name__)

COLLECTION_NAME = "context_engineering"


class ChromaIndexBuilder:
    """Builds a VectorStoreIndex from nodes using ChromaDB."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        """Initialize with Chroma persistence directory; default from config.

        :param persist_dir: Directory for Chroma persistence; default from config.
        """
        self._persist_dir = persist_dir if persist_dir is not None else CHROMA_PERSIST_DIR

    def get_chroma_vector_store(self) -> ChromaVectorStore:
        """Create ChromaDB vector store with persistent directory.

        :return: ChromaVectorStore instance.
        """
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self._persist_dir))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": "Context engineering RAG chunks",
                "index_version": INDEX_VERSION,
            },
        )
        return ChromaVectorStore(chroma_collection=collection)

    def build_index_from_nodes(self, nodes: list[BaseNode]) -> VectorStoreIndex:
        """Build a VectorStoreIndex from nodes using ChromaDB; embeddings are stored at first creation.

        Nodes are embedded and inserted into the Chroma collection; PersistentClient persists
        to disk automatically, so the vector index and embedding data are stored on first run.

        :param nodes: Chunk nodes to embed and store.
        :return: VectorStoreIndex for retrieval.
        """
        set_global_embed_model()
        vector_store = self.get_chroma_vector_store()
        index = VectorStoreIndex(
            nodes=nodes,
            vector_store=vector_store,
            show_progress=True,
        )
        logger.info("Built index from %d nodes (embeddings stored in Chroma)", len(nodes))
        return index

    @staticmethod
    def get_chroma_vector_store_default(
        persist_dir: Path | None = None,
    ) -> ChromaVectorStore:
        """Create ChromaDB vector store (convenience: create builder and run).

        :param persist_dir: Directory for Chroma persistence; default from config.
        :return: ChromaVectorStore instance.
        """
        builder = ChromaIndexBuilder(persist_dir=persist_dir)
        return builder.get_chroma_vector_store()

    @staticmethod
    def build_index_from_nodes_default(
        nodes: list[BaseNode],
        persist_dir: Path | None = None,
    ) -> VectorStoreIndex:
        """Build a VectorStoreIndex from nodes (convenience: create builder and run).

        :param nodes: Chunk nodes to embed and store.
        :param persist_dir: Chroma persist directory; default from config.
        :return: VectorStoreIndex for retrieval.
        """
        builder = ChromaIndexBuilder(persist_dir=persist_dir)
        return builder.build_index_from_nodes(nodes)


def get_chroma_vector_store(persist_dir: Path | None = None) -> ChromaVectorStore:
    """Create ChromaDB vector store. Delegates to ChromaIndexBuilder."""
    return ChromaIndexBuilder.get_chroma_vector_store_default(
        persist_dir=persist_dir
    )


def build_index_from_nodes(
    nodes: list[BaseNode],
    persist_dir: Path | None = None,
) -> VectorStoreIndex:
    """Build VectorStoreIndex from nodes. Delegates to ChromaIndexBuilder."""
    return ChromaIndexBuilder.build_index_from_nodes_default(
        nodes, persist_dir=persist_dir
    )


def load_index_from_chroma(
    persist_dir: Path | None = None,
) -> VectorStoreIndex | None:
    """Load VectorStoreIndex from existing Chroma persistence.

    :param persist_dir: Chroma persistence directory; default from config.
    :return: VectorStoreIndex if collection exists and has documents, else None.
    """
    persist_dir = persist_dir if persist_dir is not None else CHROMA_PERSIST_DIR
    if not persist_dir.exists():
        return None
    try:
        client = chromadb.PersistentClient(path=str(persist_dir))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": "Context engineering RAG chunks",
                "index_version": INDEX_VERSION,
            },
        )
        if collection.count() == 0:
            return None
        set_global_embed_model()
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        logger.info("Loaded index from Chroma (%d documents)", collection.count())
        return index
    except Exception as e:
        logger.warning("Could not load index from Chroma: %s", e)
        return None


def persist_bm25_nodes(nodes: list[BaseNode], path: Path | None = None) -> None:
    """Persist BM25 nodes to disk (pickle) for later load.

    :param nodes: Chunk nodes used by BM25 retriever.
    :param path: Output path; default from config INDEX_NODES_PATH.
    """
    path = path if path is not None else INDEX_NODES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(nodes, f)
    logger.info("Persisted %d BM25 nodes to %s", len(nodes), path)


def load_bm25_nodes(path: Path | None = None) -> list[BaseNode] | None:
    """Load BM25 nodes from disk (pickle).

    :param path: Input path; default from config INDEX_NODES_PATH.
    :return: List of nodes if file exists and is valid, else None.
    """
    path = path if path is not None else INDEX_NODES_PATH
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            nodes = pickle.load(f)
        if not isinstance(nodes, list) or len(nodes) == 0:
            return None
        logger.info("Loaded %d BM25 nodes from %s", len(nodes), path)
        return nodes
    except Exception as e:
        logger.warning("Could not load BM25 nodes from %s: %s", path, e)
        return None
