"""Vector database storage: store embedded chunks for retrieval.

Per PRD: ChromaDB for PoC. CHROMA_PERSIST_DIR = project_root/chroma_db.
"""

import logging
from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from context_engineering_rag.config import CHROMA_PERSIST_DIR
from context_engineering_rag.embedder import set_global_embed_model

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
            metadata={"description": "Context engineering RAG chunks"},
        )
        return ChromaVectorStore(chroma_collection=collection)

    def build_index_from_nodes(self, nodes: list[BaseNode]) -> VectorStoreIndex:
        """Build a VectorStoreIndex from nodes using ChromaDB.

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
        logger.info("Built index from %d nodes", len(nodes))
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
