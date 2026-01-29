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


def get_chroma_vector_store(persist_dir: Path | None = None) -> ChromaVectorStore:
    """Create ChromaDB vector store with persistent directory.

    :param persist_dir: Directory for Chroma persistence; default from config.
    :return: ChromaVectorStore instance.
    """
    dir_path = persist_dir if persist_dir is not None else CHROMA_PERSIST_DIR
    dir_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(dir_path))
    collection = client.get_or_create_collection(
        name="context_engineering",
        metadata={"description": "Context engineering RAG chunks"},
    )
    return ChromaVectorStore(chroma_collection=collection)


def build_index_from_nodes(
    nodes: list[BaseNode],
    persist_dir: Path | None = None,
) -> VectorStoreIndex:
    """Build a VectorStoreIndex from nodes using ChromaDB.

    Sets global embed model and builds index. Caller should have set
    embed model via set_global_embed_model if not using default.

    :param nodes: Chunk nodes to embed and store.
    :param persist_dir: Chroma persist directory; default from config.
    :return: VectorStoreIndex for retrieval.
    """
    set_global_embed_model()
    vector_store = get_chroma_vector_store(persist_dir=persist_dir)
    index = VectorStoreIndex(
        nodes=nodes,
        vector_store=vector_store,
        show_progress=True,
    )
    logger.info("Built index from %d nodes", len(nodes))
    return index
