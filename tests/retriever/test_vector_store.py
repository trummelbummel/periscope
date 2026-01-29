"""Tests for vector_store (context_engineering_rag.retriever.vector_store)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from llama_index.core import VectorStoreIndex

from llama_index.core.schema import TextNode

from context_engineering_rag.retriever.vector_store import (
    ChromaIndexBuilder,
    COLLECTION_NAME,
    build_index_from_nodes,
    get_chroma_vector_store,
)


def test_collection_name_constant() -> None:
    """COLLECTION_NAME is set for Chroma."""
    assert COLLECTION_NAME == "context_engineering"


def test_chroma_index_builder_init_uses_config_when_none() -> None:
    """ChromaIndexBuilder() uses config CHROMA_PERSIST_DIR when persist_dir is None."""
    builder = ChromaIndexBuilder()
    assert builder._persist_dir is not None
    assert hasattr(builder._persist_dir, "mkdir")


def test_chroma_index_builder_init_accepts_persist_dir(tmp_path: Path) -> None:
    """ChromaIndexBuilder(persist_dir=X) uses X."""
    builder = ChromaIndexBuilder(persist_dir=tmp_path)
    assert builder._persist_dir == tmp_path


def test_get_chroma_vector_store_creates_dir_and_collection(tmp_path: Path) -> None:
    """get_chroma_vector_store creates directory and returns ChromaVectorStore."""
    store = get_chroma_vector_store(persist_dir=tmp_path)
    assert store is not None
    assert tmp_path.exists()


def test_build_index_from_nodes_returns_index(tmp_path: Path) -> None:
    """build_index_from_nodes returns VectorStoreIndex when given nodes."""
    nodes = [TextNode(text="chunk one", id_="n1"), TextNode(text="chunk two", id_="n2")]

    with (
        patch(
            "context_engineering_rag.retriever.vector_store.set_global_embed_model"
        ) as mock_set,
        patch(
            "context_engineering_rag.retriever.vector_store.VectorStoreIndex"
        ) as mock_index_class,
    ):
        mock_index_class.return_value = MagicMock(spec=VectorStoreIndex)
        index = build_index_from_nodes(nodes, persist_dir=tmp_path)

    mock_set.assert_called_once()
    mock_index_class.assert_called_once()
    assert index is mock_index_class.return_value


def test_chroma_index_builder_get_chroma_vector_store(tmp_path: Path) -> None:
    """ChromaIndexBuilder.get_chroma_vector_store returns ChromaVectorStore."""
    builder = ChromaIndexBuilder(persist_dir=tmp_path)
    store = builder.get_chroma_vector_store()
    assert store is not None
    assert tmp_path.exists()
