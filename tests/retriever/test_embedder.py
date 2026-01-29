"""Tests for embedder (context_engineering_rag.retriever.embedder)."""

from unittest.mock import patch

from context_engineering_rag.retriever.embedder import (
    Embedder,
    get_embed_model,
    set_global_embed_model,
)


def test_embedder_init_uses_config_default() -> None:
    """Embedder() uses config EMBEDDING_MODEL when model_name is None."""
    embedder = Embedder()
    assert embedder._model_name is not None
    assert "bge" in embedder._model_name.lower() or "small" in embedder._model_name.lower()


def test_embedder_init_accepts_model_name() -> None:
    """Embedder(model_name=X) uses X."""
    embedder = Embedder(model_name="custom/model")
    assert embedder._model_name == "custom/model"


def test_get_embed_model_returns_huggingface_embedding() -> None:
    """get_embed_model returns a HuggingFaceEmbedding instance."""
    model = get_embed_model(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert model is not None
    assert hasattr(model, "model_name")


def test_get_embed_model_with_custom_name() -> None:
    """get_embed_model(model_name=X) uses that model."""
    model = get_embed_model(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert model is not None
    assert "MiniLM" in model.model_name or "all-MiniLM" in model.model_name


def test_set_global_embed_model_sets_settings() -> None:
    """set_global_embed_model sets LlamaIndex Settings.embed_model."""
    from llama_index.core import Settings

    set_global_embed_model(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert Settings.embed_model is not None


def test_embedder_get_embed_model() -> None:
    """Embedder.get_embed_model returns HuggingFace embedding."""
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    model = embedder.get_embed_model()
    assert model is not None
    assert "MiniLM" in model.model_name or "all-MiniLM" in model.model_name
