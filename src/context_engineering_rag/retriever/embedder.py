"""Text embedding: generate embeddings for chunked documents.

Per PRD: Generate embeddings for chunked documents. EMBEDDING_MODEL = BAAI/bge-small-en.
Use Llama-index. Low latency: use small embedding model.
"""

import logging

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from context_engineering_rag.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class Embedder:
    """HuggingFace embedding model for indexing and retrieval."""

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize with model id; default from config.

        :param model_name: Model id; default from config EMBEDDING_MODEL.
        """
        self._model_name = model_name if model_name is not None else EMBEDDING_MODEL

    def get_embed_model(self) -> HuggingFaceEmbedding:
        """Return HuggingFace embedding model instance.

        :return: HuggingFaceEmbedding instance.
        """
        return HuggingFaceEmbedding(model_name=self._model_name)

    def set_global_embed_model(self) -> None:
        """Set LlamaIndex global Settings embed_model (for pipelines)."""
        Settings.embed_model = self.get_embed_model()

    @staticmethod
    def get_embed_model_default(
        model_name: str | None = None,
    ) -> HuggingFaceEmbedding:
        """Return HuggingFace embedding model (convenience: create Embedder and run).

        :param model_name: Model id; defaults to config EMBEDDING_MODEL.
        :return: HuggingFaceEmbedding instance.
        """
        embedder = Embedder(model_name=model_name)
        return embedder.get_embed_model()

    @staticmethod
    def set_global_embed_model_default(model_name: str | None = None) -> None:
        """Set LlamaIndex global Settings embed_model (convenience: create Embedder and run).

        :param model_name: Model id; defaults to config EMBEDDING_MODEL.
        """
        embedder = Embedder(model_name=model_name)
        embedder.set_global_embed_model()


def get_embed_model(model_name: str | None = None) -> HuggingFaceEmbedding:
    """Return HuggingFace embedding model. Delegates to Embedder."""
    return Embedder.get_embed_model_default(model_name=model_name)


def set_global_embed_model(model_name: str | None = None) -> None:
    """Set LlamaIndex global Settings embed_model. Delegates to Embedder."""
    Embedder.set_global_embed_model_default(model_name=model_name)
