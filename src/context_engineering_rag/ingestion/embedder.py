"""Text embedding: generate embeddings for chunked documents.

Per PRD: Generate embeddings for chunked documents. EMBEDDING_MODEL = BAAI/bge-small-en.
Use Llama-index. Low latency: use small embedding model.
"""

import logging

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from context_engineering_rag.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def get_embed_model(model_name: str | None = None) -> HuggingFaceEmbedding:
    """Return HuggingFace embedding model for indexing and retrieval.

    :param model_name: Model id; defaults to config EMBEDDING_MODEL.
    :return: HuggingFaceEmbedding instance.
    """
    name = model_name if model_name is not None else EMBEDDING_MODEL
    return HuggingFaceEmbedding(model_name=name)


def set_global_embed_model(model_name: str | None = None) -> None:
    """Set LlamaIndex global Settings embed_model (for pipelines).

    :param model_name: Model id; defaults to config EMBEDDING_MODEL.
    """
    Settings.embed_model = get_embed_model(model_name=model_name)
