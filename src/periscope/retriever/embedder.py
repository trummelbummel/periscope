"""Text embedding: generate embeddings for chunked documents.

Per PRD: Generate embeddings for chunked documents. EMBEDDING_MODEL = BAAI/bge-small-en.
Use Llama-index. Low latency: use small embedding model.
"""

import asyncio
import logging
from io import BytesIO
from typing import List, Union

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from periscope.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Placeholder for invalid tokenizer input so we don't raise TypeError
_SAFE_PLACEHOLDER = " "


def _sanitize_texts_for_embedding(texts: List[str]) -> List[str]:
    """Ensure every item is a non-empty string so the tokenizer does not raise TypeError.

    SentenceTransformer tokenizer expects TextEncodeInput (str or tuple of str).
    Replaces None, non-str, or empty/whitespace with a single space.
    """
    out: List[str] = []
    for t in texts:
        if isinstance(t, str) and t.strip():
            out.append(t)
        else:
            out.append(_SAFE_PLACEHOLDER)
    return out


def _sanitize_embed_inputs(
    inputs: List[Union[str, BytesIO]],
) -> List[Union[str, BytesIO]]:
    """Ensure every text input is a non-empty string; leave BytesIO (images) unchanged.

    Catches all code paths into _embed so the tokenizer never sees invalid input.
    """
    out: List[Union[str, BytesIO]] = []
    for t in inputs:
        if isinstance(t, BytesIO):
            out.append(t)
        elif isinstance(t, str) and t.strip():
            out.append(t)
        else:
            out.append(_SAFE_PLACEHOLDER)
    return out


class SafeHuggingFaceEmbedding(HuggingFaceEmbedding):
    """HuggingFace embedder that sanitizes text inputs before tokenization.

    Avoids TypeError from the tokenizer when inputs are None, non-string, or empty.
    Sanitizes at _embed so every code path (batch, query, single text) is covered.
    """

    def _embed(
        self,
        inputs: List[Union[str, BytesIO]],
        prompt_name: str | None = None,
    ) -> List[List[float]]:
        sanitized = _sanitize_embed_inputs(inputs)
        return super()._embed(sanitized, prompt_name)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        sanitized = _sanitize_texts_for_embedding(texts)
        return super()._get_text_embeddings(sanitized)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        sanitized = _sanitize_texts_for_embedding(texts)
        return await asyncio.to_thread(self._get_text_embeddings, sanitized)


class Embedder:
    """HuggingFace embedding model for indexing and retrieval."""

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize with model id; default from config.

        :param model_name: Model id; default from config EMBEDDING_MODEL.
        """
        self._model_name = model_name if model_name is not None else EMBEDDING_MODEL

    def get_embed_model(self) -> HuggingFaceEmbedding:
        """Return HuggingFace embedding model instance (with tokenization-safe wrapper).

        :return: SafeHuggingFaceEmbedding instance.
        """
        return SafeHuggingFaceEmbedding(model_name=self._model_name)

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
