"""Context Engineering RAG: retrieval and generation for context engineering papers."""

from periscope.app.api import app
from periscope.config import (
    API_HOST,
    CHROMA_PERSIST_DIR,
    DATA_DIR,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    PORT,
    TOP_K,
)
from periscope.models import QueryRequest, QueryResponse

__all__ = [
    "app",
    "API_HOST",
    "PORT",
    "DATA_DIR",
    "CHROMA_PERSIST_DIR",
    "EMBEDDING_MODEL",
    "GENERATION_MODEL",
    "TOP_K",
    "QueryRequest",
    "QueryResponse",
]
