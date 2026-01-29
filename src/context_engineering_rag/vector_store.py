"""Re-export vector store from retriever for backward compatibility."""

from context_engineering_rag.retriever.vector_store import (
    build_index_from_nodes,
    get_chroma_vector_store,
)

__all__ = ["build_index_from_nodes", "get_chroma_vector_store"]
