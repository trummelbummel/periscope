"""Re-export vector store builder for context_engineering_rag.vector_store."""

from context_engineering_rag.retriever.vector_store import build_index_from_nodes

__all__ = ["build_index_from_nodes"]
