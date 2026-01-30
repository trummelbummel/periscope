"""Re-export embedder from retriever for backward compatibility."""

from periscope.retriever.embedder import get_embed_model, set_global_embed_model

__all__ = ["get_embed_model", "set_global_embed_model"]
