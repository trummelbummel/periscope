"""Re-export embedder from ingestion for backward compatibility."""

from context_engineering_rag.ingestion.embedder import get_embed_model, set_global_embed_model

__all__ = ["get_embed_model", "set_global_embed_model"]
