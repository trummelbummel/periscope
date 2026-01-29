"""Re-export pipeline from app for backward compatibility."""

from context_engineering_rag.app.pipeline import run_query

__all__ = ["run_query"]
