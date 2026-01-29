"""Re-export app and run_server from app.api for backward compatibility."""

from context_engineering_rag.app.api import app, run_server

__all__ = ["app", "run_server"]
