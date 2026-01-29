"""Command-line entrypoint for running the RAG API server.

Usage: python -m context_engineering_rag.main_api
"""

from __future__ import annotations

import uvicorn

from context_engineering_rag.app.api import app
from context_engineering_rag.config import API_HOST, PORT


def run() -> None:
    """Run the uvicorn server with config host and port."""
    uvicorn.run(app, host=API_HOST, port=PORT)


if __name__ == "__main__":  # pragma: no cover
    run()
