"""Command-line entrypoint for running the RAG API server.

Usage: python -m periscope.main_api
With auto-reload (dev): RELOAD=1 python -m periscope.main_api
"""

from __future__ import annotations

from pathlib import Path

import uvicorn

from periscope.app.api import app
from periscope.config import API_HOST, API_RELOAD, PORT

# When reload is on, watch the package source directory
_SRC_DIR = Path(__file__).resolve().parent.parent


def run() -> None:
    """Run the uvicorn server with config host, port, and optional reload."""
    uvicorn.run(
        app,
        host=API_HOST,
        port=PORT,
        reload=API_RELOAD,
        reload_dirs=[str(_SRC_DIR)] if API_RELOAD else None,
    )


if __name__ == "__main__":  # pragma: no cover
    run()
