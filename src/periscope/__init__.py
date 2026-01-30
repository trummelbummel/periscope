"""Context Engineering RAG package.

This module keeps imports lightweight so that `import periscope` works in tests
without pulling in the API server. It also exposes a `config` attribute that
re-exports the root-level config module via `periscope.config`.
"""

from periscope import config  # type: ignore[import]  # re-export of root config via stub
from periscope.models import QueryRequest, QueryResponse

__all__ = [
    "config",
    "QueryRequest",
    "QueryResponse",
]
