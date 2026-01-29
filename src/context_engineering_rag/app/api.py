"""REST API service: expose endpoints for retrieval and generation workflow.

Per PRD: FastAPI, PORT = 8000, API_HOST = '0.0.0.0'.
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from context_engineering_rag.config import (
    API_HOST,
    ARXIV_DATA_DIR,
    CHROMA_PERSIST_DIR,
    DATA_DIR,
    PORT,
    TOP_K,
)
from context_engineering_rag.ingestion import (
    chunk_documents,
    load_documents_from_directory,
)
from context_engineering_rag.vector_store import build_index_from_nodes
from context_engineering_rag.embedder import set_global_embed_model
from context_engineering_rag.models import QueryRequest, QueryResponse
from context_engineering_rag.monitoring import (
    compute_ingestion_stats,
    write_ingestion_stats as persist_ingestion_stats,
)
from context_engineering_rag.pipeline import run_query

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Context Engineering RAG API",
    description="Retrieval and generation workflow for context engineering papers",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state: index and nodes for BM25 (loaded on first ingest/query)
_vector_index = None
_bm25_nodes = None


def _ensure_index() -> tuple:
    """Load or build index from data directory; return (vector_index, bm25_nodes)."""
    global _vector_index
    global _bm25_nodes
    if _vector_index is not None and _bm25_nodes is not None:
        return _vector_index, _bm25_nodes
    set_global_embed_model()
    docs = load_documents_from_directory(DATA_DIR)
    if not docs:
        # Also try arxiv data dir
        docs = load_documents_from_directory(ARXIV_DATA_DIR)
    if not docs:
        raise HTTPException(
            status_code=503,
            detail="No documents in data directory. Run ingest first or add PDFs to data/ or data/arxiv/.",
        )
    nodes = chunk_documents(docs)
    total_chars = sum(len(n.get_content()) for n in nodes)
    paths = []
    for d in docs:
        if d.metadata and "file_path" in d.metadata:
            paths.append(str(d.metadata["file_path"]))
    stats = compute_ingestion_stats(
        document_count=len(docs),
        chunk_count=len(nodes),
        total_chars=total_chars,
        paths=paths,
    )
    persist_ingestion_stats(stats)
    _vector_index = build_index_from_nodes(nodes)
    _bm25_nodes = nodes
    return _vector_index, _bm25_nodes


@app.get("/health")
def health() -> dict:
    """Health check."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run retrieval and generation for a user question."""
    try:
        vector_index, bm25_nodes = _ensure_index()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to load index: %s", e)
        raise HTTPException(status_code=503, detail=str(e)) from e
    top_k = request.top_k if request.top_k is not None else TOP_K
    return run_query(
        query=request.query,
        vector_index=vector_index,
        bm25_nodes=bm25_nodes,
        top_k=top_k,
    )


@app.post("/ingest")
def ingest() -> dict:
    """Trigger ingestion: load PDFs from data dir, chunk, embed, store. Returns stats."""
    try:
        _ensure_index()
        return {"status": "ok", "message": "Index built from data directory"}
    except HTTPException as e:
        raise
    except Exception as e:
        logger.exception("Ingest failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


def run_server(host: str | None = None, port: int | None = None) -> None:
    """Run uvicorn server (for programmatic use)."""
    import uvicorn
    h = host if host is not None else API_HOST
    p = port if port is not None else PORT
    uvicorn.run(app, host=h, port=p)
