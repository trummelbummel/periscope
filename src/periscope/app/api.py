"""REST API service: expose endpoints for retrieval and generation workflow.

Per PRD: FastAPI, PORT = 8000, API_HOST = '0.0.0.0'.
"""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from periscope.config import (
    API_HOST,
    ARXIV_DATA_DIR,
    CHROMA_PERSIST_DIR,
    DATA_DIR,
    INDEX_NODES_PATH,
    PORT,
    TOP_K,
)
from periscope.ingestion import (
    chunk_documents,
    load_documents_from_directory,
)
from periscope.vector_store import (
    build_index_from_nodes,
    load_bm25_nodes,
    load_index_from_chroma,
    persist_bm25_nodes,
)
from periscope.embedder import set_global_embed_model
from periscope.models import QueryRequest, QueryResponse
from periscope.monitoring import (
    compute_ingestion_stats,
    write_ingestion_stats as persist_ingestion_stats,
)
from periscope.pipeline import run_query

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Periscope RAG API",
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

# UI: static files under /ui, redirect / to /ui/
_UI_DIR = Path(__file__).resolve().parent.parent.parent / "ui"
if _UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")


@app.get("/", response_model=None, include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect root to the UI."""
    return RedirectResponse(url="/ui/", status_code=302)

# In-memory state: index and nodes for BM25 (loaded on first ingest/query)
_vector_index = None
_bm25_nodes = None


def _ensure_index() -> tuple:
    """Load or build index: try persisted Chroma + BM25 nodes first, else ingest and persist."""
    global _vector_index
    global _bm25_nodes
    if _vector_index is not None and _bm25_nodes is not None:
        return _vector_index, _bm25_nodes
    # Try loading persisted index (Chroma + BM25 nodes)
    vector_index = load_index_from_chroma(CHROMA_PERSIST_DIR)
    bm25_nodes = load_bm25_nodes(INDEX_NODES_PATH)
    if vector_index is not None and bm25_nodes is not None and len(bm25_nodes) > 0:
        logger.info("Using persisted index (Chroma + %d BM25 nodes)", len(bm25_nodes))
        _vector_index = vector_index
        _bm25_nodes = bm25_nodes
        return _vector_index, _bm25_nodes
    # Full ingestion from data directory
    logger.info("Building index from data directory")
    set_global_embed_model()
    docs = load_documents_from_directory(DATA_DIR)
    if not docs:
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
    persist_bm25_nodes(nodes, INDEX_NODES_PATH)
    return _vector_index, _bm25_nodes


def _ensure_index_or_raise(status_code: int = 503, log_message: str = "Failed to load index: %s") -> tuple:
    """Call _ensure_index(); on Exception (except HTTPException) log and raise HTTPException."""
    try:
        return _ensure_index()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(log_message, e)
        raise HTTPException(status_code=status_code, detail=str(e)) from e


@app.get("/health")
def health() -> dict:
    """Health check."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run retrieval and generation for a user question."""
    vector_index, bm25_nodes = _ensure_index_or_raise(503, "Failed to load index: %s")
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
    _ensure_index_or_raise(500, "Ingest failed: %s")
    return {"status": "ok", "message": "Index built from data directory"}
