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
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INDEX_NODES_PATH,
    INGESTION_STATS_PATH,
    PORT,
    PREPROCESS_REMOVE_FOOTNOTES,
    PREPROCESS_REMOVE_INLINE_CITATIONS,
    PREPROCESS_REMOVE_REFERENCE_SECTION,
    PREPROCESS_REMOVE_TABLES,
    TOP_K,
)
from periscope.ingestion import IngestionPipeline, NoDocumentsError
from periscope.models import IngestionStats, QueryRequest, QueryResponse
from periscope.monitoring import read_ingestion_stats
from periscope.pipeline import run_query
from periscope.vector_store import load_bm25_nodes, load_index_from_chroma

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


def _current_pipeline_config() -> dict:
    """Current pipeline config fingerprint for comparison with persisted stats."""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "preprocessing_config": {
            "remove_tables": PREPROCESS_REMOVE_TABLES,
            "remove_footnotes": PREPROCESS_REMOVE_FOOTNOTES,
            "remove_inline_citations": PREPROCESS_REMOVE_INLINE_CITATIONS,
            "remove_reference_section": PREPROCESS_REMOVE_REFERENCE_SECTION,
        },
    }


def _pipeline_config_matches(stats: IngestionStats) -> bool:
    """True if persisted stats match current pipeline config (no re-ingest needed)."""
    current = _current_pipeline_config()
    if stats.embedding_model != current["embedding_model"]:
        return False
    if stats.chunk_size != 0 and stats.chunk_size != current["chunk_size"]:
        return False
    if stats.chunk_overlap != 0 and stats.chunk_overlap != current["chunk_overlap"]:
        return False
    if stats.preprocessing_config != current["preprocessing_config"]:
        return False
    return True


def _ensure_index() -> tuple:
    """Load or build index: try persisted Chroma + BM25 nodes first if config matches, else ingest and persist."""
    global _vector_index
    global _bm25_nodes
    if _vector_index is not None and _bm25_nodes is not None:
        return _vector_index, _bm25_nodes
    # Try loading persisted index (Chroma + BM25 nodes)
    vector_index = load_index_from_chroma(CHROMA_PERSIST_DIR)
    bm25_nodes = load_bm25_nodes(INDEX_NODES_PATH)
    if vector_index is not None and bm25_nodes is not None and len(bm25_nodes) > 0:
        persisted_stats = read_ingestion_stats(INGESTION_STATS_PATH)
        if persisted_stats is not None and _pipeline_config_matches(persisted_stats):
            logger.info("Using persisted index (Chroma + %d BM25 nodes)", len(bm25_nodes))
            _vector_index = vector_index
            _bm25_nodes = bm25_nodes
            return _vector_index, _bm25_nodes
        logger.info(
            "Pipeline config changed or no persisted stats; re-ingesting (embedding_model=%s, chunk_size=%s, preprocessing=%s)",
            EMBEDDING_MODEL,
            CHUNK_SIZE,
            _current_pipeline_config().get("preprocessing_config"),
        )
    # Full ingestion via pipeline (load → preprocess → chunk → stats → index → persist)
    try:
        pipeline = IngestionPipeline()
        result = pipeline.run()
    except NoDocumentsError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from e
    _vector_index = result.index
    _bm25_nodes = result.nodes
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
        min_perf_improvement=request.min_perf_improvement,
    )


@app.post("/ingest")
def ingest() -> dict:
    """Trigger ingestion: load PDFs from data dir, chunk, embed, store. Returns stats."""
    _ensure_index_or_raise(500, "Ingest failed: %s")
    return {"status": "ok", "message": "Index built from data directory"}
