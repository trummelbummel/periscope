"""Configuration for the periscope RAG service.

All paths, models, ports, and API keys are configurable via environment
variables or defaults. Load with python-dotenv for .env support.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root (directory containing pyproject.toml / periscope)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# API server
PORT = int(os.environ.get("PORT", "8000"))
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
# Reload when code changes (dev); set RELOAD=1 or API_RELOAD=1
API_RELOAD = os.environ.get("API_RELOAD", os.environ.get("RELOAD", "0")).strip() in (
    "1",
    "true",
    "yes",
)

# Data paths (defined before arXiv paths that may use DATA_DIR)
DATA_DIR: Path = Path(os.environ.get("DATA_DIR", str(_PROJECT_ROOT / "data/arxiv")))
ARXIV_DATA_DIR: Path = Path(
    os.environ.get("ARXIV_DATA_DIR", str(DATA_DIR / "arxiv"))
)

# arXiv scraper configuration
ARXIV_DEFAULT_QUERY = 'abs:"context engineering" OR ti:"context engineering" AND abs:"large language model"'
ARXIV_MAX_RESULTS = 100
ARXIV_API_BASE_URL = "https://export.arxiv.org/api/query"
ARXIV_HTTP_TIMEOUT = 30.0
ARXIV_USER_AGENT = "rag-arxiv-scraper (mailto:theresa.fruhwuerth@gmail.com)"

# Document loading: default file extensions for directory reader
DEFAULT_DOCUMENT_EXTENSIONS: list[str] = [".pdf"]
CHROMA_PERSIST_DIR: Path = Path(
    os.environ.get("CHROMA_PERSIST_DIR", str(_PROJECT_ROOT / "chroma_db"))
)
# Path for persisting BM25 retriever nodes (loaded when index is loaded from Chroma)
INDEX_NODES_PATH: Path = Path(
    os.environ.get("INDEX_NODES_PATH", str(CHROMA_PERSIST_DIR / "bm25_nodes.pkl"))
)

# Embedding model for vector index (HuggingFace model id)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Generation (LLM)
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "microsoft/tapex-base-finetuned-wikisql")
GENERATION_PROMPT = os.environ.get(
    "GENERATION_PROMPT",
    "Answer the question based only on the following context.\n\nContext:\n{context_str}\n\nQuestion: {query_str}\n\nAnswer:",
)
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "") or os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Retrieval
TOP_K = int(os.environ.get("TOP_K", "10"))

# Chunking (header-aware chunker for research papers)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))

# Guardrails: abstain from generation if best similarity score below threshold
ENABLE_GUARDRAILS = os.environ.get("ENABLE_GUARDRAILS", "true").lower() in (
    "true",
    "1",
    "yes",
)
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.03"))

# Monitoring: where to write ingestion statistics
INGESTION_STATS_PATH: Path = Path(
    os.environ.get(
        "INGESTION_STATS_PATH",
        str(_PROJECT_ROOT / "monitoring" / "data" / "ingestion_stats.json"),
    )
)

