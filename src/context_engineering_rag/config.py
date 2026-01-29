"""Configuration for the context engineering RAG service.

All paths, models, ports, and API keys are configurable via environment
variables or defaults. Load with python-dotenv for .env support.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root (directory containing pyproject.toml / context-engineering-rag)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# API server
PORT = int(os.environ.get("PORT", "8000"))
API_HOST = os.environ.get("API_HOST", "0.0.0.0")

# Data paths
ARXIV_DATA_DIR: Path = Path(
    os.environ.get("ARXIV_DATA_DIR", str(_PROJECT_ROOT / "data" / "arxiv"))
)
DATA_DIR: Path = Path(os.environ.get("DATA_DIR", str(_PROJECT_ROOT / "data")))
CHROMA_PERSIST_DIR: Path = Path(
    os.environ.get("CHROMA_PERSIST_DIR", str(_PROJECT_ROOT / "chroma_db"))
)

# Embedding model (small model for low latency per PRD)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Generation (LLM)
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "gpt-4o")
GENERATION_PROMPT = os.environ.get(
    "GENERATION_PROMPT", "Answer the question based only on the following context.\n\nContext:\n{context_str}\n\nQuestion: {query_str}\n\nAnswer:"
)

# OpenAI API key (for generation)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Retrieval
TOP_K = int(os.environ.get("TOP_K", "10"))

# Guardrails: abstain from generation if best similarity score below this
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.5"))

# Monitoring: where to write ingestion statistics
INGESTION_STATS_PATH: Path = Path(
    os.environ.get("INGESTION_STATS_PATH", str(_PROJECT_ROOT / "ingestion_stats.json"))
)
