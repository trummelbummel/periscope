"""Configuration for the periscope RAG service.

All paths, models, ports, and API keys are configurable via environment
variables or defaults. Load with python-dotenv for .env support.
Lives at project root (next to pyproject.toml).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root (directory containing pyproject.toml / this config file)
_PROJECT_ROOT = Path(__file__).resolve().parent

# API server
PORT = int(os.environ.get("PORT", "8000"))
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
# Reload when code changes (dev); set RELOAD=1 or API_RELOAD=1
API_RELOAD = os.environ.get("API_RELOAD", os.environ.get("RELOAD", "0")).strip() in (
    "1",
    "true",
    "yes",
)


ARXIV_DATA_DIR: Path = Path(
    os.environ.get("ARXIV_DATA_DIR", str(_PROJECT_ROOT / "data/arxiv"))
)
# Data paths (defined before arXiv paths that may use DATA_DIR)
DATA_DIR: Path = Path(os.environ.get("DATA_DIR", str(_PROJECT_ROOT / "data/test")))

# arXiv scraper configuration
ARXIV_DEFAULT_QUERY = 'ti:"prompt optimization" AND abs:"large language model"'
ARXIV_MAX_RESULTS = 100
ARXIV_API_BASE_URL = "https://export.arxiv.org/api/query"
ARXIV_HTTP_TIMEOUT = 30.0
ARXIV_USER_AGENT = "rag-arxiv-scraper (mailto:theresa.fruhwuerth@gmail.com)"

# Document loading: default file extensions for directory reader
DEFAULT_DOCUMENT_EXTENSIONS: list[str] = [".pdf"]
# Parsed PDF cache: store extracted text, headers, tables under data/parsed so parsing does not have to be repeated
PARSED_DIR: Path = Path(
    os.environ.get("PARSED_DIR", str(_PROJECT_ROOT / "data" / "parsed"))
)
CHROMA_PERSIST_DIR: Path = Path(
    os.environ.get("CHROMA_PERSIST_DIR", str(_PROJECT_ROOT / "chroma_db"))
)
# Path for persisting BM25 retriever nodes (loaded when index is loaded from Chroma)
INDEX_NODES_PATH: Path = Path(
    os.environ.get("INDEX_NODES_PATH", str(CHROMA_PERSIST_DIR / "bm25_nodes.pkl"))
)

# Embedding model for vector index (HuggingFace model id)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/e5-small-v2")

# Generation (LLM) – Hugging Face Inference API (serverless) model for answer generation.
# Default is a small chat model hosted on Hugging Face Inference API; override via GENERATION_MODEL env var if needed.
# NOTE: Repo IDs used here must be plain Hugging Face model IDs without provider suffixes
# (no ':hf-inference'), since the HuggingFaceInferenceAPI wrapper manages providers separately.
GENERATION_MODEL = os.environ.get(
    "GENERATION_MODEL",
    "HuggingFaceTB/SmolLM3-3B",
)
GENERATION_PROMPT = os.environ.get(
    "GENERATION_PROMPT",
    """
    Think step by step and reason about the question and the context.

    ### INSTRUCTIONS
    1. Do not duplicate or hallucinate any information.
    2. Answer the question based only on the following context and strictly 
    adhere to information in the context minimizing duplication. 
    3. Structure the important information in the answer with respect to 
    the question in bullet points and return only this summary.

    ### FORMAT
    4. Answer in well structured markdown format adding Headers with method names. 
    5. Use bullet points to structure the answer.

    Follow all these instructions strictly. 

    ### INPUT
    \n\nContext:\n{context_str}\n\n
    Question: {query_str}\n\n

    ###OUTPUT
    Return the answer:
    """,
)
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "") or os.environ.get("HF_TOKEN", "")
# Max tokens for LLM answer generation; cut-off after GENERATION_MAX_TOKENS tokens.
GENERATION_MAX_TOKENS = int(os.environ.get("GENERATION_MAX_TOKENS", "1024"))

# Retrieval
TOP_K = int(os.environ.get("TOP_K", "10"))
# Reciprocal rank fusion constant for hybrid retrieval (BM25 + vector)
RRF_K = int(os.environ.get("RRF_K", "60"))
# Chroma collection name for the vector index
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "context_engineering")

# Chunking (Markdown then SentenceSplitter via LlamaIndex IngestionPipeline)
# SentenceSplitter uses token counts; chunker result is at most CHUNK_SIZE tokens per chunk.
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "20"))
PARAGRAPH_SEPARATOR = os.environ.get("PARAGRAPH_SEPARATOR", "#")
# MarkdownNodeParser configuration: whether to include header metadata and prev/next relationships,
# and which separator to use for the header_path metadata field.
MARKDOWN_INCLUDE_METADATA = (
    os.environ.get("MARKDOWN_INCLUDE_METADATA", "false").strip().lower()
    in ("1", "true", "yes")
)
MARKDOWN_INCLUDE_PREV_NEXT_REL = (
    os.environ.get("MARKDOWN_INCLUDE_PREV_NEXT_REL", "false").strip().lower()
    in ("1", "true", "yes")
)
MARKDOWN_HEADER_PATH_SEPARATOR = os.environ.get(
    "MARKDOWN_HEADER_PATH_SEPARATOR", "/"
)

# Preprocessing during ingestion: remove noise (tables, footnotes, citations, references)
def _truthy(s: str) -> bool:
    return (s or "").strip().lower() in ("1", "true", "yes")

PREPROCESS_REMOVE_TABLES = _truthy(os.environ.get("PREPROCESS_REMOVE_TABLES", "true"))
PREPROCESS_REMOVE_FOOTNOTES = _truthy(os.environ.get("PREPROCESS_REMOVE_FOOTNOTES", "true"))
PREPROCESS_REMOVE_INLINE_CITATIONS = _truthy(os.environ.get("PREPROCESS_REMOVE_INLINE_CITATIONS", "true"))
PREPROCESS_REMOVE_REFERENCE_SECTION = _truthy(os.environ.get("PREPROCESS_REMOVE_REFERENCE_SECTION", "true"))

# Guardrails: abstain from generation if best similarity score below threshold
ENABLE_GUARDRAILS = os.environ.get("ENABLE_GUARDRAILS", "false").strip().lower() in (
    "true",
    "1",
    "yes",
)
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.5"))

# Index version: used in ingestion stats, retrieval evaluation, and Chroma collection metadata
INDEX_VERSION = os.environ.get("INDEX_VERSION", "1")

# Monitoring: where to write ingestion statistics
INGESTION_STATS_PATH: Path = Path(
    os.environ.get(
        "INGESTION_STATS_PATH",
        str(_PROJECT_ROOT / "monitoring" / "data" / "ingestion_stats.json"),
    )
)

# Retrieval experiment (monitoring): max nodes and questions per chunk
RETRIEVAL_EXPERIMENT_MAX_NODES = int(
    os.environ.get("RETRIEVAL_EXPERIMENT_MAX_NODES", "10")
)
RETRIEVAL_EXPERIMENT_NUM_QUESTIONS_PER_CHUNK = int(
    os.environ.get("RETRIEVAL_EXPERIMENT_NUM_QUESTIONS_PER_CHUNK", "1")
)

