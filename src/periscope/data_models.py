"""Pydantic data contracts for components.

Used for data passed between components (API, retrieval, generation, etc.).
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for the retrieval-and-generation query endpoint."""

    query: str = Field(..., min_length=1, description="User question")
    top_k: int | None = Field(default=None, ge=1, le=50, description="Max retrieval count")


class RetrievedNode(BaseModel):
    """A single retrieved chunk with metadata for observability."""

    text: str = Field(..., description="Chunk text")
    score: float = Field(..., description="Relevance score")
    node_id: str | None = Field(default=None, description="Source node id")
    metadata: dict = Field(default_factory=dict, description="Extra metadata")


class QueryResponse(BaseModel):
    """Structured response with answer and supporting evidence (response quality PRD)."""

    answer: str = Field(..., description="Generated answer")
    sources: list[RetrievedNode] = Field(
        default_factory=list,
        description="Retrieved chunks used as context (supporting evidence)",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Observability metadata (scores, latency, guardrail status)",
    )
    abstained: bool = Field(
        default=False,
        description="True if generation was skipped due to guardrails",
    )


class IngestionStats(BaseModel):
    """Statistics from ingestion for monitoring (ingestion statistics PRD)."""

    document_count: int = Field(..., ge=0)
    chunk_count: int = Field(..., ge=0)
    total_chars: int = Field(..., ge=0)
    avg_chunk_size: float = Field(..., ge=0)
    paths: list[str] = Field(default_factory=list)
    index_version: str = Field(
        default="",
        description="Index version from config (INDEX_VERSION); empty for legacy stats.",
    )
    embedding_model: str = Field(
        default="",
        description="HuggingFace model id used for embeddings during ingestion",
    )
    preprocessing_config: dict = Field(
        default_factory=dict,
        description="Preprocessing options used during ingestion (remove_tables, remove_footnotes, etc.)",
    )
    chunk_size: int = Field(
        default=0,
        description="CHUNK_SIZE used during ingestion; 0 means not recorded (legacy).",
    )
    chunk_overlap: int = Field(
        default=0,
        description="CHUNK_OVERLAP used during ingestion; 0 means not recorded (legacy).",
    )


class ArxivResult(BaseModel):
    """Result of fetching a single ArXiv paper."""

    entry_id: str = Field(..., description="ArXiv ID (e.g. 2301.12345)")
    title: str = Field(...)
    summary: str | None = Field(default=None)
    pdf_url: str = Field(...)
    local_path: str | None = Field(default=None, description="Path to downloaded PDF")
