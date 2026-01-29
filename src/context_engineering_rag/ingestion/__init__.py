"""Ingestion: document loading, chunking, and table extraction."""

from context_engineering_rag.ingestion.chunker import chunk_documents, get_header_aware_chunker
from context_engineering_rag.ingestion.document_reader import (
    load_documents_from_directory,
    read_pdf_path,
)
from context_engineering_rag.ingestion.table_extractor import PdfTableExtractor

__all__ = [
    "chunk_documents",
    "get_header_aware_chunker",
    "load_documents_from_directory",
    "read_pdf_path",
    "PdfTableExtractor",
]
