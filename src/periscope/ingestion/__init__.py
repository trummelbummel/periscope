"""Ingestion: document loading, preprocessing, chunking, pipeline, and table extraction."""

from periscope.ingestion.chunker import chunk_documents, get_header_aware_chunker
from periscope.ingestion.document_reader import (
    load_documents_from_directory,
    read_pdf_path,
)
from periscope.ingestion.ingestion_pipeline import (
    IngestionPipeline,
    IngestionResult,
    NoDocumentsError,
    run_ingestion,
)
from periscope.ingestion.preprocessor import PreprocessingConfig, preprocess_documents
from periscope.ingestion.table_extractor import PdfTableExtractor

__all__ = [
    "chunk_documents",
    "get_header_aware_chunker",
    "IngestionPipeline",
    "IngestionResult",
    "load_documents_from_directory",
    "NoDocumentsError",
    "preprocess_documents",
    "PreprocessingConfig",
    "read_pdf_path",
    "run_ingestion",
    "PdfTableExtractor",
]
