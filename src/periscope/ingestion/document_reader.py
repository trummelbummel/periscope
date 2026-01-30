"""Document reading: read and process PDF documents from the data directory.

Per PRD: Read and process PDF documents from the data directory. Support PDF format.
Uses Docling for extraction: layout-aware text, tables, and section headers in metadata.
"""

import logging
from pathlib import Path

from docling.document_converter import DocumentConverter
from llama_index.core import Document

from periscope.config import DATA_DIR, DEFAULT_DOCUMENT_EXTENSIONS

logger = logging.getLogger(__name__)


def _extract_headers_from_doc(doc) -> list[str]:
    """Collect section headers and titles from docling document texts."""
    headers: list[str] = []
    for item in getattr(doc, "texts", []) or []:
        label = getattr(item, "label", None)
        label_str = getattr(label, "value", str(label)) if label is not None else ""
        if label_str in ("section_header", "title"):
            text = getattr(item, "text", None)
            if text and isinstance(text, str) and text.strip():
                headers.append(text.strip())
    return headers


def _extract_tables_from_doc(conv_res) -> list[str]:
    """Export each table as HTML for metadata (no pandas required)."""
    doc = conv_res.document
    tables: list[str] = []
    for table in getattr(doc, "tables", []) or []:
        try:
            tables.append(table.export_to_html(doc=doc))
        except Exception as e:
            logger.debug("Could not export table to HTML: %s", e)
    return tables


def _convert_path_with_docling(path: Path):
    """Convert a single file with Docling; return ConversionResult."""
    converter = DocumentConverter()
    return converter.convert(path)


class DocumentReader:
    """Reads and processes PDF (and optional other) documents using Docling."""

    def __init__(
        self,
        directory: Path | None = None,
        required_extensions: list[str] | None = None,
    ) -> None:
        """Initialize with directory and file extensions to load.

        :param directory: Directory to read from; defaults to config DATA_DIR.
        :param required_extensions: File extensions to load; default from config.
        """
        self.directory = directory if directory is not None else DATA_DIR
        self._directory = Path(self.directory)
        self.required_extensions = (
            required_extensions if required_extensions is not None else DEFAULT_DOCUMENT_EXTENSIONS
        )

    def read_pdf_path(self, path: Path) -> str:
        """Extract text from a single PDF file (Markdown with layout and tables).

        :param path: Path to PDF file.
        :return: Extracted text (Markdown).
        :raises FileNotFoundError: If path does not exist.
        :raises Exception: On conversion failure.
        """
        if not path.exists():
            logger.warning("PDF path does not exist: %s", path)
            raise FileNotFoundError(path)
        try:
            conv_res = _convert_path_with_docling(path)
            return conv_res.document.export_to_markdown()
        except Exception as e:
            logger.exception("Failed to convert PDF %s: %s", path, e)
            raise

    def _path_to_llama_document(self, path: Path) -> Document | None:
        """Convert a single file to a LlamaIndex Document with headers and tables in metadata."""
        if not path.exists():
            return None
        try:
            conv_res = _convert_path_with_docling(path)
            doc = conv_res.document
            text = doc.export_to_markdown()
            headers = _extract_headers_from_doc(doc)
            tables = _extract_tables_from_doc(conv_res)
            metadata: dict = {
                "file_path": str(path.resolve()),
            }
            if headers:
                metadata["headers"] = headers
            if tables:
                metadata["tables"] = tables
            return Document(text=text, metadata=metadata)
        except Exception as e:
            logger.warning("Docling conversion failed for %s: %s", path, e)
            return None

    def load_documents(self) -> list[Document]:
        """Load documents from the configured directory as LlamaIndex Documents.

        Each document has text from Docling (Markdown with layout and tables) and
        metadata: file_path, headers (section headers and titles), tables (table Markdown/HTML).

        :return: List of LlamaIndex Document objects.
        """
        if not self._directory.exists():
            logger.warning("Data directory does not exist: %s", self._directory)
            return []
        paths: list[Path] = []
        for ext in self.required_extensions:
            paths.extend(self._directory.glob(f"*{ext}"))
        paths = sorted(set(paths))
        if not paths:
            logger.info("No matching files in %s (extensions: %s)", self._directory, self.required_extensions)
            return []
        docs: list[Document] = []
        for path in paths:
            if not path.is_file():
                continue
            doc = self._path_to_llama_document(path)
            if doc is not None:
                docs.append(doc)
        logger.info("Loaded %d documents from %s", len(docs), self._directory)
        return docs

    @staticmethod
    def read_pdf_path_default(path: Path) -> str:
        """Extract text from a single PDF file (convenience; uses default directory).

        :param path: Path to PDF file.
        :return: Extracted text (Markdown).
        """
        reader = DocumentReader()
        return reader.read_pdf_path(path)

    @staticmethod
    def load_documents_from_directory_default(
        directory: Path | None = None,
        required_extensions: list[str] | None = None,
    ) -> list[Document]:
        """Load PDF (and optional other) documents from a directory as LlamaIndex Documents.

        :param directory: Directory to read from; defaults to config DATA_DIR.
        :param required_extensions: e.g. [".pdf"]; default from config.
        :return: List of LlamaIndex Document objects.
        """
        reader = DocumentReader(
            directory=directory,
            required_extensions=required_extensions,
        )
        return reader.load_documents()


def read_pdf_path(path: Path) -> str:
    """Extract text from a single PDF file. Delegates to DocumentReader."""
    return DocumentReader.read_pdf_path_default(path)


def load_documents_from_directory(
    directory: Path | None = None,
    required_extensions: list[str] | None = None,
) -> list[Document]:
    """Load documents from a directory. Delegates to DocumentReader."""
    return DocumentReader.load_documents_from_directory_default(
        directory=directory, required_extensions=required_extensions
    )
