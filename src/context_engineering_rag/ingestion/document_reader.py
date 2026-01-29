"""Document reading: read and process PDF documents from the data directory.

Per PRD: Read and process PDF documents from the data directory. Support PDF format.
"""

import logging
from pathlib import Path

from llama_index.core import Document
from llama_index.core.readers import SimpleDirectoryReader
from pypdf import PdfReader

from context_engineering_rag.config import DATA_DIR, DEFAULT_DOCUMENT_EXTENSIONS

logger = logging.getLogger(__name__)


class DocumentReader:
    """Reads and processes PDF (and optional other) documents from a directory."""

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
        self.required_extensions = (
            required_extensions if required_extensions is not None else DEFAULT_DOCUMENT_EXTENSIONS
        )

    def read_pdf_path(self, path: Path) -> str:
        """Extract text from a single PDF file.

        :param path: Path to PDF file.
        :return: Extracted text.
        :raises FileNotFoundError: If path does not exist.
        :raises Exception: On PDF read or parse failure.
        """
        if not path.exists():
            logger.warning("PDF path does not exist: %s", path)
            raise FileNotFoundError(path)
        try:
            reader = PdfReader(str(path))
            parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    parts.append(text)
            return "\n\n".join(parts)
        except Exception as e:
            logger.exception("Failed to read PDF %s: %s", path, e)
            raise

    def load_documents(self) -> list[Document]:
        """Load documents from the configured directory as LlamaIndex Documents.

        :return: List of LlamaIndex Document objects.
        """
        if not self.directory.exists():
            logger.warning("Data directory does not exist: %s", self.directory)
            return []
        try:
            reader = SimpleDirectoryReader(
                input_dir=str(self.directory),
                required_exts=self.required_extensions,
            )
            docs = reader.load_data()
            logger.info("Loaded %d documents from %s", len(docs), self.directory)
            return docs
        except Exception as e:
            logger.exception("Failed to load documents from %s: %s", self.directory, e)
            raise

    @staticmethod
    def read_pdf_path_default(path: Path) -> str:
        """Extract text from a single PDF file (convenience; uses default directory).

        :param path: Path to PDF file.
        :return: Extracted text.
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
