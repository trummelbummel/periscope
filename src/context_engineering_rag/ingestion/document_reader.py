"""Document reading: read and process PDF documents from the data directory.

Per PRD: Read and process PDF documents from the data directory. Support PDF format.
"""

import logging
from pathlib import Path

from llama_index.core import Document
from llama_index.core.readers import SimpleDirectoryReader
from pypdf import PdfReader

from context_engineering_rag.config import DATA_DIR

logger = logging.getLogger(__name__)


def read_pdf_path(path: Path) -> str:
    """Extract text from a single PDF file.

    :param path: Path to PDF file.
    :return: Extracted text.
    """
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def load_documents_from_directory(
    directory: Path | None = None,
    required_extensions: list[str] | None = None,
) -> list[Document]:
    """Load PDF (and optional other) documents from a directory as LlamaIndex Documents.

    :param directory: Directory to read from; defaults to config DATA_DIR.
    :param required_extensions: e.g. [".pdf"]; default [".pdf"].
    :return: List of LlamaIndex Document objects.
    """
    dir_path = directory if directory is not None else DATA_DIR
    exts = required_extensions if required_extensions is not None else [".pdf"]
    if not dir_path.exists():
        logger.warning("Data directory does not exist: %s", dir_path)
        return []
    try:
        reader = SimpleDirectoryReader(
            input_dir=str(dir_path),
            required_extensions=exts,
        )
        return reader.load_data()
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to load documents from %s: %s", dir_path, e)
        raise
