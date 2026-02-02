"""Document reading: read and process PDF documents from the data directory.

Per PRD: Read and process PDF documents from the data directory. Support PDF format.
Uses PyMuPDF (fitz) for extraction: markdown text only (no per-chunk metadata).
Parsed results are cached under PARSED_DIR (default data/parsed) so parsing does not have to be repeated.
"""

import contextlib
import hashlib
import json
import logging
import os
import statistics
from pathlib import Path

import fitz
from llama_index.core import Document

from periscope.config import DATA_DIR, DEFAULT_DOCUMENT_EXTENSIONS, PARSED_DIR

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_mupdf_stderr():
    """Redirect C-level stderr (fd 2) so MuPDF ExtGState/resource warnings are not printed.

    MuPDF writes non-fatal errors (e.g. 'cannot find ExtGState resource') to stderr;
    the PDF often still opens and text extraction works. This suppresses those messages.
    """
    stderr_fd = 2
    try:
        saved_fd = os.dup(stderr_fd)
    except OSError:
        yield
        return
    devnull = None
    try:
        devnull = open(os.devnull, "w")
        os.dup2(devnull.fileno(), stderr_fd)
        yield
    finally:
        if devnull is not None:
            try:
                os.dup2(saved_fd, stderr_fd)
            except OSError:
                pass
            os.close(saved_fd)
            devnull.close()


def _cache_key(path: Path) -> str:
    """Stable cache key for a source PDF path (hash of resolved path)."""
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:32]


def _load_parsed(parsed_path: Path) -> dict | None:
    """Load parsed PDF data from JSON; return None if missing or invalid."""
    if not parsed_path.exists():
        return None
    try:
        data = json.loads(parsed_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "text" in data:
            return data
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Could not load parsed cache %s: %s", parsed_path, e)
    return None


def _save_parsed(parsed_path: Path, data: dict) -> None:
    """Write parsed PDF data to JSON."""
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_path.write_text(json.dumps(data, ensure_ascii=False, indent=0), encoding="utf-8")


def _extract_markdown_from_pdf(doc: fitz.Document) -> str:
    """Extract markdown from a PyMuPDF document (headers as ## from font-size heuristics)."""
    all_sizes: list[float] = []
    for page in doc:
        block_dict = page.get_text("dict", sort=True)
        for block in block_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = float(span.get("size", 0))
                    if size > 0:
                        all_sizes.append(size)
    median_size = statistics.median(all_sizes) if all_sizes else 12.0
    header_threshold = median_size * 1.15

    parts: list[str] = []
    for page in doc:
        block_dict = page.get_text("dict", sort=True)
        for block in block_dict.get("blocks", []):
            for line in block.get("lines", []):
                line_parts: list[str] = []
                for span in line.get("spans", []):
                    text = (span.get("text") or "").strip()
                    if not text:
                        continue
                    size = float(span.get("size", 0))
                    if size >= header_threshold:
                        line_parts.append("## " + text)
                    else:
                        line_parts.append(text)
                if line_parts:
                    parts.append(" ".join(line_parts))
            if parts and parts[-1].strip():
                parts.append("")
    return "\n".join(parts).strip()


def _open_pdf(path: Path) -> fitz.Document:
    """Open a PDF file with PyMuPDF; caller must close the document."""
    return fitz.open(path)


class DocumentReader:
    """Reads and processes PDF (and optional other) documents using PyMuPDF; caches parsed output."""

    def __init__(
        self,
        directory: Path | None = None,
        required_extensions: list[str] | None = None,
        parsed_dir: Path | None = None,
    ) -> None:
        """Initialize with directory, file extensions, and parsed cache directory.

        :param directory: Directory to read from; defaults to config DATA_DIR.
        :param required_extensions: File extensions to load; default from config.
        :param parsed_dir: Directory for cached parsed PDFs; defaults to config PARSED_DIR.
        """
        self.directory = directory if directory is not None else DATA_DIR
        self._directory = Path(self.directory)
        self.required_extensions = (
            required_extensions if required_extensions is not None else DEFAULT_DOCUMENT_EXTENSIONS
        )
        self._parsed_dir = Path(parsed_dir) if parsed_dir is not None else PARSED_DIR

    def read_pdf_path(self, path: Path) -> str:
        """Extract text from a single PDF file (from cache if available and up to date).

        :param path: Path to PDF file.
        :return: Extracted text (plain text, reading order).
        :raises FileNotFoundError: If path does not exist.
        :raises Exception: On conversion failure.
        """
        if not path.exists():
            logger.warning("PDF path does not exist: %s", path)
            raise FileNotFoundError(path)
        resolved = path.resolve()
        cached = self._parsed_dir / (_cache_key(resolved) + ".json")
        data = _load_parsed(cached)
        if data is not None and cached.stat().st_mtime >= resolved.stat().st_mtime:
            logger.debug("Using cached parse for %s", path)
            return data["text"]
        try:
            with _suppress_mupdf_stderr():
                doc = _open_pdf(path)
                try:
                    text = _extract_markdown_from_pdf(doc)
                finally:
                    doc.close()
            _save_parsed(
                cached,
                {
                    "text": text,
                    "file_path": str(resolved),
                },
            )
            return text
        except Exception as e:
            logger.exception("Failed to convert PDF %s: %s", path, e)
            raise

    def _path_to_llama_document(self, path: Path) -> Document | None:
        """Convert a single file to a LlamaIndex Document (from cache if available and up to date)."""
        if not path.exists():
            return None
        resolved = path.resolve()
        cached = self._parsed_dir / (_cache_key(resolved) + ".json")
        data = _load_parsed(cached)
        if data is not None and cached.stat().st_mtime >= resolved.stat().st_mtime:
            logger.debug("Using cached parse for %s", path)
            metadata = {"file_path": data.get("file_path", str(resolved))}
            return Document(text=data["text"], metadata=metadata)
        try:
            with _suppress_mupdf_stderr():
                doc = _open_pdf(path)
                try:
                    text = _extract_markdown_from_pdf(doc)
                finally:
                    doc.close()
            _save_parsed(
                cached,
                {
                    "text": text,
                    "file_path": str(resolved),
                },
            )
            metadata = {"file_path": str(resolved)}
            return Document(text=text, metadata=metadata)
        except Exception as e:
            logger.warning("PyMuPDF conversion failed for %s: %s", path, e)
            return None

    def load_documents(self) -> list[Document]:
        """Load documents from the configured directory as LlamaIndex Documents."""
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
        logger.info(
            "Loading %d files from %s (extensions: %s)",
            len(paths),
            self._directory,
            self.required_extensions,
        )
        docs: list[Document] = []
        for path in paths:
            if not path.is_file():
                logger.debug("Skipping non-file path: %s", path)
                continue
            logger.debug("Loading document from path: %s", path)
            doc = self._path_to_llama_document(path)
            if doc is not None:
                docs.append(doc)
            else:
                logger.debug("Skipped %s (conversion returned None)", path)
        logger.info("Loaded %d documents from %s", len(docs), self._directory)
        return docs

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
