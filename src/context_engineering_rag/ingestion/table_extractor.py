"""Table extraction from PDFs using pdfplumber.

Extracts tables per page and attaches them as metadata to the appropriate
section (page). Produces one Document per page so tables are associated
with the section they came from.
"""

import logging
from pathlib import Path

import pdfplumber
from llama_index.core import Document

logger = logging.getLogger(__name__)

# Type for a single table: list of rows, each row is list of cell strings
TableData = list[list[str | None]]


class PdfTableExtractor:
    """Extracts tables from PDFs with pdfplumber and attaches them to page-level sections."""

    def extract_tables_from_pdf(self, path: Path) -> dict[int, list[TableData]]:
        """Extract all tables from a PDF by page number.

        :param path: Path to PDF file.
        :return: Mapping page_number -> list of tables (each table is list of rows, row = list of cells).
        :raises FileNotFoundError: If path does not exist.
        """
        if not path.exists():
            logger.warning("PDF path does not exist: %s", path)
            raise FileNotFoundError(path)
        tables_by_page: dict[int, list[TableData]] = {}
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    raw_tables = page.extract_tables()
                    if not raw_tables:
                        tables_by_page[i] = []
                        continue
                    # Normalize: ensure cells are str | None for JSON-serializable metadata
                    normalized = []
                    for table in raw_tables:
                        row_list: TableData = []
                        for row in table or []:
                            row_list.append(
                                [str(cell).strip() if cell is not None else None for cell in (row or [])]
                            )
                        normalized.append(row_list)
                    tables_by_page[i] = normalized
                    logger.debug("Page %d: extracted %d table(s)", i, len(normalized))
        except Exception as e:
            logger.exception("Failed to extract tables from PDF %s: %s", path, e)
            raise
        return tables_by_page

    def documents_from_pdf_with_tables(self, path: Path) -> list[Document]:
        """Build one Document per page with text and tables in metadata for that section.

        Each Document has:
        - text: text extracted from that page
        - metadata["file_path"]: path to the PDF
        - metadata["page_number"]: 1-based page index
        - metadata["tables"]: list of tables on that page (list of list of cell strings)

        :param path: Path to PDF file.
        :return: List of LlamaIndex Documents, one per page, with tables in metadata.
        :raises FileNotFoundError: If path does not exist.
        """
        if not path.exists():
            logger.warning("PDF path does not exist: %s", path)
            raise FileNotFoundError(path)
        docs: list[Document] = []
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    page_text = (text or "").strip()
                    raw_tables = page.extract_tables()
                    tables: list[TableData] = []
                    for table in raw_tables or []:
                        row_list: TableData = []
                        for row in table or []:
                            row_list.append(
                                [str(cell).strip() if cell is not None else None for cell in (row or [])]
                            )
                        tables.append(row_list)
                    metadata: dict = {
                        "file_path": str(path.resolve()),
                        "page_number": i,
                        "tables": tables,
                    }
                    docs.append(Document(text=page_text, metadata=metadata))
                logger.info("Built %d page-level document(s) with tables from %s", len(docs), path)
        except Exception as e:
            logger.exception("Failed to build documents with tables from PDF %s: %s", path, e)
            raise
        return docs

    def enrich_documents_with_tables(
        self, documents: list[Document], pdf_extensions: tuple[str, ...] = (".pdf",)
    ) -> list[Document]:
        """Replace PDF documents with page-level documents that include tables in metadata.

        For each Document that has a file_path in metadata and a PDF extension,
        replaces it with one Document per page, each with metadata["tables"] for that page.
        Non-PDF documents are left unchanged.

        :param documents: Documents from a reader (e.g. one per file).
        :param pdf_extensions: File extensions treated as PDF; default (".pdf",).
        :return: List of Documents; PDFs expanded to one per page with tables in metadata.
        """
        result: list[Document] = []
        for doc in documents:
            file_path = doc.metadata.get("file_path") if doc.metadata else None
            if not file_path:
                result.append(doc)
                continue
            path = Path(file_path)
            if path.suffix.lower() not in pdf_extensions:
                result.append(doc)
                continue
            if not path.exists():
                logger.warning("Skipping missing path from metadata: %s", path)
                result.append(doc)
                continue
            try:
                page_docs = self.documents_from_pdf_with_tables(path)
                result.extend(page_docs)
            except Exception as e:
                logger.exception("Enriching %s with tables failed: %s", path, e)
                result.append(doc)
        return result
