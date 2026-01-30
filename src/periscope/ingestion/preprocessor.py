"""Preprocessing: remove noise from document text before chunking.

Removes tables, footnotes, inline citations, and reference sections
so chunking and embedding focus on main content.
"""

import logging
import re
from dataclasses import dataclass

from llama_index.core import Document

logger = logging.getLogger(__name__)


# --- Patterns -----------------------------------------------------------------

# Inline citations: [1], [2, 3], [1-5], [1, 2, 3]
_CITATION_BRACKET = re.compile(r"\[\s*\d+(?:\s*[,\-–]\s*\d+)*\s*\]")
# (Author et al., 2020), (Smith, 1999), (Smith 1999), (Smith and Jones 2000)
_CITATION_PAREN = re.compile(
    r"\(\s*[A-Z][a-zA-Z\-]*(?:\s+et\s+al\.?|\s+and\s+[A-Z][a-zA-Z\-]*)*\s*,?\s*(?:19|20)\d{2}\s*\)"
)
# Reference section headers
_REF_SECTION = re.compile(
    r"(\n\s*References?\s*\n|\n\s*Bibliography\s*\n|\n\s*Works?\s+Cited\s*\n)",
    re.IGNORECASE,
)
# Markdown-style table rows: | cell |
_TABLE_ROW = re.compile(r"^\s*\|[^\n|]+\|\s*$", re.MULTILINE)
_TABLE_SEP = re.compile(r"^\s*[-=]{2,}\s*$", re.MULTILINE)
# Footnote lines and refs
_FOOTNOTE_LINE = re.compile(r"^\s*\d+\.\s+.{1,120}\s*$", re.MULTILINE)
_FOOTNOTE_REF = re.compile(r"\b(?:see\s+)?footnote\s+\d+\b", re.IGNORECASE)
_FOOTNOTE_LABEL = re.compile(r"Footnote\s+\d+\s*:\s*", re.IGNORECASE)

_MULTI_SPACE = re.compile(r"  +")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_ALIGNED_COLUMNS = re.compile(r"^[ \t]{2,}.{0,200}$", re.MULTILINE)


# --- Config -------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    """Options for what to strip during preprocessing."""

    remove_tables: bool = True
    remove_footnotes: bool = True
    remove_inline_citations: bool = True
    remove_reference_section: bool = True

    def to_dict(self) -> dict:
        """Serialize for ingestion stats."""
        return {
            "remove_tables": self.remove_tables,
            "remove_footnotes": self.remove_footnotes,
            "remove_inline_citations": self.remove_inline_citations,
            "remove_reference_section": self.remove_reference_section,
        }


# --- Cleaners (internal) -------------------------------------------------------


def _strip_reference_section(text: str) -> str:
    """Keep only content before References/Bibliography/Works Cited."""
    parts = _REF_SECTION.split(text, maxsplit=1)
    if len(parts) >= 2:
        logger.debug("Stripped reference section (kept %d chars)", len(parts[0].rstrip()))
        return parts[0].rstrip()
    return text


def _strip_inline_citations(text: str) -> str:
    """Remove [1], (Author et al., 2020) style citations and collapse spaces."""
    out = _CITATION_BRACKET.sub("", text)
    out = _CITATION_PAREN.sub("", out)
    return _MULTI_SPACE.sub(" ", out)


def _strip_footnotes(text: str) -> str:
    """Remove footnote refs, labels, and short numbered footnote lines."""
    out = _FOOTNOTE_REF.sub("", text)
    out = _FOOTNOTE_LABEL.sub("", out)
    out = _FOOTNOTE_LINE.sub("", out)
    return _MULTI_NEWLINE.sub("\n\n", out)


def _strip_tables(text: str) -> str:
    """Remove markdown table rows, separators, and aligned-column lines."""
    out = _TABLE_ROW.sub("", text)
    out = _TABLE_SEP.sub("", out)
    out = _ALIGNED_COLUMNS.sub("", out)
    return _MULTI_NEWLINE.sub("\n\n", out)


def clean_text(text: str, config: PreprocessingConfig) -> str:
    """Remove noise from raw document text according to config.

    :param text: Raw text from a document.
    :param config: What to remove (tables, footnotes, citations, references).
    :return: Cleaned text.
    """
    if not text or not text.strip():
        return text

    out = text
    if config.remove_reference_section:
        out = _strip_reference_section(out)
    if config.remove_inline_citations:
        out = _strip_inline_citations(out)
    if config.remove_footnotes:
        out = _strip_footnotes(out)
    if config.remove_tables:
        out = _strip_tables(out)

    return out.strip()


def preprocess_documents(
    documents: list[Document],
    config: PreprocessingConfig,
) -> list[Document]:
    """Run preprocessing on each document's text; return new documents with cleaned text.

    :param documents: LlamaIndex Documents (with .text and .metadata).
    :param config: Preprocessing options.
    :return: New list of Documents with cleaned text; metadata preserved.
    """
    if documents:
        logger.info(
            "Preprocessing %d documents (remove_tables=%s, remove_footnotes=%s, "
            "remove_inline_citations=%s, remove_reference_section=%s)",
            len(documents),
            config.remove_tables,
            config.remove_footnotes,
            config.remove_inline_citations,
            config.remove_reference_section,
        )
    result: list[Document] = []
    for doc in documents:
        cleaned = clean_text(doc.text, config)
        new_doc = Document(
            text=cleaned,
            metadata=dict(doc.metadata) if doc.metadata else {},
        )
        result.append(new_doc)
    if documents:
        logger.info("Preprocessed %d documents", len(documents))
    return result
