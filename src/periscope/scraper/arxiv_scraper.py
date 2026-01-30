"""Simple arXiv scraper using the public API.

This module provides a small, focused scraper that:

* Sends a query to the arXiv API.
* Parses the Atom XML feed into structured `ArxivPaper` models.
* Optionally downloads PDF files for the returned papers.

It is intentionally minimal and synchronous for ease of use in scripts
and pipelines.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import httpx
from xml.etree import ElementTree as ET

from periscope.config import (
    ARXIV_API_BASE_URL,
    ARXIV_DATA_DIR,
    ARXIV_DEFAULT_QUERY,
    ARXIV_HTTP_TIMEOUT,
    ARXIV_MAX_RESULTS,
    ARXIV_USER_AGENT,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ArxivPaper:
    """Structured representation of a single arXiv paper."""

    id: str
    title: str
    summary: str
    authors: list[str]
    pdf_url: str | None

    @property
    def display_id(self) -> str:
        """Id or title for use in log messages."""
        return self.id or self.title

    def filename(self) -> str:
        """Create a safe filename for storing the PDF."""
        # Prefer using the last path segment of the arXiv ID (which is often a URL),
        # falling back to a slugified title.
        base = ""
        if self.id:
            parsed = urlparse(self.id)
            last_segment = parsed.path.rsplit("/", maxsplit=1)[-1]
            base = last_segment.strip()

        if not base:
            base = re.sub(r"[^a-zA-Z0-9_-]+", "_", self.title.strip())[:80]

        if not base:
            base = "arxiv-paper"
        return f"{base}.pdf"


class ArxivScraper:
    """Scraper for fetching and downloading papers from arXiv."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        """Initialize the scraper.

        Args:
            client: Optional httpx.Client to use (for testability). If not
                provided, a new client will be created with configuration
                based on `config.py`.
        """
        if client is not None:
            self._client = client
        else:
            # Create a client configured according to arXiv API guidelines.
            self._client = httpx.Client(
                timeout=ARXIV_HTTP_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": ARXIV_USER_AGENT},
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fetch_papers(
        self,
        query: str,
        max_results: int | None = None,
        start: int = 0,
    ) -> list[ArxivPaper]:
        """Fetch papers from arXiv for a given query.

        Args:
            query: arXiv search query, e.g. "cat:cs.LG AND deep learning".
            max_results: Maximum number of results to return. If None, uses
                `ARXIV_MAX_RESULTS` from config.
            start: Result offset for pagination.
        """
        effective_max = max_results or ARXIV_MAX_RESULTS
        params = {
            "search_query": query,
            "start": str(start),
            "max_results": str(effective_max),
        }
        logger.info("Querying arXiv API with query=%r, max_results=%d", query, effective_max)

        response = self._client.get(ARXIV_API_BASE_URL, params=params)

        try:
            response.raise_for_status()
        except httpx.HTTPError as err:
            logger.error("arXiv API request failed: %s", err, exc_info=True)
            raise RuntimeError("Failed to fetch results from arXiv API") from err

        logger.debug("Received %d bytes from arXiv API", len(response.content))
        return list(self._parse_atom_feed(response.content))

    def download_pdfs(
        self,
        papers: Iterable[ArxivPaper],
        download_dir: Path | str | None = None,
    ) -> list[Path]:
        """Download PDFs for the given list of papers.

        Args:
            papers: Iterable of `ArxivPaper` instances.
            download_dir: Directory to store PDFs. Defaults to
                ``ARXIV_DATA_DIR`` from config.
        """
        target_dir = Path(download_dir) if download_dir is not None else ARXIV_DATA_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []
        for paper in papers:
            if not paper.pdf_url:
                logger.warning("Paper %s has no PDF URL; skipping download", paper.display_id)
                continue

            pdf_path = target_dir / paper.filename()
            logger.info("Downloading PDF for %s to %s", paper.display_id, pdf_path)
            try:
                self._download_file(paper.pdf_url, pdf_path)
            except Exception as err:  # noqa: BLE001 - network/filesystem failures
                logger.error("Failed to download PDF for %s: %s", paper.display_id, err, exc_info=True)
                continue

            saved_paths.append(pdf_path)

        return saved_paths

    def fetch_default_from_config(self) -> list[ArxivPaper]:
        """Convenience method to fetch using defaults from `config.py`."""
        return self.fetch_papers(query=ARXIV_DEFAULT_QUERY, max_results=ARXIV_MAX_RESULTS)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _parse_atom_feed(self, content: bytes) -> Iterable[ArxivPaper]:
        """Parse the Atom XML feed returned by arXiv into ArxivPaper objects."""
        try:
            root = ET.fromstring(content)
        except ET.ParseError as err:
            logger.error("Failed to parse arXiv Atom feed: %s", err, exc_info=True)
            raise RuntimeError("Could not parse arXiv API response") from err

        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            paper_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()

            authors: list[str] = []
            for author_elem in entry.findall("atom:author", ns):
                name = author_elem.findtext("atom:name", default="", namespaces=ns)
                if name:
                    authors.append(name.strip())

            pdf_url: str | None = None
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("type") == "application/pdf":
                    pdf_url = link.attrib.get("href")
                    break

            yield ArxivPaper(
                id=paper_id,
                title=title,
                summary=summary,
                authors=authors,
                pdf_url=pdf_url,
            )

    def _download_file(self, url: str, path: Path) -> None:
        """Download a single file from URL to the given path."""
        logger.debug("Starting download from %s", url)
        with self._client.stream("GET", url) as response:
            self._write_stream_to_path(response, path)

    @staticmethod
    def _write_stream_to_path(response: httpx.Response, path: Path) -> None:
        """Write a streamed HTTP response to disk with basic error handling."""
        try:
            response.raise_for_status()
        except httpx.HTTPError as err:
            raise RuntimeError(f"Failed to download file from {response.url!s}") from err

        try:
            with path.open("wb") as f:
                for chunk in response.iter_bytes():
                    if chunk:
                        f.write(chunk)
        except OSError as err:
            raise RuntimeError(f"Failed to write file to {path!s}") from err

