"""Command-line entrypoint for running the arXiv scraper.

This script:
  * Reads the default arXiv query and data directory from `config.py`.
  * Uses `ArxivScraper` (built on LlamaIndex's ArxivReader) to fetch papers.
  * Stores PDFs in the configured data directory.
"""

from __future__ import annotations

import logging

from context_engineering_rag.config import ARXIV_DATA_DIR, ARXIV_DEFAULT_QUERY, ARXIV_MAX_RESULTS
from context_engineering_rag.scraper import ArxivScraper

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure a simple logging setup for CLI usage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def run() -> None:
    """Run the arXiv scraper with configuration from `config.py`."""
    _configure_logging()
    logger.info("Starting arXiv scraper with query=%r", ARXIV_DEFAULT_QUERY)

    scraper = ArxivScraper()

    papers = scraper.fetch_papers(
        query=ARXIV_DEFAULT_QUERY,
        max_results=ARXIV_MAX_RESULTS,
    )
    if not papers:
        logger.warning("No papers returned from arXiv for query=%r", ARXIV_DEFAULT_QUERY)
        return

    logger.info("Fetched %d papers from arXiv", len(papers))
    saved_paths = scraper.download_pdfs(papers, download_dir=ARXIV_DATA_DIR)

    if not saved_paths:
        logger.warning("No PDFs were downloaded for query=%r", ARXIV_DEFAULT_QUERY)
    else:
        logger.info("Downloaded %d PDFs to %s", len(saved_paths), ARXIV_DATA_DIR)


if __name__ == "__main__":  # pragma: no cover
    run()

