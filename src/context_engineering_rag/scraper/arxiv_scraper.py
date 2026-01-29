"""ArXiv scraper: query and download context engineering papers.

Per PRD: Query and download context engineering papers based on user query.
ARXIV_DATA_DIR = project_root/data/arxiv.
"""

import logging
from pathlib import Path

import arxiv

from context_engineering_rag.config import ARXIV_DATA_DIR
from context_engineering_rag.models import ArxivResult

logger = logging.getLogger(__name__)


class ArxivScraper:
    """Query ArXiv and download PDFs for context engineering papers."""

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialize with output directory for PDFs."""
        self.data_dir = data_dir if data_dir is not None else ARXIV_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def search_and_download(
        self, query: str, max_results: int = 5
    ) -> list[ArxivResult]:
        """Search ArXiv for papers matching query and download PDFs.

        :param query: Search query (e.g. "context engineering RAG").
        :param max_results: Maximum number of papers to download.
        :return: List of ArxivResult with entry_id, title, summary, pdf_url, local_path.
        """
        results: list[ArxivResult] = []
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            for i, result in enumerate(search.results()):
                try:
                    local_path = self._download_pdf(result, i)
                    results.append(
                        ArxivResult(
                            entry_id=result.entry_id,
                            title=result.title or "",
                            summary=result.summary if result.summary else None,
                            pdf_url=result.pdf_url,
                            local_path=str(local_path) if local_path else None,
                        )
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to download %s: %s", result.entry_id, e)
                    results.append(
                        ArxivResult(
                            entry_id=result.entry_id,
                            title=result.title or "",
                            summary=result.summary if result.summary else None,
                            pdf_url=result.pdf_url,
                            local_path=None,
                        )
                    )
        except arxiv.UnexpectedEmptyPageError:
            logger.warning("ArXiv returned empty page for query: %s", query)
        except Exception as e:  # noqa: BLE001
            logger.exception("ArXiv search failed: %s", e)
            raise
        return results

    def _download_pdf(self, result: arxiv.Result, index: int) -> Path | None:
        """Download PDF for one result; return local path or None."""
        # Sanitize filename
        safe_id = result.entry_id.split("/")[-1].replace(".", "_")
        filename = f"{index:02d}_{safe_id}.pdf"
        path = self.data_dir / filename
        result.download_pdf(filename=str(filename), dirpath=str(self.data_dir))
        return path
