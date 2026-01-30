"""Tests for main scraper CLI (periscope.main_scraper)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from periscope.scraper.arxiv_scraper import ArxivPaper


def test_run_calls_scraper_and_downloads(tmp_path: Path) -> None:
    """run() instantiates ArxivScraper, fetches papers, and downloads PDFs."""
    mock_papers = [
        ArxivPaper(
            id="http://arxiv.org/abs/2401.12345",
            title="Example",
            summary="",
            authors=[],
            pdf_url="http://arxiv.org/pdf/2401.12345",
        ),
    ]
    mock_scraper = MagicMock()
    mock_scraper.fetch_papers.return_value = mock_papers
    mock_scraper.download_pdfs.return_value = [tmp_path / "2401.12345.pdf"]

    with (
        patch("periscope.main_scraper.ArxivScraper", return_value=mock_scraper),
        patch("periscope.main_scraper.ARXIV_DEFAULT_QUERY", "test query"),
        patch("periscope.main_scraper.ARXIV_MAX_RESULTS", 5),
        patch("periscope.main_scraper.ARXIV_DATA_DIR", tmp_path),
    ):
        from periscope.main_scraper import run

        run()

    mock_scraper.fetch_papers.assert_called_once_with(query="test query", max_results=5)
    mock_scraper.download_pdfs.assert_called_once_with(mock_papers, download_dir=tmp_path)


def test_run_exits_early_when_no_papers() -> None:
    """run() returns without calling download_pdfs when fetch_papers returns empty."""
    mock_scraper = MagicMock()
    mock_scraper.fetch_papers.return_value = []

    with (
        patch("periscope.main_scraper.ArxivScraper", return_value=mock_scraper),
        patch("periscope.main_scraper.ARXIV_DEFAULT_QUERY", "empty"),
        patch("periscope.main_scraper.ARXIV_MAX_RESULTS", 5),
        patch("periscope.main_scraper.ARXIV_DATA_DIR", Path("/tmp")),
    ):
        from periscope.main_scraper import run

        run()

    mock_scraper.fetch_papers.assert_called_once()
    mock_scraper.download_pdfs.assert_not_called()
