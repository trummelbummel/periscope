"""Tests for arXiv scraper (context_engineering_rag.scraper.arxiv_scraper)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from context_engineering_rag.scraper.arxiv_scraper import ArxivPaper, ArxivScraper


# --------------------------------------------------------------------------- #
# ArxivPaper.filename()
# --------------------------------------------------------------------------- #


def test_arxiv_paper_filename_uses_id_last_segment() -> None:
    """filename() uses last path segment of id when id is a URL."""
    paper = ArxivPaper(
        id="http://arxiv.org/abs/2401.12345",
        title="Some Title",
        summary="",
        authors=[],
        pdf_url=None,
    )
    assert paper.filename() == "2401.12345.pdf"


def test_arxiv_paper_filename_plain_id() -> None:
    """filename() uses id as base when id has no path slashes."""
    paper = ArxivPaper(
        id="2401.12345",
        title="Some Title",
        summary="",
        authors=[],
        pdf_url=None,
    )
    # urlparse("2401.12345") gives path "2401.12345", last segment is "2401.12345"
    assert paper.filename() == "2401.12345.pdf"


def test_arxiv_paper_filename_fallback_to_title_slug() -> None:
    """filename() slugifies title when id is empty."""
    paper = ArxivPaper(
        id="",
        title="Deep Learning for NLP: A Survey",
        summary="",
        authors=[],
        pdf_url=None,
    )
    assert paper.filename() == "Deep_Learning_for_NLP_A_Survey.pdf"


def test_arxiv_paper_filename_fallback_arxiv_paper_when_empty() -> None:
    """filename() returns arxiv-paper.pdf when id and title are empty."""
    paper = ArxivPaper(
        id="",
        title="",
        summary="",
        authors=[],
        pdf_url=None,
    )
    assert paper.filename() == "arxiv-paper.pdf"


# --------------------------------------------------------------------------- #
# ArxivScraper.fetch_papers() and _parse_atom_feed
# --------------------------------------------------------------------------- #

_MINIMAL_ATOM_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345</id>
    <title>Example Paper Title</title>
    <summary>Abstract text here.</summary>
    <author><name>Alice</name></author>
    <author><name>Bob</name></author>
    <link href="http://arxiv.org/pdf/2401.12345" type="application/pdf" />
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.67890</id>
    <title>Second Paper</title>
    <summary>Second abstract.</summary>
    <author><name>Carol</name></author>
    <link href="http://arxiv.org/pdf/2401.67890" type="application/pdf" />
  </entry>
</feed>
"""


def test_fetch_papers_parses_atom_feed() -> None:
    """fetch_papers returns list of ArxivPaper from Atom XML."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.content = _MINIMAL_ATOM_FEED
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_response

    scraper = ArxivScraper(client=mock_client)
    papers = scraper.fetch_papers(query="test", max_results=10)

    assert len(papers) == 2
    assert papers[0].id == "http://arxiv.org/abs/2401.12345"
    assert papers[0].title == "Example Paper Title"
    assert papers[0].summary == "Abstract text here."
    assert papers[0].authors == ["Alice", "Bob"]
    assert papers[0].pdf_url == "http://arxiv.org/pdf/2401.12345"

    assert papers[1].id == "http://arxiv.org/abs/2401.67890"
    assert papers[1].title == "Second Paper"
    assert papers[1].authors == ["Carol"]


def test_fetch_papers_raises_on_http_error() -> None:
    """fetch_papers raises RuntimeError when API request fails."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server error",
        request=MagicMock(),
        response=mock_response,
    )

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_response

    scraper = ArxivScraper(client=mock_client)
    with pytest.raises(RuntimeError, match="Failed to fetch results from arXiv API"):
        scraper.fetch_papers(query="test", max_results=5)


# --------------------------------------------------------------------------- #
# ArxivScraper.download_pdfs()
# --------------------------------------------------------------------------- #


def test_download_pdfs_skips_paper_without_pdf_url(tmp_path: Path) -> None:
    """download_pdfs skips papers with no pdf_url and returns empty list."""
    paper = ArxivPaper(
        id="2401.12345",
        title="No PDF",
        summary="",
        authors=[],
        pdf_url=None,
    )
    mock_client = MagicMock(spec=httpx.Client)
    scraper = ArxivScraper(client=mock_client)
    saved = scraper.download_pdfs([paper], download_dir=tmp_path)
    assert saved == []
    mock_client.stream.assert_not_called()


def test_download_pdfs_downloads_and_returns_paths(tmp_path: Path) -> None:
    """download_pdfs downloads PDFs and returns list of saved paths."""
    paper = ArxivPaper(
        id="http://arxiv.org/abs/2401.12345",
        title="Example",
        summary="",
        authors=[],
        pdf_url="http://arxiv.org/pdf/2401.12345",
    )
    mock_stream = MagicMock()
    mock_stream.iter_bytes.return_value = [b"pdf content"]
    mock_stream.raise_for_status = MagicMock()
    mock_response = MagicMock()
    mock_response.__enter__ = MagicMock(return_value=mock_stream)
    mock_response.__exit__ = MagicMock(return_value=None)

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.stream.return_value = mock_response

    scraper = ArxivScraper(client=mock_client)
    saved = scraper.download_pdfs([paper], download_dir=tmp_path)

    assert len(saved) == 1
    assert saved[0] == tmp_path / "2401.12345.pdf"
    assert saved[0].read_bytes() == b"pdf content"


def test_download_pdfs_continues_on_single_failure(tmp_path: Path) -> None:
    """download_pdfs continues with other papers when one download fails."""
    good = ArxivPaper(
        id="http://arxiv.org/abs/2401.11111",
        title="Good",
        summary="",
        authors=[],
        pdf_url="http://arxiv.org/pdf/2401.11111",
    )
    bad = ArxivPaper(
        id="http://arxiv.org/abs/2401.99999",
        title="Bad",
        summary="",
        authors=[],
        pdf_url="http://arxiv.org/pdf/2401.99999",
    )
    mock_stream_ok = MagicMock()
    mock_stream_ok.iter_bytes.return_value = [b"ok"]
    mock_stream_ok.raise_for_status = MagicMock()
    resp_ok = MagicMock()
    resp_ok.__enter__ = MagicMock(return_value=mock_stream_ok)
    resp_ok.__exit__ = MagicMock(return_value=None)

    mock_stream_fail = MagicMock()
    mock_stream_fail.raise_for_status.side_effect = RuntimeError("network error")
    resp_fail = MagicMock()
    resp_fail.__enter__ = MagicMock(return_value=mock_stream_fail)
    resp_fail.__exit__ = MagicMock(return_value=None)

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.stream.side_effect = [resp_ok, resp_fail]

    scraper = ArxivScraper(client=mock_client)
    saved = scraper.download_pdfs([good, bad], download_dir=tmp_path)

    assert len(saved) == 1
    assert saved[0].read_bytes() == b"ok"


# --------------------------------------------------------------------------- #
# ArxivScraper.fetch_default_from_config
# --------------------------------------------------------------------------- #


def test_fetch_default_from_config_calls_fetch_papers() -> None:
    """fetch_default_from_config calls fetch_papers with config defaults."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.content = _MINIMAL_ATOM_FEED
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_response

    scraper = ArxivScraper(client=mock_client)
    papers = scraper.fetch_default_from_config()

    assert len(papers) >= 1
    mock_client.get.assert_called_once()
    _, kwargs = mock_client.get.call_args
    assert "search_query" in (kwargs or {}).get("params", {})
