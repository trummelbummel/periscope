"""Tests for arXiv scraper (periscope.scraper.arxiv_scraper)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx

from periscope.scraper.arxiv_scraper import ArxivPaper, ArxivScraper


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
