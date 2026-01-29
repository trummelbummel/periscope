"""Tests for monitoring (context_engineering_rag.monitoring.monitoring)."""

import json
import tempfile
from pathlib import Path

from context_engineering_rag.models import IngestionStats
from context_engineering_rag.monitoring import (
    compute_ingestion_stats,
    write_ingestion_stats,
)


def test_compute_ingestion_stats() -> None:
    """compute_ingestion_stats returns IngestionStats with correct avg."""
    stats = compute_ingestion_stats(
        document_count=2,
        chunk_count=10,
        total_chars=1000,
        paths=["/a.pdf", "/b.pdf"],
    )
    assert stats.document_count == 2
    assert stats.chunk_count == 10
    assert stats.total_chars == 1000
    assert stats.avg_chunk_size == 100.0
    assert len(stats.paths) == 2


def test_write_ingestion_stats() -> None:
    """write_ingestion_stats persists JSON file."""
    stats = IngestionStats(
        document_count=1,
        chunk_count=5,
        total_chars=500,
        avg_chunk_size=100.0,
        paths=[],
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stats.json"
        write_ingestion_stats(stats, output_path=path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["document_count"] == 1
        assert data["chunk_count"] == 5
