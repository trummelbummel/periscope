"""Tests for ingestion pipeline (periscope.ingestion.ingestion_pipeline)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from periscope.ingestion.ingestion_pipeline import (
    IngestionPipeline,
    IngestionResult,
    NoDocumentsError,
)
from periscope.models import IngestionStats


def test_run_raises_no_documents_error_when_no_docs() -> None:
    """Pipeline.run() raises NoDocumentsError when no documents in data dirs."""
    pipeline = IngestionPipeline(
        data_dir=Path("/empty"),
        arxiv_data_dir=Path("/also_empty"),
    )
    with patch(
        "periscope.ingestion.ingestion_pipeline.set_global_embed_model",
    ), patch(
        "periscope.ingestion.ingestion_pipeline.load_documents_from_directory",
        return_value=[],
    ) as load_mock:
        with pytest.raises(NoDocumentsError) as exc_info:
            pipeline.run()
        assert "No documents" in str(exc_info.value)
        assert load_mock.call_count >= 1


def test_run_returns_ingestion_result_when_docs_present() -> None:
    """Pipeline.run() returns IngestionResult with index, nodes, stats when docs loaded."""
    from llama_index.core import Document

    fake_index = MagicMock()
    fake_stats = IngestionStats(
        document_count=1,
        chunk_count=2,
        total_chars=100,
        avg_chunk_size=50.0,
        paths=[],
        embedding_model="test-model",
        preprocessing_config={},
    )

    pipeline = IngestionPipeline(
        data_dir=Path("/any"),
        arxiv_data_dir=Path("/any"),
    )
    with patch(
        "periscope.ingestion.ingestion_pipeline.set_global_embed_model",
    ), patch(
        "periscope.ingestion.ingestion_pipeline.load_documents_from_directory",
        return_value=[Document(text="Some content here for chunking." * 20)],
    ), patch(
        "periscope.ingestion.ingestion_pipeline.compute_ingestion_stats",
        return_value=fake_stats,
    ), patch(
        "periscope.ingestion.ingestion_pipeline.write_ingestion_stats",
    ), patch(
        "periscope.ingestion.ingestion_pipeline.build_index_from_nodes",
        side_effect=lambda nodes, persist_dir=None: (fake_index, list(nodes)),
    ), patch(
        "periscope.ingestion.ingestion_pipeline.persist_bm25_nodes",
    ):
        result = pipeline.run()

    assert isinstance(result, IngestionResult)
    assert result.index is fake_index
    assert result.stats is fake_stats
    assert len(result.nodes) >= 1
