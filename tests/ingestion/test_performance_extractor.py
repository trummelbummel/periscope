"""Tests for performance extractor (periscope.ingestion.performance_extractor)."""

from unittest.mock import patch

from llama_index.core import Document

from periscope.ingestion.performance_extractor import (
    annotate_performance_improvement,
)


def test_annotate_performance_improvement_adds_flattened_metadata() -> None:
    """annotate_performance_improvement writes perf_* metadata from extractor output."""
    docs = [Document(text="Some paper text about performance.", metadata={})]

    with patch(
        "periscope.ingestion.performance_extractor.PerformanceImprovementExtractor.extract_from_text"
    ) as extract_mock:
        extract_mock.return_value = {
            "metric_name": "accuracy",
            "improvement_value": 2.5,
            "improvement_unit": "percentage points",
            "dataset_or_task": "MNIST",
            "description": "Improves accuracy by 2.5 percentage points on MNIST.",
        }
        enriched = annotate_performance_improvement(docs)

    assert len(enriched) == 1
    md = enriched[0].metadata
    assert md["perf_improvement_value"] == 2.5
    assert md["perf_improvement_metric"] == "accuracy"
    assert md["perf_improvement_unit"] == "percentage points"
    assert md["perf_improvement_dataset"] == "MNIST"
    assert "perf_improvement_desc" in md

