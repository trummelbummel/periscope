"""Monitoring: ingestion statistics and observability.

Per PRD: Compute and log statistics including chunk size and document count.
Part of Monitoring component. Write to file.
"""

import json
import logging
from pathlib import Path

from context_engineering_rag.config import INGESTION_STATS_PATH
from context_engineering_rag.models import IngestionStats

logger = logging.getLogger(__name__)


def compute_ingestion_stats(
    document_count: int,
    chunk_count: int,
    total_chars: int,
    paths: list[str] | None = None,
) -> IngestionStats:
    """Compute ingestion statistics for observability.

    :param document_count: Number of source documents.
    :param chunk_count: Number of chunks produced.
    :param total_chars: Total character count across chunks.
    :param paths: Optional list of source paths.
    :return: IngestionStats model.
    """
    avg = total_chars / chunk_count if chunk_count else 0.0
    paths_list = paths if paths is not None else []
    return IngestionStats(
        document_count=document_count,
        chunk_count=chunk_count,
        total_chars=total_chars,
        avg_chunk_size=round(avg, 2),
        paths=paths_list,
    )


def write_ingestion_stats(
    stats: IngestionStats,
    output_path: Path | None = None,
) -> None:
    """Write ingestion stats to file for monitoring.

    :param stats: IngestionStats to persist.
    :param output_path: File path; default from config.
    """
    path = output_path if output_path is not None else INGESTION_STATS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats.model_dump(), f, indent=2)
    logger.info("Wrote ingestion stats to %s", path)
