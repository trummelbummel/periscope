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


class IngestionStatsWriter:
    """Computes and writes ingestion statistics for observability."""

    @staticmethod
    def _resolve_output_path(path: Path | None) -> Path:
        """Return path or config default for ingestion stats."""
        return path if path is not None else INGESTION_STATS_PATH

    def __init__(self, output_path: Path | None = None) -> None:
        """Initialize with optional output path; default from config.

        :param output_path: File path for stats; default from config.
        """
        self._output_path = IngestionStatsWriter._resolve_output_path(output_path)

    def compute_ingestion_stats(
        self,
        document_count: int,
        chunk_count: int,
        total_chars: int,
        paths: list[str] | None = None,
    ) -> IngestionStats:
        """Compute ingestion statistics.

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
        self,
        stats: IngestionStats,
        output_path: Path | None = None,
    ) -> None:
        """Write ingestion stats to file.

        :param stats: IngestionStats to persist.
        :param output_path: Override path; default uses instance path.
        """
        path = output_path if output_path is not None else self._output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(stats.model_dump(), f, indent=2)
        logger.info("Wrote ingestion stats to %s", path)

    @staticmethod
    def compute_ingestion_stats_default(
        document_count: int,
        chunk_count: int,
        total_chars: int,
        paths: list[str] | None = None,
    ) -> IngestionStats:
        """Compute ingestion statistics (convenience: create writer and run).

        :param document_count: Number of source documents.
        :param chunk_count: Number of chunks produced.
        :param total_chars: Total character count across chunks.
        :param paths: Optional list of source paths.
        :return: IngestionStats model.
        """
        writer = IngestionStatsWriter()
        return writer.compute_ingestion_stats(
            document_count=document_count,
            chunk_count=chunk_count,
            total_chars=total_chars,
            paths=paths,
        )

    @staticmethod
    def write_ingestion_stats_default(
        stats: IngestionStats,
        output_path: Path | None = None,
    ) -> None:
        """Write ingestion stats to file (convenience: create writer and run).

        :param stats: IngestionStats to persist.
        :param output_path: File path; default from config.
        """
        writer = IngestionStatsWriter(output_path=output_path)
        writer.write_ingestion_stats(stats)


def compute_ingestion_stats(
    document_count: int,
    chunk_count: int,
    total_chars: int,
    paths: list[str] | None = None,
) -> IngestionStats:
    """Compute ingestion statistics. Delegates to IngestionStatsWriter."""
    return IngestionStatsWriter.compute_ingestion_stats_default(
        document_count=document_count,
        chunk_count=chunk_count,
        total_chars=total_chars,
        paths=paths,
    )


def write_ingestion_stats(
    stats: IngestionStats,
    output_path: Path | None = None,
) -> None:
    """Write ingestion stats to file. Delegates to IngestionStatsWriter."""
    IngestionStatsWriter.write_ingestion_stats_default(
        stats=stats, output_path=output_path
    )
