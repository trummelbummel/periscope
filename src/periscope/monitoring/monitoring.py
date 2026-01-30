"""Monitoring: ingestion statistics and observability.

Per PRD: Compute and log statistics including chunk size and document count.
Stats are written only after the index and BM25 nodes are persisted, so they
reflect the persisted index and embedding data. Part of Monitoring component.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import (
    RetrieverEvaluator,
    generate_question_context_pairs,
)
from llama_index.core.schema import BaseNode

from periscope.config import INGESTION_STATS_PATH
from periscope.generation.generator import AnswerGenerator
from periscope.models import IngestionStats
from periscope.retriever.retriever import HybridRetriever

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
        embedding_model: str = "",
        preprocessing_config: dict | None = None,
        chunk_size: int = 0,
        chunk_overlap: int = 0,
    ) -> IngestionStats:
        """Compute ingestion statistics.

        :param document_count: Number of source documents.
        :param chunk_count: Number of chunks produced.
        :param total_chars: Total character count across chunks.
        :param paths: Optional list of source paths.
        :param embedding_model: Model id used for embeddings (e.g. from config).
        :param preprocessing_config: Preprocessing options used during ingestion.
        :param chunk_size: CHUNK_SIZE used during ingestion (for config fingerprint).
        :param chunk_overlap: CHUNK_OVERLAP used during ingestion (for config fingerprint).
        :return: IngestionStats model.
        """
        avg = total_chars / chunk_count if chunk_count else 0.0
        paths_list = paths if paths is not None else []
        preprocess = preprocessing_config if preprocessing_config is not None else {}
        return IngestionStats(
            document_count=document_count,
            chunk_count=chunk_count,
            total_chars=total_chars,
            avg_chunk_size=round(avg, 2),
            paths=paths_list,
            embedding_model=embedding_model,
            preprocessing_config=preprocess,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
        embedding_model: str = "",
        preprocessing_config: dict | None = None,
        chunk_size: int = 0,
        chunk_overlap: int = 0,
    ) -> IngestionStats:
        """Compute ingestion statistics (convenience: create writer and run)."""
        writer = IngestionStatsWriter()
        return writer.compute_ingestion_stats(
            document_count=document_count,
            chunk_count=chunk_count,
            total_chars=total_chars,
            paths=paths,
            embedding_model=embedding_model,
            preprocessing_config=preprocessing_config,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
    embedding_model: str = "",
    preprocessing_config: dict | None = None,
    chunk_size: int = 0,
    chunk_overlap: int = 0,
) -> IngestionStats:
    """Compute ingestion statistics. Delegates to IngestionStatsWriter."""
    return IngestionStatsWriter.compute_ingestion_stats_default(
        document_count=document_count,
        chunk_count=chunk_count,
        total_chars=total_chars,
        paths=paths,
        embedding_model=embedding_model,
        preprocessing_config=preprocessing_config,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def read_ingestion_stats(output_path: Path | None = None) -> IngestionStats | None:
    """Load ingestion stats from file if it exists.

    :param output_path: File path; default from config INGESTION_STATS_PATH.
    :return: IngestionStats if file exists and is valid JSON, else None.
    """
    path = output_path if output_path is not None else INGESTION_STATS_PATH
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return IngestionStats.model_validate(data)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Could not read ingestion stats from %s: %s", path, e)
        return None


def write_ingestion_stats(
    stats: IngestionStats,
    output_path: Path | None = None,
) -> None:
    """Write ingestion stats to file. Delegates to IngestionStatsWriter."""
    IngestionStatsWriter.write_ingestion_stats_default(
        stats=stats, output_path=output_path
    )


class RetrievalExperiment:
    """Run retrieval quality metrics on an existing index and write results to disk.

    Uses LlamaIndex's RetrieverEvaluator with standard retrieval metrics
    (hit_rate, mrr, precision, recall, ap, ndcg), following the pattern from:
    https://developers.llamaindex.ai/python/examples/evaluation/retrieval/retriever_eval/
    """

    def __init__(
        self,
        output_path: Path | None = None,
        num_questions_per_chunk: int = 1,
        max_nodes: int = 50,
    ) -> None:
        """Initialize experiment configuration.

        :param output_path: Where to write JSON results; default monitoring/data/retrieval_evaluation.json.
        :param num_questions_per_chunk: Synthetic questions to generate per node.
        :param max_nodes: Maximum number of nodes to include in the eval dataset.
        """
        default_dir = INGESTION_STATS_PATH.parent
        self._output_path = (
            output_path if output_path is not None else default_dir / "retrieval_evaluation.json"
        )
        self._num_questions_per_chunk = num_questions_per_chunk
        self._max_nodes = max_nodes

    def _subset_nodes(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """Return a small subset of nodes for a lightweight experiment."""
        if self._max_nodes <= 0 or len(nodes) <= self._max_nodes:
            return nodes
        return nodes[: self._max_nodes]

    def run(
        self,
        vector_index: VectorStoreIndex,
        bm25_nodes: list[BaseNode],
    ) -> Path:
        """Run retrieval evaluation on the given index and nodes.

        Builds a HybridRetriever over the provided index + nodes, generates a small
        synthetic QA dataset from the nodes using the configured LLM, evaluates
        retrieval with standard metrics, and writes aggregated results to JSON.

        :param vector_index: Persisted VectorStoreIndex (e.g. Chroma-backed).
        :param bm25_nodes: Nodes used for BM25 (same as indexed).
        :return: Path to the written retrieval_evaluation.json file.
        """
        if not bm25_nodes:
            raise ValueError("RetrievalExperiment requires at least one node for evaluation.")

        nodes_subset = self._subset_nodes(bm25_nodes)
        logger.info(
            "Running retrieval experiment on %d nodes (subset of %d total)",
            len(nodes_subset),
            len(bm25_nodes),
        )

        # Build hybrid retriever over the existing index and nodes.
        retriever = HybridRetriever(vector_index=vector_index, bm25_nodes=bm25_nodes)

        # Generate a synthetic QA dataset over the subset of nodes.
        llm = AnswerGenerator.get_llm()
        qa_dataset = generate_question_context_pairs(
            nodes_subset,
            llm=llm,
            num_questions_per_chunk=self._num_questions_per_chunk,
        )

        metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]
        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            metrics, retriever=retriever
        )

        metric_dicts: list[dict[str, float]] = []
        for sample_id, query in qa_dataset.queries.items():
            expected = qa_dataset.relevant_docs[sample_id]
            eval_result = retriever_evaluator.evaluate(query, expected)
            metric_dicts.append(eval_result.metric_vals_dict)

        aggregated: dict[str, float] = {}
        if metric_dicts:
            for key in metrics:
                values = [m.get(key, 0.0) for m in metric_dicts]
                aggregated[key] = float(sum(values) / len(values))

        result: dict[str, object] = {
            "metrics": aggregated,
            "num_queries": len(metric_dicts),
            "num_nodes": len(nodes_subset),
            "num_questions_per_chunk": self._num_questions_per_chunk,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._output_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info("Wrote retrieval evaluation metrics to %s", self._output_path)
        return self._output_path


def run_retrieval_experiment(
    vector_index: VectorStoreIndex,
    bm25_nodes: list[BaseNode],
    output_path: Path | None = None,
    num_questions_per_chunk: int = 1,
    max_nodes: int = 50,
) -> Path:
    """Convenience wrapper to run RetrievalExperiment with defaults."""
    experiment = RetrievalExperiment(
        output_path=output_path,
        num_questions_per_chunk=num_questions_per_chunk,
        max_nodes=max_nodes,
    )
    return experiment.run(vector_index=vector_index, bm25_nodes=bm25_nodes)

