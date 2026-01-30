"""Orchestrates retrieval and generation with guardrails and observability.

Composes: hybrid retrieval -> guardrails -> answer generation.
Returns structured QueryResponse with answer, sources, metadata (per PRD).
"""

import logging
import time

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore

from periscope.config import ENABLE_GUARDRAILS
from periscope.generation.generator import (
    AnswerGenerator,
    format_tables_for_display,
)
from periscope.monitoring.guardrails import should_abstain
from periscope.models import QueryResponse, RetrievedNode
from periscope.retriever.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates retrieval, guardrails, and answer generation."""

    @staticmethod
    def _node_with_score_to_retrieved_node(nws: NodeWithScore) -> RetrievedNode:
        """Convert LlamaIndex NodeWithScore to our Pydantic RetrievedNode."""
        text = nws.node.get_content()
        node_id = nws.node.node_id
        metadata = dict(nws.node.metadata) if nws.node.metadata else {}
        if metadata.get("tables"):
            metadata["tables_display"] = format_tables_for_display(metadata["tables"])
        return RetrievedNode(
            text=text,
            score=float(nws.score),
            node_id=node_id,
            metadata=metadata,
        )

    @staticmethod
    def run_query(
        query: str,
        vector_index: VectorStoreIndex,
        bm25_nodes: list[BaseNode],
        top_k: int | None = None,
    ) -> QueryResponse:
        """Run retrieval + optional generation with guardrails.

        :param query: User question.
        :param vector_index: VectorStoreIndex (Chroma).
        :param bm25_nodes: Nodes for BM25 (same as indexed).
        :param top_k: Retrieval top_k; default from config.
        :return: QueryResponse with answer, sources, metadata, abstained flag.
        """
        start = time.perf_counter()
        retrieved = HybridRetriever.hybrid_retrieve(
            query=query,
            vector_index=vector_index,
            bm25_nodes=bm25_nodes,
            top_k=top_k,
        )
        sources = [
            Pipeline._node_with_score_to_retrieved_node(nws) for nws in retrieved
        ]
        retrieval_time = time.perf_counter() - start

        metadata: dict = {
            "retrieval_time_seconds": round(retrieval_time, 4),
            "num_sources": len(sources),
        }

        if ENABLE_GUARDRAILS and should_abstain(sources):
            return QueryResponse(
                answer="",
                sources=sources,
                metadata={
                    **metadata,
                    "abstained_reason": "similarity_below_threshold",
                },
                abstained=True,
            )

        gen_start = time.perf_counter()
        try:
            answer = AnswerGenerator.generate_answer_with_options(
                query=query, context_nodes=sources
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Generation failed: %s", e)
            return QueryResponse(
                answer="",
                sources=sources,
                metadata={**metadata, "generation_error": str(e)},
                abstained=False,
            )
        gen_time = time.perf_counter() - gen_start
        metadata["generation_time_seconds"] = round(gen_time, 4)

        return QueryResponse(
            answer=answer,
            sources=sources,
            metadata=metadata,
            abstained=False,
        )


def _node_with_score_to_retrieved_node(nws: NodeWithScore) -> RetrievedNode:
    """Convert NodeWithScore to RetrievedNode. Delegates to Pipeline."""
    return Pipeline._node_with_score_to_retrieved_node(nws)


def run_query(
    query: str,
    vector_index: VectorStoreIndex,
    bm25_nodes: list[BaseNode],
    top_k: int | None = None,
) -> QueryResponse:
    """Run retrieval and generation. Delegates to Pipeline."""
    return Pipeline.run_query(
        query=query,
        vector_index=vector_index,
        bm25_nodes=bm25_nodes,
        top_k=top_k,
    )
