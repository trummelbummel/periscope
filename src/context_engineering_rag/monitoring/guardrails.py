"""Guardrails: safety mechanisms for generation.

Per PRD: Abstain from generation if similarity threshold not met.
"""

import logging

from context_engineering_rag.config import SIMILARITY_THRESHOLD
from context_engineering_rag.models import RetrievedNode

logger = logging.getLogger(__name__)


def should_abstain(
    nodes: list[RetrievedNode],
    threshold: float | None = None,
) -> bool:
    """Decide whether to abstain from generation based on retrieval scores.

    :param nodes: Retrieved nodes with scores.
    :param threshold: Minimum best score to allow generation; default from config.
    :return: True if we should abstain (best score below threshold).
    """
    if not nodes:
        logger.info("Guardrails: no nodes retrieved, abstaining")
        return True
    thresh = threshold if threshold is not None else SIMILARITY_THRESHOLD
    best_score = max(n.score for n in nodes)
    if best_score < thresh:
        logger.info(
            "Guardrails: best score %.4f below threshold %.4f, abstaining",
            best_score,
            thresh,
        )
        return True
    return False
