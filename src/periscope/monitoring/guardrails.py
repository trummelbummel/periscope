"""Guardrails: safety mechanisms for generation.

Per PRD: Abstain from generation if similarity threshold not met.
"""

import logging

from periscope.config import SIMILARITY_THRESHOLD
from periscope.models import RetrievedNode

logger = logging.getLogger(__name__)


class Guardrails:
    """Decides whether to abstain from generation based on retrieval scores."""

    def __init__(self, threshold: float | None = None) -> None:
        """Initialize with minimum score threshold; default from config.

        :param threshold: Minimum best score to allow generation; default from config.
        """
        self._threshold = (
            threshold if threshold is not None else SIMILARITY_THRESHOLD
        )

    def should_abstain(self, nodes: list[RetrievedNode]) -> bool:
        """Return True if we should abstain (no nodes or best score below threshold).

        :param nodes: Retrieved nodes with scores.
        :return: True if we should abstain.
        """
        if not nodes:
            logger.info("Guardrails: no nodes retrieved, abstaining")
            return True
        best_score = max(n.score for n in nodes)
        if best_score < self._threshold:
            logger.info(
                "Guardrails: best score %.4f below threshold %.4f, abstaining",
                best_score,
                self._threshold,
            )
            return True
        return False

    @staticmethod
    def should_abstain_with_options(
        nodes: list[RetrievedNode],
        threshold: float | None = None,
    ) -> bool:
        """Decide whether to abstain (convenience: create Guardrails and run).

        :param nodes: Retrieved nodes with scores.
        :param threshold: Minimum best score to allow generation; default from config.
        :return: True if we should abstain (best score below threshold).
        """
        guardrails = Guardrails(threshold=threshold)
        return guardrails.should_abstain(nodes)


def should_abstain(
    nodes: list[RetrievedNode],
    threshold: float | None = None,
) -> bool:
    """Decide whether to abstain. Delegates to Guardrails."""
    return Guardrails.should_abstain_with_options(nodes, threshold=threshold)
