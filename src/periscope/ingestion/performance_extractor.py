"""LLM-based enrichment: extract main performance improvement from papers.

Per PRD extension: use an LLM to read each paper and extract the main
performance improvement metric the authors report. The result is stored
in document metadata so it is propagated to chunks and can be used as an
index filter at query time.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from llama_index.core import Document
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from periscope.config import GENERATION_MODEL, HUGGINGFACE_TOKEN

logger = logging.getLogger(__name__)


class PerformanceImprovementExtractor:
    """Extracts main performance improvement information using an LLM."""

    # Class-level flag to disable LLM calls after a hard failure (e.g. payment required).
    _disabled: bool = False

    def __init__(
        self,
        model: str | None = None,
        token: str | None = None,
        llm: Any | None = None,
        max_chars: int = 5000,
    ) -> None:
        """Initialize extractor with optional model, token or injected LLM.

        :param model: Hugging Face model id for the Inference API; default GENERATION_MODEL.
        :param token: Hugging Face token; default HUGGINGFACE_TOKEN/HF_TOKEN.
        :param llm: Optional pre-configured LLM implementing .complete().
        :param max_chars: Max characters of document text to send to the LLM.
        """
        self._model = model or GENERATION_MODEL
        self._token = (token or HUGGINGFACE_TOKEN or "").strip()
        self._llm = llm
        self._max_chars = max_chars

    def _get_llm(self) -> HuggingFaceInferenceAPI:
        """Return LLM instance; use injected or create from config."""
        if self._llm is not None:
            return self._llm
        if not self._token:
            logger.warning(
                "HUGGINGFACE_TOKEN not set; performance extractor may be rate-limited "
                "or rejected by the Hugging Face Inference API"
            )
        self._llm = HuggingFaceInferenceAPI(
            model_name=self._model,
            token=self._token or None,
        )
        return self._llm

    def _build_prompt(self, text: str) -> str:
        """Build prompt asking the LLM to extract performance improvement."""
        snippet = text[: self._max_chars]
        return (
            "You are analyzing a machine learning research paper.\n\n"
            "Task: Identify the main performance improvement the authors claim "
            "over baselines.\n\n"
            "From the following paper text, extract:\n"
            '- "metric_name": the evaluation metric '
            '(e.g. "accuracy", "F1", "BLEU", "throughput") or null\n'
            '- "improvement_value": numeric value of the improvement '
            "(e.g. 2.5) or null\n"
            '- "improvement_unit": unit (e.g. "percentage points", "%", '
            '"x speedup") or null\n'
            '- "dataset_or_task": main dataset or task name associated with this '
            "improvement or null\n"
            '- "description": a short one-sentence human-readable summary of the '
            "improvement.\n\n"
            "Return ONLY a JSON object with these keys. Do not include any extra text.\n\n"
            "Paper text:\n"
            f"{snippet}\n"
        )

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        """Try to coerce a value to float; return None on failure."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def extract_from_text(self, text: str) -> dict[str, Any]:
        """Extract performance improvement information from text using the LLM."""
        if not text or not text.strip():
            return {}

        # If we've previously encountered a hard failure (e.g. 402 Payment Required),
        # skip further calls to avoid spamming logs and unnecessary API traffic.
        if self.__class__._disabled:
            return {}

        llm = self._get_llm()
        prompt = self._build_prompt(text)
        try:
            resp = llm.complete(prompt)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            logger.warning("Performance extractor LLM call failed: %s", msg)
            # If credits are depleted or payment is required, disable this component
            # for the remainder of the process to avoid repeated failures.
            if "Payment Required" in msg or "Credit balance is depleted" in msg:
                logger.warning(
                    "Disabling performance extractor due to Hugging Face Inference "
                    "payment/credit error. No further extraction calls will be made "
                    "until the process is restarted."
                )
                self.__class__._disabled = True
            return {}

        raw = getattr(resp, "text", "") if resp is not None else ""
        if not raw:
            return {}

        data: dict[str, Any]
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise TypeError("Expected JSON object")
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not parse performance extractor JSON: %s; raw=%r", e, raw)
            # Fallback: store raw text as description only.
            return {
                "metric_name": None,
                "improvement_value": None,
                "improvement_unit": None,
                "dataset_or_task": None,
                "description": raw[:200],
            }

        # Normalize fields and types.
        value = self._coerce_float(data.get("improvement_value"))
        result: dict[str, Any] = {
            "metric_name": data.get("metric_name"),
            "improvement_value": value,
            "improvement_unit": data.get("improvement_unit"),
            "dataset_or_task": data.get("dataset_or_task"),
            "description": data.get("description"),
        }
        return result


def annotate_performance_improvement(documents: list[Document]) -> list[Document]:
    """Annotate documents with main performance improvement metadata via LLM.

    Adds the following flattened metadata keys to each document (when available):
    - perf_improvement_value: float | None
    - perf_improvement_metric: str | None
    - perf_improvement_unit: str | None
    - perf_improvement_dataset: str | None
    - perf_improvement_desc: str | None

    These metadata fields propagate to chunk nodes and can be used as
    filters at retrieval time.
    """
    if not documents:
        return documents

    extractor = PerformanceImprovementExtractor()
    enriched: list[Document] = []
    for doc in documents:
        try:
            info = extractor.extract_from_text(doc.text)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to extract performance improvement: %s", e)
            info = {}

        md = dict(doc.metadata) if doc.metadata else {}
        if info:
            md["perf_improvement_value"] = info.get("improvement_value")
            md["perf_improvement_metric"] = info.get("metric_name")
            md["perf_improvement_unit"] = info.get("improvement_unit")
            md["perf_improvement_dataset"] = info.get("dataset_or_task")
            md["perf_improvement_desc"] = info.get("description")

        enriched.append(Document(text=doc.text, metadata=md))

    logger.info(
        "Annotated %d documents with performance improvement metadata", len(enriched)
    )
    return enriched

