"""Answer generation: generate responses using LLM based on retrieved context.

Per PRD: GENERATION_MODEL, GENERATION_PROMPT for answer-from-context.
Uses Hugging Face Inference API (serverless). Error handling for model calls.
"""

import logging
from typing import Any

from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from periscope.config import (
    GENERATION_MAX_TOKENS,
    GENERATION_MODEL,
    GENERATION_PROMPT,
    HUGGINGFACE_TOKEN,
)
from periscope.models import RetrievedNode

logger = logging.getLogger(__name__)

# Table data from metadata: list of tables, each table = list of rows, row = list of cell strings
TableData = list[list[list[str | None]]]


def format_tables_for_display(tables: TableData) -> str:
    """Format tables as markdown for context and display.

    :param tables: List of tables (each table is list of rows, row is list of cells).
    :return: Markdown string with one table per section.
    """
    if not tables:
        return ""
    parts = []
    for table in tables:
        if not table:
            continue
        rows = []
        for row in table:
            cells = [str(c) if c is not None else "" for c in row]
            rows.append("| " + " | ".join(cells) + " |")
        if rows:
            # Insert header separator after first row (assume first row is header)
            header_sep = "| " + " | ".join("---" for _ in table[0]) + " |"
            parts.append(rows[0] + "\n" + header_sep + "\n" + "\n".join(rows[1:]))
    return "\n\n".join(parts) if parts else ""


class AnswerGenerator:
    """Generates answers from query and retrieved context via LLM."""

    def __init__(
        self,
        model: str | None = None,
        token: str | None = None,
        prompt_template: str | None = None,
        llm: Any = None,
    ) -> None:
        """Initialize with optional model, token, prompt; or inject LLM.

        :param model: HuggingFace model id for Inference API; default from config.
        :param token: HuggingFace token for Inference API; default from config.
        :param prompt_template: Template with {context_str} and {query_str}; default from config.
        :param llm: Injected LLM (e.g. HuggingFaceInferenceAPI); if None, created from model/token.
        """
        self._model = model if model is not None else GENERATION_MODEL
        self._token = token if token is not None else HUGGINGFACE_TOKEN
        self._prompt_template = (
            prompt_template
            if prompt_template is not None
            else GENERATION_PROMPT
        )
        self._llm = llm

    @staticmethod
    def _build_context_str(nodes: list[RetrievedNode]) -> str:
        """Build context string from retrieved nodes, including tables when present.

        When a node has metadata["tables"], appends formatted tables so the LLM
        can reference them in the answer.

        :param nodes: Retrieved chunks.
        :return: Formatted context string.
        """
        if not nodes:
            return ""
        parts = []
        for i, node in enumerate(nodes, start=1):
            block = f"[{i}]\n{node.text}"
            tables = (node.metadata or {}).get("tables")
            if tables:
                table_str = format_tables_for_display(tables)
                if table_str:
                    block += "\n\nTables:\n" + table_str
            parts.append(block)
        return "\n\n".join(parts)

    def _get_llm(self) -> HuggingFaceInferenceAPI:
        """Return LLM instance; use injected or create from config (Hugging Face Inference API)."""
        if self._llm is not None:
            return self._llm
        if not self._token or not self._token.strip():
            logger.warning("HUGGINGFACE_TOKEN not set; Inference API may rate-limit or reject")
        return HuggingFaceInferenceAPI(
            model_name=self._model,
            token=self._token.strip() if self._token and self._token.strip() else None,
            num_output=GENERATION_MAX_TOKENS,
        )

    def generate_answer(
        self,
        query: str,
        context_nodes: list[RetrievedNode],
    ) -> str:
        """Generate answer from query and retrieved context.

        :param query: User question.
        :param context_nodes: Retrieved chunks (supporting evidence).
        :return: Generated answer text.
        """
        context_str = AnswerGenerator._build_context_str(context_nodes)
        prompt = self._prompt_template.format(
            context_str=context_str,
            query_str=query,
        )
        llm = self._get_llm()
        try:
            response = llm.complete(prompt)
            return response.text if response else ""
        except Exception as e:
            logger.exception("Generation error: %s", e)
            raise

    @staticmethod
    def get_llm(
        model: str | None = None,
        token: str | None = None,
    ) -> HuggingFaceInferenceAPI:
        """Return Hugging Face Inference API LLM for generation.

        :param model: Model name; default from config.
        :param token: HuggingFace token; default from config.
        :return: HuggingFaceInferenceAPI instance.
        """
        generator = AnswerGenerator(model=model, token=token)
        return generator._get_llm()

    @staticmethod
    def generate_answer_with_options(
        query: str,
        context_nodes: list[RetrievedNode],
        prompt_template: str | None = None,
        llm: Any = None,
    ) -> str:
        """Generate answer (convenience: create default generator and run).

        :param query: User question.
        :param context_nodes: Retrieved chunks (supporting evidence).
        :param prompt_template: Optional template with {context_str} and {query_str}.
        :param llm: Optional LLM instance; default from get_llm().
        :return: Generated answer text.
        """
        generator = AnswerGenerator(
            prompt_template=prompt_template,
            llm=llm,
        )
        return generator.generate_answer(query=query, context_nodes=context_nodes)


def get_llm(
    model: str | None = None,
    token: str | None = None,
) -> HuggingFaceInferenceAPI:
    """Return Hugging Face Inference API LLM for generation. Delegates to AnswerGenerator."""
    return AnswerGenerator.get_llm(model=model, token=token)


def generate_answer(
    query: str,
    context_nodes: list[RetrievedNode],
    prompt_template: str | None = None,
    llm: Any = None,
) -> str:
    """Generate answer from query and context. Delegates to AnswerGenerator."""
    return AnswerGenerator.generate_answer_with_options(
        query=query,
        context_nodes=context_nodes,
        prompt_template=prompt_template,
        llm=llm,
    )
