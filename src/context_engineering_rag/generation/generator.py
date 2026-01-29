"""Answer generation: generate responses using LLM based on retrieved context.

Per PRD: GENERATION_MODEL = GPT-5, GENERATION_PROMPT = 'Answer Question based on Context'.
Uses OpenAI-compatible API (OpenAI or compatible). Error handling for external API calls.
"""

import logging

from llama_index.llms.openai import OpenAI
from openai import OpenAIError

from context_engineering_rag.config import (
    GENERATION_MODEL,
    GENERATION_PROMPT,
    OPENAI_API_KEY,
)
from context_engineering_rag.models import RetrievedNode

logger = logging.getLogger(__name__)


def _build_context_str(nodes: list[RetrievedNode]) -> str:
    """Build context string from retrieved nodes."""
    parts = []
    for i, node in enumerate(nodes, start=1):
        parts.append(f"[{i}]\n{node.text}")
    return "\n\n".join(parts)


def get_llm(
    model: str | None = None,
    api_key: str | None = None,
) -> OpenAI:
    """Return OpenAI LLM for generation.

    :param model: Model name; default from config.
    :param api_key: OpenAI API key; default from config.
    :return: OpenAI instance.
    """
    name = model if model is not None else GENERATION_MODEL
    key = api_key if api_key is not None else OPENAI_API_KEY
    if not key:
        logger.warning("OPENAI_API_KEY not set; generation may fail")
    return OpenAI(model=name, api_key=key or "sk-placeholder")


def generate_answer(
    query: str,
    context_nodes: list[RetrievedNode],
    prompt_template: str | None = None,
    llm: OpenAI | None = None,
) -> str:
    """Generate answer from query and retrieved context.

    :param query: User question.
    :param context_nodes: Retrieved chunks (supporting evidence).
    :param prompt_template: Optional template with {context_str} and {query_str}.
    :param llm: Optional LLM instance; default from get_llm().
    :return: Generated answer text.
    :raises OpenAIError: On API failure (caller should handle).
    """
    template = (
        prompt_template
        if prompt_template is not None
        else GENERATION_PROMPT
    )
    context_str = _build_context_str(context_nodes)
    prompt = template.format(
        context_str=context_str,
        query_str=query,
    )
    model = llm if llm is not None else get_llm()
    try:
        response = model.complete(prompt)
        return response.text if response else ""
    except OpenAIError as e:
        logger.exception("OpenAI API error during generation: %s", e)
        raise
