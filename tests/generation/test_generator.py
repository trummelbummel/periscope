"""Tests for generator (context_engineering_rag.generation.generator)."""

from unittest.mock import MagicMock

from context_engineering_rag.generation.generator import (
    AnswerGenerator,
    generate_answer,
    get_llm,
)
from context_engineering_rag.models import RetrievedNode


def test_answer_generator_with_injected_llm_returns_response() -> None:
    """AnswerGenerator.generate_answer uses injected LLM and returns its response."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Generated answer text"
    mock_llm.complete.return_value = mock_response

    generator = AnswerGenerator(llm=mock_llm)
    nodes = [RetrievedNode(text="Context chunk", score=0.9)]
    result = generator.generate_answer(query="What is it?", context_nodes=nodes)

    assert result == "Generated answer text"
    mock_llm.complete.assert_called_once()
    call_prompt = mock_llm.complete.call_args[0][0]
    assert "Context chunk" in call_prompt
    assert "What is it?" in call_prompt


def test_answer_generator_empty_nodes_formats_empty_context() -> None:
    """AnswerGenerator with empty context_nodes still calls LLM with formatted prompt."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = MagicMock(text="No context")

    generator = AnswerGenerator(llm=mock_llm)
    result = generator.generate_answer(query="Q?", context_nodes=[])

    assert result == "No context"
    call_prompt = mock_llm.complete.call_args[0][0]
    assert "Q?" in call_prompt


def test_answer_generator_uses_custom_prompt_template() -> None:
    """AnswerGenerator uses prompt_template when provided."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = MagicMock(text="OK")

    generator = AnswerGenerator(
        llm=mock_llm,
        prompt_template="Context: {context_str}\nQ: {query_str}\nA:",
    )
    generator.generate_answer(query="Hi", context_nodes=[RetrievedNode(text="x", score=1.0)])

    call_prompt = mock_llm.complete.call_args[0][0]
    assert call_prompt.startswith("Context:")
    assert "Q: Hi" in call_prompt
    assert "A:" in call_prompt


def test_get_llm_returns_openai_instance() -> None:
    """get_llm returns an OpenAI-compatible instance (or from config)."""
    llm = get_llm(model="gpt-4o-mini", api_key="sk-test")
    assert llm is not None
    assert hasattr(llm, "complete")


def test_generate_answer_convenience_function() -> None:
    """generate_answer() delegates to AnswerGenerator and returns text."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = MagicMock(text="Convenience answer")

    result = generate_answer(
        query="Test?",
        context_nodes=[RetrievedNode(text="ctx", score=0.8)],
        llm=mock_llm,
    )

    assert result == "Convenience answer"
    mock_llm.complete.assert_called_once()
