"""Tests for generator (periscope.generation.generator)."""

from unittest.mock import MagicMock

from periscope.generation.generator import (
    AnswerGenerator,
    generate_answer,
)
from periscope.models import RetrievedNode


def test_answer_generator_with_injected_llm_returns_response() -> None:
    """AnswerGenerator.generate_answer uses injected LLM and returns its response."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = MagicMock(text="Generated answer text")

    generator = AnswerGenerator(llm=mock_llm)
    nodes = [RetrievedNode(text="Context chunk", score=0.9)]
    result = generator.generate_answer(query="What is it?", context_nodes=nodes)

    assert result == "Generated answer text"
    mock_llm.complete.assert_called_once()
    call_prompt = mock_llm.complete.call_args[0][0]
    assert "Context chunk" in call_prompt
    assert "What is it?" in call_prompt


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
