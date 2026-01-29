"""Tests for main_api (context_engineering_rag.main_api)."""

from unittest.mock import MagicMock, patch

from context_engineering_rag.main_api import run


def test_run_calls_uvicorn() -> None:
    """run() calls uvicorn.run with app and config host/port."""
    with (
        patch("context_engineering_rag.main_api.uvicorn.run") as mock_run,
        patch("context_engineering_rag.main_api.API_HOST", "0.0.0.0"),
        patch("context_engineering_rag.main_api.PORT", 8000),
    ):
        run()
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args[0][0] is not None  # app
    assert call_args[1]["host"] == "0.0.0.0"
    assert call_args[1]["port"] == 8000
