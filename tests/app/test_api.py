"""Tests for REST API (periscope.app.api)."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from periscope.app.api import app
from periscope.models import QueryResponse

client = TestClient(app)


def test_health() -> None:
    """GET /health returns 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- /query (retrieve + generate) ---


def test_query_returns_200_and_query_response() -> None:
    """POST /query returns 200 and QueryResponse when index is available."""
    mock_index = MagicMock()
    mock_nodes: list = []
    expected = QueryResponse(
        answer="Generated answer",
        sources=[],
        metadata={"retrieval_time_seconds": 0.1, "generation_time_seconds": 0.2},
        abstained=False,
    )
    with (
        patch("periscope.app.api._ensure_index_or_raise", return_value=(mock_index, mock_nodes)),
        patch("periscope.app.api.run_query", return_value=expected),
    ):
        response = client.post("/query", json={"query": "What is context engineering?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Generated answer"
    assert data["abstained"] is False
    assert "sources" in data
    assert "metadata" in data


def test_query_includes_source_metadata_in_response() -> None:
    """POST /query returns sources with metadata (e.g. file_path) for UI."""
    mock_index = MagicMock()
    mock_nodes: list = []
    expected = QueryResponse(
        answer="Generated answer",
        sources=[
            {
                "text": "ctx",
                "score": 0.9,
                "node_id": "id1",
                "metadata": {"file_path": "/data/doc.pdf", "page_number": 3},
            }
        ],
        metadata={"retrieval_time_seconds": 0.1, "generation_time_seconds": 0.2},
        abstained=False,
    )
    with (
        patch(
            "periscope.app.api._ensure_index_or_raise",
            return_value=(mock_index, mock_nodes),
        ),
        patch("periscope.app.api.run_query", return_value=expected),
    ):
        response = client.post("/query", json={"query": "What is Periscope?"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data.get("sources"), list)
    assert len(data["sources"]) == 1
    source = data["sources"][0]
    assert source["metadata"]["file_path"] == "/data/doc.pdf"
    assert source["metadata"]["page_number"] == 3


def test_query_accepts_optional_top_k() -> None:
    """POST /query accepts optional top_k and passes it to run_query."""
    mock_index = MagicMock()
    mock_nodes: list = []
    expected = QueryResponse(answer="", sources=[], metadata={}, abstained=True)
    with (
        patch("periscope.app.api._ensure_index_or_raise", return_value=(mock_index, mock_nodes)),
        patch("periscope.app.api.run_query", return_value=expected) as mock_run,
    ):
        response = client.post(
            "/query",
            json={"query": "test question", "top_k": 5},
        )
    assert response.status_code == 200
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args[1]
    assert call_kwargs.get("top_k") == 5


def test_query_returns_422_when_query_empty() -> None:
    """POST /query returns 422 when query is empty or missing."""
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422

    response = client.post("/query", json={})
    assert response.status_code == 422


def test_query_returns_503_when_index_unavailable() -> None:
    """POST /query returns 503 when index cannot be loaded."""
    from fastapi import HTTPException

    with patch(
        "periscope.app.api._ensure_index_or_raise",
        side_effect=HTTPException(status_code=503, detail="No index"),
    ):
        response = client.post("/query", json={"query": "any question"})
    assert response.status_code == 503
    assert "detail" in response.json()


# --- /ingest ---


def test_ingest_returns_200_when_successful() -> None:
    """POST /ingest returns 200 and status ok when ingestion succeeds."""
    mock_index = MagicMock()
    mock_nodes: list = []
    with patch(
        "periscope.app.api._ensure_index_or_raise",
        return_value=(mock_index, mock_nodes),
    ):
        response = client.post("/ingest")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "message" in data
    assert "Index built" in data["message"]


def test_ingest_returns_500_on_failure() -> None:
    """POST /ingest returns 500 when ingestion fails."""
    from fastapi import HTTPException

    with patch(
        "periscope.app.api._ensure_index_or_raise",
        side_effect=HTTPException(status_code=500, detail="Ingest failed"),
    ):
        response = client.post("/ingest")
    assert response.status_code == 500
    assert "detail" in response.json()
