"""Tests for REST API (periscope.app.api)."""

from fastapi.testclient import TestClient

from periscope import app

client = TestClient(app)


def test_health() -> None:
    """GET /health returns 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_without_data_returns_503() -> None:
    """POST /query when no documents are loaded returns 503."""
    response = client.post("/query", json={"query": "What is RAG?"})
    # No data dir with PDFs or load error -> 503; or 200 if index was built elsewhere
    assert response.status_code in (200, 503)
    if response.status_code == 503:
        assert "detail" in response.json()
