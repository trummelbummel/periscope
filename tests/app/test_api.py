"""Tests for REST API (periscope.app.api)."""

from fastapi.testclient import TestClient

from periscope.app.api import app

client = TestClient(app)


def test_health() -> None:
    """GET /health returns 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
