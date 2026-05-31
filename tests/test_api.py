"""Integration tests for the FastAPI endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_list_repos(client):
    response = await client.get("/api/repos")
    assert response.status_code == 200
    assert "repositories" in response.json()


@pytest.mark.asyncio
async def test_query_without_repo_returns_404(client):
    response = await client.post(
        "/api/query",
        json={"question": "How does auth work?", "repo_name": "nonexistent"},
    )
    assert response.status_code in (404, 503)


@pytest.mark.asyncio
async def test_ingest_invalid_url(client):
    response = await client.post(
        "/api/ingest",
        json={"repo_url": "not-a-valid-url"},
    )
    assert response.status_code == 400
