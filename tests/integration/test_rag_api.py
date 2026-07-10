"""
tests/integration/test_rag_api.py

Integration test for the RAG API — requires PostgreSQL, Qdrant and Ollama
(dense embeddings + BM42 sparse download on first run).

Run with: pytest -m integration tests/integration/test_rag_api.py

The app is driven over HTTP via httpx ASGITransport. The lifespan is entered
manually so app.state singletons (clients + RAG services) are wired exactly as
in production.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import create_app, lifespan

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = create_app()
    async with lifespan(app):
        transport = ASGITransport(app=app)
        # Phase 7: the gateway now requires an API key on /v1/* routes.
        headers = {"X-API-Key": get_settings().app.api_key}
        async with AsyncClient(
            transport=transport, base_url="http://test", headers=headers
        ) as ac:
            yield ac


async def test_ingest_then_query(client: AsyncClient) -> None:
    marker = uuid.uuid4().hex[:8]
    text = (
        f"Project Zephyr-{marker} is an internal tool for log analysis. "
        f"It was created by the platform team and is written in Rust. "
        f"Zephyr-{marker} ships metrics to a dashboard every minute."
    )

    ingest = await client.post(
        "/v1/rag/ingest",
        json={"text": text, "title": f"zephyr-{marker}", "doc_type": "text"},
    )
    assert ingest.status_code == 200, ingest.text
    body = ingest.json()
    assert body["skipped"] is False
    assert body["chunks"] >= 1
    document_id = body["document_id"]

    # Re-ingesting identical content is idempotent.
    dup = await client.post(
        "/v1/rag/ingest",
        json={"text": text, "title": f"zephyr-{marker}", "doc_type": "text"},
    )
    assert dup.json()["skipped"] is True

    # Query should surface the ingested content.
    query = await client.post(
        "/v1/rag/query",
        json={"query": f"What language is Zephyr-{marker} written in?", "top_k": 5},
    )
    assert query.status_code == 200, query.text
    results = query.json()["results"]
    assert results, "expected at least one retrieved chunk"
    assert any(f"Zephyr-{marker}" in r["content"] for r in results)
    assert any(r["document_id"] == document_id for r in results)


async def test_documents_listing(client: AsyncClient) -> None:
    resp = await client.get("/v1/rag/documents", params={"limit": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert body["limit"] == 5
    assert isinstance(body["total"], int)


async def test_query_validation_rejects_empty(client: AsyncClient) -> None:
    resp = await client.post("/v1/rag/query", json={"query": "", "top_k": 5})
    assert resp.status_code == 422


async def test_ingest_requires_exactly_one_input(client: AsyncClient) -> None:
    both = await client.post(
        "/v1/rag/ingest", json={"source": "/tmp/x.md", "text": "hi"}
    )
    assert both.status_code == 422
    neither = await client.post("/v1/rag/ingest", json={})
    assert neither.status_code == 422
