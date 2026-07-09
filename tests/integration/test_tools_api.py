"""
tests/integration/test_tools_api.py

Integration test for the tools introspection API and end-to-end tool execution
through the registry wired in the lifespan. Requires PG / Redis / Qdrant /
Ollama (same backing services as the RAG API test) plus Docker for the sandbox.

Run with: pytest -m integration tests/integration/test_tools_api.py
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import create_app, lifespan

pytestmark = pytest.mark.integration

_EXPECTED_BUILTINS = {
    "get_datetime",
    "web_search",
    "web_fetch",
    "rag_query",
    "fs_ops",
    "http_request",
    "code_exec",
}


@pytest_asyncio.fixture
async def wired_app() -> AsyncIterator[tuple[AsyncClient, object]]:
    app = create_app()
    async with lifespan(app):
        transport = ASGITransport(app=app)
        # Phase 7: the gateway now requires an API key on /v1/* routes.
        headers = {"X-API-Key": get_settings().app.api_key}
        async with AsyncClient(
            transport=transport, base_url="http://test", headers=headers
        ) as ac:
            yield ac, app


async def test_tools_catalogue_lists_all_builtins(
    wired_app: tuple[AsyncClient, object],
) -> None:
    client, _ = wired_app
    resp = await client.get("/v1/tools")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    names = {t["name"] for t in body["tools"]}
    assert names >= _EXPECTED_BUILTINS
    assert body["count"] >= 7
    # Each tool carries a native function schema.
    for t in body["tools"]:
        assert t["schema"]["type"] == "function"
        assert t["schema"]["function"]["name"] == t["name"]


async def test_single_tool_and_404(wired_app: tuple[AsyncClient, object]) -> None:
    client, _ = wired_app
    ok = await client.get("/v1/tools/rag_query")
    assert ok.status_code == 200
    assert ok.json()["name"] == "rag_query"

    missing = await client.get("/v1/tools/nope")
    assert missing.status_code == 404
    assert missing.json()["error"] == "tool_not_found"


async def test_side_effect_flags_are_exposed(
    wired_app: tuple[AsyncClient, object],
) -> None:
    client, _ = wired_app
    body = (await client.get("/v1/tools")).json()
    flags = {t["name"]: t["side_effects"] for t in body["tools"]}
    assert flags["code_exec"] is True
    assert flags["fs_ops"] is True
    assert flags["get_datetime"] is False


async def test_datetime_tool_executes_through_registry(
    wired_app: tuple[AsyncClient, object],
) -> None:
    # Exercise a real tool via the registry singleton on app.state.
    _, app = wired_app
    registry = app.state.tool_registry  # type: ignore[attr-defined]
    result = await registry.get("get_datetime").run({"timezone": "UTC"})
    assert result.ok
    assert "iso" in result.data
