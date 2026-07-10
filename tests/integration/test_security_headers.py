"""
tests/integration/test_security_headers.py

Assert the security response headers are present on the real app, on both a
public probe and an authenticated route — matching the DoD's `curl -I` check.

Run with: pytest -m integration tests/integration/test_security_headers.py
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import create_app, lifespan

pytestmark = pytest.mark.integration

_EXPECTED = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


@pytest_asyncio.fixture
async def wired() -> AsyncIterator[tuple[AsyncClient, str]]:
    app = create_app()
    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac, get_settings().app.api_key


async def test_headers_on_public_probe(wired: tuple[AsyncClient, str]) -> None:
    client, _ = wired
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    for name, value in _EXPECTED.items():
        assert resp.headers.get(name) == value
    assert "frame-ancestors" in resp.headers.get("Content-Security-Policy", "")
    assert "Permissions-Policy" in resp.headers


async def test_headers_on_authenticated_route(wired: tuple[AsyncClient, str]) -> None:
    client, user_key = wired
    resp = await client.get("/v1/tools", headers={"X-API-Key": user_key})
    assert resp.status_code == 200
    for name, value in _EXPECTED.items():
        assert resp.headers.get(name) == value


async def test_headers_on_auth_failure(wired: tuple[AsyncClient, str]) -> None:
    client, _ = wired
    # Even a 401 (rejected by the auth middleware) is stamped with headers.
    resp = await client.get("/v1/tools")
    assert resp.status_code == 401
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
