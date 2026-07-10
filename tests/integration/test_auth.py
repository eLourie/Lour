"""
tests/integration/test_auth.py

Full-stack auth + rate-limit enforcement against the real app (lifespan-wired,
running PG/Redis/Qdrant). Mirrors the Phase-7 DoD: no key → 401, wrong key → 403,
admin route needs the admin key, and exceeding the per-route budget → 429 backed
by real Redis counters.

The rate-limit budget is lowered via env (and the cached Settings reset) so the
429 path is reachable in a handful of requests; the rate-limit Redis DB is flushed
on setup so the window starts clean.

Run with: pytest -m integration tests/integration/test_auth.py
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import create_app, lifespan

pytestmark = pytest.mark.integration

_LOW_LIMIT = 5


@pytest_asyncio.fixture
async def wired() -> AsyncIterator[tuple[AsyncClient, str, str]]:
    prev = {
        "GATEWAY_RATE_LIMIT_DEFAULT": os.environ.get("GATEWAY_RATE_LIMIT_DEFAULT"),
        "GATEWAY_RATE_LIMIT_WINDOW_S": os.environ.get("GATEWAY_RATE_LIMIT_WINDOW_S"),
    }
    os.environ["GATEWAY_RATE_LIMIT_DEFAULT"] = str(_LOW_LIMIT)
    os.environ["GATEWAY_RATE_LIMIT_WINDOW_S"] = "3600"
    get_settings.cache_clear()

    settings = get_settings()
    user_key = settings.app.api_key
    admin_key = settings.app.admin_api_key

    app = create_app()
    async with lifespan(app):
        # Clean slate so counters don't carry across runs within the window.
        await app.state.redis.rate_limit.flushdb()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac, user_key, admin_key

    # Restore env + cached settings for any later tests in the session.
    for name, value in prev.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    get_settings.cache_clear()


async def test_missing_key_returns_401(wired: tuple[AsyncClient, str, str]) -> None:
    client, _, _ = wired
    resp = await client.get("/v1/tools")
    assert resp.status_code == 401
    assert resp.json()["error"] == "authentication_error"


async def test_wrong_key_returns_403(wired: tuple[AsyncClient, str, str]) -> None:
    client, _, _ = wired
    resp = await client.get("/v1/tools", headers={"X-API-Key": "definitely-wrong"})
    assert resp.status_code == 403
    assert resp.json()["error"] == "authorization_error"


async def test_valid_user_key_reaches_route(wired: tuple[AsyncClient, str, str]) -> None:
    client, user_key, _ = wired
    resp = await client.get("/v1/tools", headers={"X-API-Key": user_key})
    assert resp.status_code == 200
    assert resp.json()["count"] >= 7


async def test_admin_route_rejects_user_key(wired: tuple[AsyncClient, str, str]) -> None:
    client, user_key, _ = wired
    resp = await client.get("/v1/admin/metrics", headers={"X-API-Key": user_key})
    assert resp.status_code == 403


async def test_admin_route_accepts_admin_key(wired: tuple[AsyncClient, str, str]) -> None:
    client, _, admin_key = wired
    resp = await client.get("/v1/admin/metrics", headers={"X-API-Key": admin_key})
    assert resp.status_code == 200
    body = resp.json()
    assert body["tools"]["count"] >= 7
    assert body["skills"]["count"] == 4
    assert body["config"]["auth_mode"] == "apikey"


async def test_rate_limit_trips_at_429(wired: tuple[AsyncClient, str, str]) -> None:
    client, user_key, _ = wired
    headers = {"X-API-Key": user_key}
    statuses = [(await client.get("/v1/tools", headers=headers)).status_code for _ in range(8)]
    # First _LOW_LIMIT succeed, the rest are throttled.
    assert statuses[:_LOW_LIMIT] == [200] * _LOW_LIMIT
    assert 429 in statuses[_LOW_LIMIT:]

    throttled = await client.get("/v1/tools", headers=headers)
    assert throttled.status_code == 429
    assert "Retry-After" in throttled.headers


async def test_reload_skills_admin_action(wired: tuple[AsyncClient, str, str]) -> None:
    client, _, admin_key = wired
    resp = await client.post("/v1/admin/reload-skills", headers={"X-API-Key": admin_key})
    assert resp.status_code == 200
    assert resp.json()["count"] == 4
