"""
tests/unit/test_gateway_hardening.py

Deterministic coverage for the Phase-7 gateway edge — auth, slowapi rate limiting
and security headers — wired onto a *minimal* app with an in-memory slowapi
storage (no Redis). Runs in the default `make test` and pins the behaviour the
live DoD (curl) exercises: 401 without a key, 403 on a wrong key, 429 past a
per-route @limiter.limit budget, admin-only routes, and headers on every response
(including errors). Mirrors production: decorators enforce, no SlowAPIMiddleware.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from fastapi import Depends, FastAPI, Request, Response
from httpx import ASGITransport, AsyncClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from app.core.config import AuthMode, Settings
from app.core.exceptions import AppError
from app.core.security import Role, create_access_token
from app.gateway.middleware.auth import AuthMiddleware, require_admin
from app.gateway.middleware.error_handler import app_error_handler
from app.gateway.middleware.rate_limit import rate_limit_exceeded_handler, rate_limit_key
from app.gateway.security import SecurityHeadersMiddleware

pytestmark = pytest.mark.unit

_USER_KEY = "user-secret"
_ADMIN_KEY = "admin-secret"


# ── App builder ──────────────────────────────────────────────────────────────


def _build_settings(*, auth_mode: AuthMode = AuthMode.APIKEY) -> Settings:
    s = Settings()
    s.app.api_key = _USER_KEY
    s.app.admin_api_key = _ADMIN_KEY
    s.app.auth_mode = auth_mode
    return s


def _build_app(settings: Settings, *, route_limit: str = "2/hour") -> FastAPI:
    # Per-app slowapi limiter with in-memory storage → no Redis, fresh counters.
    # Production enforces via @limiter.limit decorators (not SlowAPIMiddleware),
    # so this app does the same.
    test_limiter = Limiter(
        key_func=rate_limit_key,
        storage_uri="memory://",
        headers_enabled=True,
        swallow_errors=True,
    )

    app = FastAPI()
    app.state.limiter = test_limiter
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_middleware(AuthMiddleware, settings=settings)
    app.add_middleware(SecurityHeadersMiddleware, settings=settings)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/thing")
    async def thing() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/v1/admin/thing")
    async def admin_thing(_: Any = Depends(require_admin)) -> dict[str, bool]:
        return {"admin": True}

    @app.get("/v1/decorated")
    @test_limiter.limit(route_limit)
    async def decorated(request: Request, response: Response) -> dict[str, bool]:
        return {"decorated": True}

    return app


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = _build_app(_build_settings())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Auth: API-key mode ───────────────────────────────────────────────────────


async def test_public_path_needs_no_key(client: AsyncClient) -> None:
    assert (await client.get("/healthz")).status_code == 200


async def test_missing_key_is_401(client: AsyncClient) -> None:
    resp = await client.get("/v1/thing")
    assert resp.status_code == 401
    assert resp.json()["error"] == "authentication_error"


async def test_wrong_key_is_403(client: AsyncClient) -> None:
    resp = await client.get("/v1/thing", headers={"X-API-Key": "nope"})
    assert resp.status_code == 403
    assert resp.json()["error"] == "authorization_error"


async def test_valid_user_key_passes(client: AsyncClient) -> None:
    resp = await client.get("/v1/thing", headers={"X-API-Key": _USER_KEY})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


# ── Authorisation: admin routes ──────────────────────────────────────────────


async def test_admin_route_forbidden_for_user(client: AsyncClient) -> None:
    resp = await client.get("/v1/admin/thing", headers={"X-API-Key": _USER_KEY})
    assert resp.status_code == 403


async def test_admin_route_allows_admin(client: AsyncClient) -> None:
    resp = await client.get("/v1/admin/thing", headers={"X-API-Key": _ADMIN_KEY})
    assert resp.status_code == 200
    assert resp.json() == {"admin": True}


async def test_admin_route_requires_credentials(client: AsyncClient) -> None:
    assert (await client.get("/v1/admin/thing")).status_code == 401


# ── Rate limiting (slowapi) ──────────────────────────────────────────────────


async def test_per_route_decorator_limit_returns_429() -> None:
    app = _build_app(_build_settings(), route_limit="2/hour")
    transport = ASGITransport(app=app)
    headers = {"X-API-Key": _USER_KEY}
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r1 = await ac.get("/v1/decorated", headers=headers)
        r2 = await ac.get("/v1/decorated", headers=headers)
        r3 = await ac.get("/v1/decorated", headers=headers)

    assert (r1.status_code, r2.status_code, r3.status_code) == (200, 200, 429)
    assert r3.json()["error"] == "rate_limit_exceeded"
    assert "Retry-After" in r3.headers
    assert r1.headers["X-RateLimit-Limit"] == "2"


async def test_limit_is_per_identity() -> None:
    # A different principal (admin key) has its own bucket, unaffected by the
    # user key exhausting theirs.
    app = _build_app(_build_settings(), route_limit="1/hour")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        u1 = await ac.get("/v1/decorated", headers={"X-API-Key": _USER_KEY})
        u2 = await ac.get("/v1/decorated", headers={"X-API-Key": _USER_KEY})
        a1 = await ac.get("/v1/decorated", headers={"X-API-Key": _ADMIN_KEY})

    assert (u1.status_code, u2.status_code, a1.status_code) == (200, 429, 200)


async def test_undecorated_route_is_not_rate_limited() -> None:
    app = _build_app(_build_settings(), route_limit="1/hour")
    transport = ASGITransport(app=app)
    headers = {"X-API-Key": _USER_KEY}
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        for _ in range(5):
            assert (await ac.get("/v1/thing", headers=headers)).status_code == 200


# ── Security headers ─────────────────────────────────────────────────────────


async def test_security_headers_present_on_success(client: AsyncClient) -> None:
    resp = await client.get("/healthz")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "no-referrer"
    assert "frame-ancestors" in resp.headers["Content-Security-Policy"]
    assert "Permissions-Policy" in resp.headers


async def test_security_headers_present_on_auth_error(client: AsyncClient) -> None:
    resp = await client.get("/v1/thing")
    assert resp.status_code == 401
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"


async def test_hsts_only_when_enabled() -> None:
    off = _build_settings()
    assert off.gateway.hsts_enabled is False
    async with AsyncClient(
        transport=ASGITransport(app=_build_app(off)), base_url="http://test"
    ) as ac:
        assert "Strict-Transport-Security" not in (await ac.get("/healthz")).headers

    on = _build_settings()
    on.gateway.hsts_enabled = True
    async with AsyncClient(
        transport=ASGITransport(app=_build_app(on)), base_url="http://test"
    ) as ac:
        assert "Strict-Transport-Security" in (await ac.get("/healthz")).headers


# ── Auth: JWT showcase mode ──────────────────────────────────────────────────


async def test_jwt_mode_accepts_valid_bearer() -> None:
    settings = _build_settings(auth_mode=AuthMode.JWT)
    token = create_access_token("user-1", Role.USER, secret=settings.gateway.jwt_secret)
    app = _build_app(settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        ok = await ac.get("/v1/thing", headers={"Authorization": f"Bearer {token}"})
        missing = await ac.get("/v1/thing")
        bad = await ac.get("/v1/thing", headers={"Authorization": "Bearer garbage"})

    assert ok.status_code == 200
    assert missing.status_code == 401
    assert bad.status_code == 403


async def test_jwt_admin_claim_reaches_admin_route() -> None:
    settings = _build_settings(auth_mode=AuthMode.JWT)
    token = create_access_token("op", Role.ADMIN, secret=settings.gateway.jwt_secret)
    app = _build_app(settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/v1/admin/thing", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
