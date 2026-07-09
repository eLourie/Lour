"""
app/gateway/middleware/rate_limit.py

Redis-backed rate limiting with per-route budgets (Phase 7), built on slowapi
(§3.1 stack choice; ADR-011 keeps *enforcement* at the gateway edge).

Layout:
  • `limiter` — a process-wide slowapi Limiter, storage = the rate-limit logical
    Redis DB (REDIS_DB_RATELIMIT), keyed by the authenticated principal (falling
    back to client IP).
  • Per-route budgets — enforced through `@limiter.limit(...)` decorators on the
    routes: `/v1/chat` (agent turns, `chat_limit`), `/v1/rag/ingest` (heavy,
    bursty, `ingest_limit`) and `/v1/tools` (`default_limit`, the general-purpose
    budget other routes can adopt with a one-line decorator).
  • `rate_limit_exceeded_handler` — a *synchronous* 429 handler (slowapi's own
    middleware would swap an async handler for its plain-text fallback; keeping it
    sync guarantees the uniform JSON error shape).

Why decorators, not SlowAPIMiddleware: the middleware's default-limit path
resolves the endpoint via `app.routes`, which this FastAPI version keeps as lazy
`_IncludedRouter` wrappers — so it can't see `/v1/*` handlers and would silently
exempt them. The `@limiter.limit` decorator enforces inside the endpoint call and
is unaffected.

Limit values are callables that re-read Settings on each evaluation, so the
budgets follow config (and test overrides) without rebuilding the limiter.
`swallow_errors=True` makes the limiter fail-open if Redis is unreachable —
rate limiting is an availability safeguard, not a correctness gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import get_settings
from app.core.exceptions import RateLimitError
from app.core.logging import get_logger

if TYPE_CHECKING:
    from slowapi.errors import RateLimitExceeded

logger = get_logger(__name__)


def rate_limit_key(request: Request) -> str:
    """Rate-limit domain: the authenticated principal, else the client IP.

    AuthMiddleware runs outside the limiter, so `request.state.principal` is
    already set for authenticated traffic; unauthenticated paths key off IP.
    """
    principal = getattr(request.state, "principal", None)
    if principal is not None:
        return f"sub:{principal.subject}"
    return f"ip:{get_remote_address(request)}"


def _limit_string(count: int, window_s: int) -> str:
    # limits grammar: "<count>/<multiple> <unit>" → e.g. "30/60 seconds".
    return f"{count}/{window_s} seconds"


def default_limit() -> str:
    gw = get_settings().gateway
    return _limit_string(gw.rate_limit_default, gw.rate_limit_window_s)


def chat_limit() -> str:
    gw = get_settings().gateway
    return _limit_string(gw.rate_limit_chat, gw.rate_limit_window_s)


def ingest_limit() -> str:
    gw = get_settings().gateway
    return _limit_string(gw.rate_limit_ingest, gw.rate_limit_window_s)


def _build_limiter() -> Limiter:
    settings = get_settings()
    return Limiter(
        key_func=rate_limit_key,
        default_limits=[default_limit],
        storage_uri=settings.redis.url(settings.redis.db_ratelimit),
        strategy="fixed-window",
        headers_enabled=True,  # emit X-RateLimit-* / Retry-After
        swallow_errors=True,  # Redis down → allow (fail-open)
        enabled=settings.gateway.rate_limit_enabled,
    )


# Process-wide singleton: route modules import it for @limiter.limit decorators,
# main.py registers it on app.state and adds SlowAPIMiddleware.
limiter = _build_limiter()


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Uniform JSON 429 (sync — see module docstring) with rate-limit headers."""
    error = RateLimitError()
    response: Response = JSONResponse(status_code=error.status_code, content=error.to_dict())
    current_limit = getattr(request.state, "view_rate_limit", None)
    if current_limit is not None:
        # Mutates response in place, adding X-RateLimit-* and Retry-After.
        request.app.state.limiter._inject_headers(response, current_limit)
    logger.info("rate_limited", path=request.url.path)
    return response
