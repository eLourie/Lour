"""
app/gateway/security.py

Transport-level hardening (Phase 7): security response headers + CORS.

`SecurityHeadersMiddleware` stamps a conservative header set onto every response
(including error responses, since it wraps the auth/rate-limit layers). CORS is
handled by Starlette's own middleware, configured from the single .env-driven
GatewaySettings.

Header choices for an API (no first-party browser UI in MVP):
  X-Content-Type-Options: nosniff   — no MIME sniffing
  X-Frame-Options: DENY             — never framed (clickjacking)
  Referrer-Policy: no-referrer      — don't leak URLs to upstreams
  Content-Security-Policy: frame-ancestors 'none'
                                    — framing guard that (unlike default-src)
                                      doesn't break the Swagger docs page
  Permissions-Policy                — drop powerful browser features
  Strict-Transport-Security         — only when behind TLS (GATEWAY_HSTS_ENABLED)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

from app.core.config import Settings, get_settings

if TYPE_CHECKING:
    from app.core.config import GatewaySettings

_STATIC_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "frame-ancestors 'none'",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, settings: Settings | None = None) -> None:
        super().__init__(app)
        self._settings = settings

    def _get_settings(self) -> Settings:
        return self._settings if self._settings is not None else get_settings()

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        response = await call_next(request)
        for name, value in _STATIC_HEADERS.items():
            response.headers.setdefault(name, value)

        gw = self._get_settings().gateway
        if gw.hsts_enabled:
            response.headers.setdefault(
                "Strict-Transport-Security",
                f"max-age={gw.hsts_max_age}; includeSubDomains",
            )
        return response


def configure_cors(app: FastAPI, settings: GatewaySettings) -> None:
    """Attach Starlette's CORS middleware from config.

    Empty origins (default) means no cross-origin access is granted — the safe
    default for a personal, network-isolated instance. Adding the middleware
    unconditionally keeps preflight handling in one place if origins are set.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Thread-Id"],
    )
