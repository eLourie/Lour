"""
app/gateway/middleware/auth.py

Authentication at the gateway edge (Phase 7).

Two modes behind ``AUTH_MODE`` (config), one enforcement point:

  • apikey (core)   — caller sends ``X-API-Key``; matched against the user and
                      admin secrets. The matched Principal is stashed on
                      ``request.state.principal``.
  • jwt (showcase)  — caller sends ``Authorization: Bearer <token>``; the token
                      is verified and decoded into the same Principal shape.

Missing credentials → 401; present-but-invalid → 403 (see DoD §Phase 7).

Errors are returned as JSONResponse *directly* rather than raised: a Starlette
BaseHTTPMiddleware sits outside the ExceptionMiddleware, so an AppError raised
here would not reach the global handler. We therefore mirror its JSON shape
({"error", "message"}) by hand.

Per-route authorisation (admin) is a thin dependency (``require_admin``) that
reads the Principal this middleware attached — the middleware authenticates,
the dependency authorises.
"""

from __future__ import annotations

import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from app.core.config import AuthMode, Settings, get_settings
from app.core.exceptions import AuthenticationError, AuthorizationError
from app.core.security import Principal, Role, decode_access_token, identify_api_key

logger = structlog.get_logger(__name__)

_API_KEY_HEADER = "X-API-Key"
_BEARER_PREFIX = "Bearer "

# Paths reachable without credentials: liveness/readiness probes, the OpenAPI
# docs (dev only) and the schema they load. Everything else requires auth.
_PUBLIC_PATHS: frozenset[str] = frozenset(
    {"/", "/healthz", "/readyz", "/metrics", "/docs", "/redoc", "/openapi.json"}
)


def _is_public(path: str) -> bool:
    return path in _PUBLIC_PATHS


def _error_response(exc: AuthenticationError | AuthorizationError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


class AuthMiddleware(BaseHTTPMiddleware):
    """Authenticate every non-public request and attach a Principal."""

    def __init__(self, app: ASGIApp, settings: Settings | None = None) -> None:
        super().__init__(app)
        self._settings = settings

    def _get_settings(self) -> Settings:
        # Resolved lazily so tests can construct the middleware with an explicit
        # Settings, while production falls back to the cached singleton.
        return self._settings if self._settings is not None else get_settings()

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        # CORS preflight carries no credentials by design — let it through so the
        # (outer) CORS middleware can answer it.
        if _is_public(request.url.path) or request.method == "OPTIONS":
            return await call_next(request)

        settings = self._get_settings()
        try:
            principal = (
                self._auth_jwt(request, settings)
                if settings.app.auth_mode is AuthMode.JWT
                else self._auth_api_key(request, settings)
            )
        except (AuthenticationError, AuthorizationError) as exc:
            logger.info(
                "auth_rejected",
                path=request.url.path,
                reason=exc.code,
                status_code=exc.status_code,
            )
            return _error_response(exc)

        request.state.principal = principal
        return await call_next(request)

    # ── API-key mode (core) ──────────────────────────────────────────────
    def _auth_api_key(self, request: Request, settings: Settings) -> Principal:
        key = request.headers.get(_API_KEY_HEADER)
        if not key:
            raise AuthenticationError("Missing API key.")
        principal = identify_api_key(
            key,
            user_key=settings.app.api_key,
            admin_key=settings.app.admin_api_key,
        )
        if principal is None:
            raise AuthorizationError("Invalid API key.")
        return principal

    # ── JWT mode (showcase) ──────────────────────────────────────────────
    def _auth_jwt(self, request: Request, settings: Settings) -> Principal:
        header = request.headers.get("Authorization", "")
        if not header.startswith(_BEARER_PREFIX):
            raise AuthenticationError("Missing bearer token.")
        token = header[len(_BEARER_PREFIX) :].strip()
        principal = decode_access_token(
            token,
            secret=settings.gateway.jwt_secret,
            algorithm=settings.gateway.jwt_algorithm,
        )
        if principal is None:
            raise AuthorizationError("Invalid or expired token.")
        return principal


# ── Authorisation dependencies ───────────────────────────────────────────────


def get_principal(request: Request) -> Principal:
    """Return the Principal attached by AuthMiddleware.

    Falls back to 401 if absent — e.g. a route mounted on a public path that
    still wants an identity.
    """
    principal: Principal | None = getattr(request.state, "principal", None)
    if principal is None:
        raise AuthenticationError("Authentication required.")
    return principal


def require_admin(request: Request) -> Principal:
    """FastAPI dependency: allow only admin principals through (403 otherwise)."""
    principal = get_principal(request)
    if principal.role is not Role.ADMIN:
        raise AuthorizationError("Admin privileges required.")
    return principal
