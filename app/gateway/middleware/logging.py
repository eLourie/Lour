"""
app/gateway/middleware/logging.py

Access-logging middleware.
Logs one structured record per request with:
  method, path, status_code, duration_ms, client_ip, user_agent.

Integrates with structlog contextvars so request_id is always present.
Does NOT log request/response bodies (sensitive data; use Langfuse for that).
"""

from __future__ import annotations

import time

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

logger = structlog.get_logger(__name__)

# Paths excluded from access logging (noisy health-check spam)
_SKIP_PATHS: frozenset[str] = frozenset({"/healthz", "/readyz", "/metrics"})


class AccessLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        response: Response | None = None

        try:
            response = await call_next(request)
            return response
        except Exception:
            raise
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            status = response.status_code if response is not None else 500

            log = logger.info if status < 400 else logger.warning
            log(
                "http_request",
                method=request.method,
                path=request.url.path,
                status_code=status,
                duration_ms=duration_ms,
                client_ip=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )
