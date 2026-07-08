"""
app/gateway/middleware/request_id.py

Middleware: generates a unique request_id per HTTP request and binds it to:
  1. request.state.request_id — accessible in route handlers.
  2. structlog contextvars — shows up in every log line within this request.
  3. X-Request-ID response header — for client-side correlation.

If the incoming request already has X-Request-ID, we honour it (useful for
upstream proxies / load balancers that inject correlation IDs).
"""

from __future__ import annotations

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from app.core.logging import new_request_id, set_request_context

logger = structlog.get_logger(__name__)

_HEADER_NAME = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        # Accept upstream ID or generate a fresh one
        request_id = request.headers.get(_HEADER_NAME) or new_request_id()

        # Bind to request.state for handler access
        request.state.request_id = request_id

        # Bind to structlog contextvars — visible in all log records for this request
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        set_request_context(request_id=request_id)

        response = await call_next(request)

        # Propagate back to the caller
        response.headers[_HEADER_NAME] = request_id
        return response
