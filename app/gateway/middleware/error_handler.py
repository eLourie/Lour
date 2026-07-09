"""
app/gateway/middleware/error_handler.py

Global exception handler.

Maps:
  AppError subclasses → uniform JSON error response (status from exception)
  RequestValidationError → 422 JSON (Pydantic / FastAPI validation)
  Exception → 500 JSON (unhandled; logs the traceback)

All responses share the same shape:
  { "error": "<code>", "message": "<human text>", "detail": <optional> }
"""

from __future__ import annotations

import structlog
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import AppError

logger = structlog.get_logger(__name__)


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Convert any AppError (and subclasses) to a JSON response."""
    if exc.status_code >= 500:
        logger.error(
            "app_error",
            error_code=exc.code,
            message=exc.message,
            status_code=exc.status_code,
            exc_info=exc,
        )
    else:
        logger.warning(
            "app_error",
            error_code=exc.code,
            message=exc.message,
            status_code=exc.status_code,
        )
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Convert FastAPI / Pydantic validation errors to 422 JSON."""
    # jsonable_encoder normalises non-serialisable ctx values (e.g. the
    # ValueError raised inside a Pydantic model_validator) into plain strings.
    errors = jsonable_encoder(exc.errors())
    logger.info("validation_error", errors=errors)
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Request validation failed.",
            "detail": errors,
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions — log full traceback, return 500."""
    logger.exception(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred.",
        },
    )
