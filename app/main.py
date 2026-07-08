"""
app/main.py

FastAPI application factory + lifespan.

Lifespan responsibility:
  - Initialise backing-service clients (PG, Redis, Qdrant) once at startup.
  - Expose them on app.state for dependency injection.
  - Cleanly shut them down on exit.

Later phases add more singletons to lifespan (LLM service, memory, etc.).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging
from app.gateway.middleware.error_handler import (
    app_error_handler,
    unhandled_exception_handler,
    validation_error_handler,
)
from app.gateway.middleware.logging import AccessLoggingMiddleware
from app.gateway.middleware.request_id import RequestIDMiddleware
from app.gateway.routes import health

logger = structlog.get_logger(__name__)



# Lifespan

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup → yield → Shutdown.

    Phase 0: no persistent clients yet (added in Phase 1).
    The health probes handle missing clients gracefully.
    """
    settings = get_settings()
    logger.info(
        "startup",
        env=settings.app.env,
        deploy_profile=settings.deploy_profile,
        llm_provider=settings.llm.provider,
        llm_model=settings.llm.main_model,
    )

    # Phase 1 will initialise real clients here and attach to app.state:
    #   app.state.db_pool = ...
    #   app.state.redis   = ...
    #   app.state.qdrant  = ...
    # For now, set None so the health check can detect "not initialised" state.
    app.state.db_pool = None
    app.state.redis = None
    app.state.qdrant = None

    logger.info("startup_complete")

    yield  # ← application serves requests here

    # Shutdown
    logger.info("shutdown")

    if app.state.db_pool is not None:
        await app.state.db_pool.dispose()
    if app.state.redis is not None:
        await app.state.redis.aclose()
    # Qdrant client doesn't need explicit teardown (httpx manages connections)

    logger.info("shutdown_complete")



# Application factory

def create_app() -> FastAPI:
    settings = get_settings()

    # Configure structlog — must run before any logger is used.
    configure_logging(
        log_level=settings.app.log_level,
        json_logs=settings.app.env != "development",
    )

    app = FastAPI(
        title="AI Agent",
        description="Production-grade multi-agent system — portfolio & personal assistant.",
        version="0.1.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # Middleware (order matters — outermost added last)
    app.add_middleware(AccessLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # Exception handlers
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # Routes
    app.include_router(health.router)

    # Later phases add:
    # app.include_router(chat.router,   prefix="/v1")
    # app.include_router(skills.router, prefix="/v1")
    # app.include_router(rag.router,    prefix="/v1")
    # app.include_router(tools.router,  prefix="/v1")
    # app.include_router(admin.router,  prefix="/v1")

    return app


# Module-level instance (uvicorn entry point)
app = create_app()
