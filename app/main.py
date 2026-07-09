"""
app/main.py

FastAPI application factory + lifespan.

Lifespan responsibility:
  - Initialise all backing-service clients once at startup (singletons).
  - Expose them on app.state for dependency injection via app/core/di.py.
  - Cleanly shut them down on exit.

Phase 1 singletons:
  postgres   — PostgresClient   (engine + session factory)
  redis      — RedisClient      (cache / memory / rate-limit pools)
  qdrant     — QdrantClient     (vector store, bootstrap collections)
  ollama     — OllamaClient     (raw HTTP, shared by LLM + embeddings)
  llm        — LLMProvider      (selected via LLM_PROVIDER)
  embeddings — CachedEmbeddingService (bge-m3 dense + Redis cache)
  reranker   — LocalMPSReranker (lazy-load, degrades gracefully)
  telemetry  — LangfuseTelemetryClient | NoOpTelemetryClient
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.core.config import LLMProvider as LLMProviderEnum
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging
from app.core.telemetry import set_telemetry_client
from app.gateway.middleware.error_handler import (
    app_error_handler,
    unhandled_exception_handler,
    validation_error_handler,
)
from app.gateway.middleware.logging import AccessLoggingMiddleware
from app.gateway.middleware.request_id import RequestIDMiddleware
from app.gateway.routes import health
from app.infra.clients.ollama import OllamaClient
from app.infra.clients.postgres import PostgresClient
from app.infra.clients.qdrant import QdrantClient
from app.infra.clients.redis import RedisClient
from app.infra.clients.reranker import RerankerClient
from app.infra.clients.telemetry import build_telemetry_client
from app.services.embeddings.bge_m3 import BgeM3EmbeddingService
from app.services.embeddings.cache import CachedEmbeddingService
from app.services.llm.factory import build_llm_provider
from app.services.reranker.local_mps import LocalMPSReranker

logger = structlog.get_logger(__name__)


# Lifespan


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup → yield → Shutdown."""
    settings = get_settings()
    logger.info(
        "startup",
        env=settings.app.env,
        deploy_profile=settings.deploy_profile,
        llm_provider=settings.llm.provider,
        llm_model=settings.llm.main_model,
    )

    # 1. Infrastructure clients
    postgres = PostgresClient(settings.postgres)
    redis = RedisClient(settings.redis)
    qdrant = QdrantClient(settings.qdrant)

    # Qdrant: bootstrap collections (idempotent — safe to run every start)
    await qdrant.ensure_collection(settings.qdrant.collection_docs)
    await qdrant.ensure_collection(settings.qdrant.collection_memory)

    # 2. Ollama client (shared by LLM provider + embedding service)
    ollama: OllamaClient | None = None
    if settings.llm.provider == LLMProviderEnum.OLLAMA:
        ollama = OllamaClient(settings.ollama)

    # 3. LLM provider (protocol-based — swap via LLM_PROVIDER)
    llm = build_llm_provider(settings.llm, settings.ollama)

    # 4. Embedding service (bge-m3 dense + Redis cache)
    _embed_client = ollama or OllamaClient(settings.ollama)  # cloud path still needs embed
    bge_m3 = BgeM3EmbeddingService(_embed_client, settings.llm.embed_model)
    embeddings = CachedEmbeddingService(bge_m3, redis)

    # 5. Reranker (lazy-load local MPS; no-op when RERANKER_MODE=none)
    reranker_client = RerankerClient(settings.reranker)
    reranker = LocalMPSReranker(reranker_client, settings.reranker)

    # 6. Telemetry (Langfuse Cloud or no-op)
    telemetry = build_telemetry_client(settings.telemetry)
    set_telemetry_client(telemetry)

    # 7. Attach to app.state for DI
    app.state.postgres = postgres
    app.state.redis = redis
    app.state.qdrant = qdrant
    app.state.ollama = ollama
    app.state.llm = llm
    app.state.embeddings = embeddings
    app.state.reranker = reranker
    app.state.telemetry = telemetry

    # Legacy alias: health.py (Phase 0) used db_pool name — keep until health.py is updated
    app.state.db_pool = postgres

    logger.info("startup_complete")

    yield  # ← application serves requests here

    # Shutdown — reverse order of initialisation
    logger.info("shutdown")

    telemetry.shutdown()
    await reranker_client.aclose()
    await qdrant.aclose()
    await redis.aclose()
    await postgres.aclose()
    if ollama is not None:
        await ollama.aclose()

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
