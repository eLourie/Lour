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

Phase 3 singletons:
  sandbox       — DockerSandbox (isolated code execution)
  tool_registry — ToolRegistry (builtins + MCP-adapter tools)
  tool_gate     — ToolGate (allowlist + approval enforcement)
  mcp_client    — McpClient (external MCP servers; dormant when none configured)

Phase 4 singletons:
  memory        — MemoryManager (short-term / long-term / episodic facade)
  consolidation — ConsolidationService (APScheduler background fact distillation)

Phase 5 singletons:
  checkpointer  — CheckpointerManager (AsyncPostgresSaver, resume/replay, ADR-008)
  agent_graph   — compiled supervisor graph (researcher/coder/direct + HITL + SSE)

Phase 6 singletons:
  skill_registry — SkillRegistry (YAML declarations + optional Python overrides)
  skill_router   — SkillRouter (free text → one skill, §5.3)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from slowapi.errors import RateLimitExceeded

from app.agents.checkpointing import CheckpointerManager
from app.agents.deps import GraphDeps
from app.agents.enforcement.budget import BudgetEnforcer
from app.agents.graphs.builder import build_graph
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging
from app.core.telemetry import set_telemetry_client
from app.gateway.middleware.auth import AuthMiddleware
from app.gateway.middleware.error_handler import (
    app_error_handler,
    unhandled_exception_handler,
    validation_error_handler,
)
from app.gateway.middleware.logging import AccessLoggingMiddleware
from app.gateway.middleware.rate_limit import limiter, rate_limit_exceeded_handler
from app.gateway.middleware.request_id import RequestIDMiddleware
from app.gateway.routes import admin, chat, health, rag, sessions, skills, tools
from app.gateway.security import SecurityHeadersMiddleware, configure_cors
from app.infra.clients.ollama import OllamaClient
from app.infra.clients.postgres import PostgresClient
from app.infra.clients.qdrant import QdrantClient
from app.infra.clients.redis import RedisClient
from app.infra.clients.reranker import RerankerClient
from app.infra.clients.telemetry import build_telemetry_client
from app.services.embeddings.bge_m3 import BgeM3EmbeddingService
from app.services.embeddings.cache import CachedEmbeddingService
from app.services.embeddings.sparse import Bm42SparseEmbeddingService
from app.services.llm.factory import build_llm_provider
from app.services.llm.structured import StructuredOutputService
from app.services.memory.base import MemoryManager
from app.services.memory.consolidation import ConsolidationService
from app.services.memory.episodic import EpisodicMemory
from app.services.memory.long_term import LongTermMemory
from app.services.memory.scoring import ImportanceScorer
from app.services.memory.short_term import ShortTermMemory
from app.services.rag.chunking import SemanticChunker
from app.services.rag.ingestion import IngestionPipeline
from app.services.rag.loaders import default_loaders
from app.services.rag.query_transform import QueryTransformer
from app.services.rag.retrieval import HybridRetriever
from app.services.reranker.local_mps import LocalMPSReranker
from app.services.sandbox.docker_sandbox import DockerSandbox
from app.skills.registry import SkillRegistry
from app.skills.router import SkillRouter
from app.tools.builtins import build_builtin_tools
from app.tools.gate import ToolGate
from app.tools.mcp.adapter import adapt_mcp_tools
from app.tools.mcp.client import McpClient
from app.tools.registry import ToolRegistry

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

    # 2. Ollama client — one per process, shared by the LLM provider (in
    #    ollama mode) and the embedding service (bge-m3 is always local).
    #    The lifespan owns it and closes it on shutdown.
    ollama = OllamaClient(settings.ollama)

    # 3. LLM provider (protocol-based — swap via LLM_PROVIDER).
    #    Reuses the shared client in ollama mode instead of opening a second pool.
    llm = build_llm_provider(settings.llm, settings.ollama, ollama_client=ollama)

    # 4. Embedding service (bge-m3 dense + Redis cache) — always via local Ollama
    bge_m3 = BgeM3EmbeddingService(ollama, settings.llm.embed_model)
    embeddings = CachedEmbeddingService(bge_m3, redis)

    # 5. Reranker (lazy-load local MPS; no-op when RERANKER_MODE=none)
    reranker_client = RerankerClient(settings.reranker)
    reranker = LocalMPSReranker(reranker_client, settings.reranker)

    # 6. Telemetry (Langfuse Cloud or no-op)
    telemetry = build_telemetry_client(settings.telemetry)
    set_telemetry_client(telemetry)

    # 7. RAG pipeline (Phase 2) — sparse leg via FastEmbed, hybrid retrieval,
    #    idempotent ingestion. Sparse model loads lazily on first use.
    sparse_embeddings = Bm42SparseEmbeddingService(settings.sparse_model)
    chunker = SemanticChunker(embeddings)
    rag_retriever = HybridRetriever(
        qdrant=qdrant,
        dense_embedder=embeddings,
        sparse_embedder=sparse_embeddings,
        reranker=reranker,
        collection=settings.qdrant.collection_docs,
    )
    rag_ingestion = IngestionPipeline(
        loaders=default_loaders(),
        chunker=chunker,
        dense_embedder=embeddings,
        sparse_embedder=sparse_embeddings,
        qdrant=qdrant,
        postgres=postgres,
        collection=settings.qdrant.collection_docs,
    )
    rag_query_transformer = QueryTransformer(llm)

    # 8. Tools layer (Phase 3) — sandbox, builtin registry, ToolGate, MCP bridge.
    #    Builtins are wired with their deps; MCP-adapter tools are added after the
    #    client connects to any configured external servers.
    sandbox = DockerSandbox(settings.sandbox)
    tool_registry = ToolRegistry()
    tool_registry.register_all(
        build_builtin_tools(settings=settings, retriever=rag_retriever, sandbox=sandbox)
    )

    mcp_client = McpClient(settings.mcp.servers())
    await mcp_client.connect()
    for adapter in adapt_mcp_tools(mcp_client):
        tool_registry.register(adapter, replace=True)

    tool_gate = ToolGate(tool_registry)
    logger.info("tools_ready", count=len(tool_registry), tools=sorted(tool_registry.names()))

    # 9. Memory system (Phase 4) — three layers behind one facade, plus the
    #    background consolidation scheduler (APScheduler, ADR-012). Long-term
    #    reuses the cached dense embedder; scoring/extraction use structured LLM.
    structured_llm = StructuredOutputService(settings.llm, settings.ollama)
    short_term = ShortTermMemory(redis, llm, settings.memory)
    long_term = LongTermMemory(
        qdrant=qdrant,
        embedder=embeddings,
        settings=settings.memory,
        collection=settings.qdrant.collection_memory,
    )
    episodic = EpisodicMemory(postgres)
    memory = MemoryManager(short_term=short_term, long_term=long_term, episodic=episodic)
    importance_scorer = ImportanceScorer(structured_llm, redis)
    consolidation = ConsolidationService(
        short_term=short_term,
        long_term=long_term,
        episodic=episodic,
        scorer=importance_scorer,
        structured=structured_llm,
        settings=settings.memory,
    )
    consolidation.start()

    # 10. Orchestration (Phase 5) — Postgres checkpointer + compiled supervisor
    #     graph. The graph closes over its dependencies via GraphDeps; the
    #     checkpointer (ADR-008) gives it resume/replay across restarts.
    checkpointer = CheckpointerManager(settings.postgres)
    saver = await checkpointer.start()
    graph_deps = GraphDeps(
        llm=llm,
        structured=structured_llm,
        tool_registry=tool_registry,
        tool_gate=tool_gate,
        memory=memory,
        enforcer=BudgetEnforcer(),
        settings=settings,
    )
    agent_graph = build_graph(graph_deps, checkpointer=saver)
    logger.info("agent_graph_ready")

    # 11. Skills layer (Phase 6) — the public catalogue of high-level scenarios.
    #     The registry discovers YAML declarations (+ optional Python overrides);
    #     the router classifies free text to one skill (§5.3) using the same
    #     structured LLM the graph uses. Skills drive the compiled graph above.
    skill_registry = SkillRegistry().load()
    skill_router = SkillRouter(skill_registry, structured_llm)

    # 12. Attach to app.state for DI
    app.state.postgres = postgres
    app.state.redis = redis
    app.state.qdrant = qdrant
    app.state.ollama = ollama
    app.state.llm = llm
    app.state.embeddings = embeddings
    app.state.sparse_embeddings = sparse_embeddings
    app.state.reranker = reranker
    app.state.telemetry = telemetry
    app.state.rag_retriever = rag_retriever
    app.state.rag_ingestion = rag_ingestion
    app.state.rag_query_transformer = rag_query_transformer
    app.state.sandbox = sandbox
    app.state.tool_registry = tool_registry
    app.state.tool_gate = tool_gate
    app.state.mcp_client = mcp_client
    app.state.memory = memory
    app.state.consolidation = consolidation
    app.state.checkpointer = checkpointer
    app.state.agent_graph = agent_graph
    app.state.skill_registry = skill_registry
    app.state.skill_router = skill_router

    # Legacy alias: health.py (Phase 0) used db_pool name — keep until health.py is updated
    app.state.db_pool = postgres

    logger.info("startup_complete")

    yield  # ← application serves requests here

    # Shutdown — reverse order of initialisation
    logger.info("shutdown")

    await checkpointer.aclose()
    await consolidation.shutdown()
    telemetry.shutdown()
    await mcp_client.aclose()
    await reranker_client.aclose()
    await qdrant.aclose()
    await redis.aclose()
    await postgres.aclose()
    await ollama.aclose()

    logger.info("shutdown_complete")


# Application factory


def create_app() -> FastAPI:
    settings = get_settings()

    # Configure structlog — must run before any logger is used.
    configure_logging(
        log_level=settings.app.log_level,
        json_logs=settings.app.log_format == "json",
    )

    app = FastAPI(
        title="AI Agent",
        description="Production-grade multi-agent system — portfolio & personal assistant.",
        version="0.1.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # Rate limiting (slowapi): the limiter singleton is enforced through the
    # per-route @limiter.limit decorators (routes/chat.py, rag.py, tools.py). We
    # deliberately do NOT mount SlowAPIMiddleware: its default-limit path resolves
    # the endpoint via app.routes, which this FastAPI version keeps as lazy
    # _IncludedRouter wrappers, so it can't see /v1/* handlers. Decorators enforce
    # in-call and are unaffected. Register the limiter + a uniform JSON 429.
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # Middleware (order matters — outermost added last, so add inner→outer).
    # Effective request path (outer → inner):
    #   CORS → RequestID → AccessLogging → SecurityHeaders → Auth → routes
    # Rationale:
    #   • CORS outermost so preflight OPTIONS is answered before auth.
    #   • RequestID before AccessLogging so every access log carries the id.
    #   • SecurityHeaders outside Auth so 401/403/429 responses are stamped too.
    #   • Auth authenticates before the route-level limiter runs, so the limiter
    #     keys off the authenticated principal and unauth traffic gets 401.
    app.add_middleware(AuthMiddleware, settings=settings)
    app.add_middleware(SecurityHeadersMiddleware, settings=settings)
    app.add_middleware(AccessLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    configure_cors(app, settings.gateway)

    # Exception handlers
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # Routes
    app.include_router(health.router)
    app.include_router(rag.router, prefix="/v1")
    app.include_router(tools.router, prefix="/v1")
    app.include_router(chat.router, prefix="/v1")
    app.include_router(sessions.router, prefix="/v1")
    app.include_router(skills.router, prefix="/v1")
    app.include_router(admin.router, prefix="/v1")

    return app


# Module-level instance (uvicorn entry point)
app = create_app()
