"""
app/gateway/routes/health.py

Liveness and readiness probes.

GET /healthz  — liveness: is the process running? Always 200 if the app started.
GET /readyz   — readiness: are all backing services reachable?
               Returns 200 with per-service status, or 503 if any are down.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", status_code=status.HTTP_200_OK)
async def liveness() -> dict[str, str]:
    """Process is alive."""
    return {"status": "ok"}


@router.get("/readyz")
async def readiness(request: Request) -> JSONResponse:
    """
    Check all backing-service dependencies.

    Returns 200 if every service is reachable, 503 otherwise.
    Each service gets its own status entry so the caller can see which one failed.
    """
    start = time.perf_counter()
    checks: dict[str, str] = {}
    healthy = True

    # Postgres
    postgres = getattr(request.app.state, "postgres", None)
    if postgres is not None:
        ok = await postgres.ping()
        checks["postgres"] = "ok" if ok else "unreachable"
        if not ok:
            healthy = False
    else:
        checks["postgres"] = "not_initialised"
        healthy = False

    # Redis
    redis = getattr(request.app.state, "redis", None)
    if redis is not None:
        ok = await redis.ping_all()
        checks["redis"] = "ok" if ok else "unreachable"
        if not ok:
            healthy = False
    else:
        checks["redis"] = "not_initialised"
        healthy = False

    # Qdrant
    qdrant = getattr(request.app.state, "qdrant", None)
    if qdrant is not None:
        ok = await qdrant.ping()
        checks["qdrant"] = "ok" if ok else "unreachable"
        if not ok:
            healthy = False
    else:
        checks["qdrant"] = "not_initialised"
        healthy = False

    # Ollama (optional in cloud-provider mode — warn, don't fail readiness)
    ollama = getattr(request.app.state, "ollama", None)
    if ollama is not None:
        ok = await ollama.ping()
        checks["ollama"] = "ok" if ok else "unreachable"
        # Ollama unreachable is a warning, not a hard failure:
        # cloud provider mode runs without it.
    else:
        checks["ollama"] = "not_used"  # cloud provider path

    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    status_str = "ok" if healthy else "degraded"

    return JSONResponse(
        status_code=status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": status_str, "checks": checks, "elapsed_ms": elapsed_ms},
    )
