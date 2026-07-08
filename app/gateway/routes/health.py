"""
app/gateway/routes/health.py

Health / readiness probes.

GET /healthz  — liveness: returns 200 if the process is alive.
GET /readyz   — readiness: pings each backing service; returns 200 only if
                ALL are reachable, otherwise 503 with per-service detail.

Used by:
  - Docker / k8s health checks
  - CI acceptance gates
  - `make up` (--wait)
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["health"])


# /healthz — liveness


@router.get("/healthz", include_in_schema=False)
async def liveness() -> dict[str, str]:
    return {"status": "ok"}


# /readyz — readiness (checks all backing services)


@router.get("/readyz", include_in_schema=False)
async def readiness(request: Request) -> JSONResponse:
    checks = await asyncio.gather(
        _check_postgres(request),
        _check_redis(request),
        _check_qdrant(request),
        _check_ollama(request),
        return_exceptions=True,
    )

    results: dict[str, Any] = {}
    names = ("postgres", "redis", "qdrant", "ollama")
    all_ok = True

    for name, result in zip(names, checks, strict=True):
        if isinstance(result, BaseException):
            results[name] = {"ok": False, "error": str(result)}
            all_ok = False
        elif isinstance(result, dict):
            results[name] = result
            if not result.get("ok", False):
                all_ok = False
        else:
            results[name] = {"ok": False, "error": "unexpected result type"}
            all_ok = False

    status_code = 200 if all_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ok" if all_ok else "degraded", "services": results},
    )


# Individual service checks


async def _check_postgres(request: Request) -> dict[str, Any]:
    try:
        pool = getattr(request.app.state, "db_pool", None)
        if pool is None:
            return {"ok": False, "error": "pool not initialised"}
        async with pool.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        return {"ok": True}
    except Exception as exc:
        logger.debug("readyz.postgres_fail", error=str(exc))
        return {"ok": False, "error": str(exc)}


async def _check_redis(request: Request) -> dict[str, Any]:
    try:
        redis = getattr(request.app.state, "redis", None)
        if redis is None:
            return {"ok": False, "error": "client not initialised"}
        await redis.ping()
        return {"ok": True}
    except Exception as exc:
        logger.debug("readyz.redis_fail", error=str(exc))
        return {"ok": False, "error": str(exc)}


async def _check_qdrant(request: Request) -> dict[str, Any]:
    try:
        qdrant = getattr(request.app.state, "qdrant", None)
        if qdrant is None:
            return {"ok": False, "error": "client not initialised"}
        await qdrant.get_collections()
        return {"ok": True}
    except Exception as exc:
        logger.debug("readyz.qdrant_fail", error=str(exc))
        return {"ok": False, "error": str(exc)}


async def _check_ollama(request: Request) -> dict[str, Any]:
    try:
        from app.core.config import get_settings

        settings = get_settings()
        url = settings.ollama.base_url
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url}/api/tags")
            resp.raise_for_status()
        return {"ok": True}
    except Exception as exc:
        logger.debug("readyz.ollama_fail", error=str(exc))
        return {"ok": False, "error": str(exc)}
