"""
app/gateway/routes/admin.py

/v1/admin — operator endpoints, gated behind the admin key.

    GET  /v1/admin/metrics          — a snapshot of registry sizes + active config.
    POST /v1/admin/reload-skills     — re-discover skill YAML without a restart.
    POST /v1/admin/cache-invalidate  — flush the cache logical Redis DB.

Every handler depends on ``require_admin``: AuthMiddleware has already
authenticated the caller and attached a Principal; the dependency rejects any
principal that is not admin with 403. There is no separate admin key check here —
the user/admin distinction is decided once, at authentication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Request

from app.core.config import Settings, get_settings
from app.core.metrics import get_metrics
from app.core.security import Principal
from app.gateway.middleware.auth import require_admin

if TYPE_CHECKING:
    from app.infra.clients.redis import RedisClient
    from app.skills.registry import SkillRegistry
    from app.tools.registry import ToolRegistry

router = APIRouter(prefix="/admin", tags=["admin"])

AdminDep = Annotated[Principal, Depends(require_admin)]


@router.get("/metrics")
async def metrics(request: Request, _: AdminDep) -> dict[str, Any]:
    """Return a lightweight operational snapshot for dashboards / probes."""
    state = request.app.state
    settings: Settings = get_settings()

    tool_registry: ToolRegistry | None = getattr(state, "tool_registry", None)
    skill_registry: SkillRegistry | None = getattr(state, "skill_registry", None)

    return {
        "tools": {
            "count": len(tool_registry) if tool_registry is not None else 0,
            "names": sorted(tool_registry.names()) if tool_registry is not None else [],
        },
        "skills": {
            "count": len(skill_registry) if skill_registry is not None else 0,
            "names": sorted(skill_registry.names()) if skill_registry is not None else [],
        },
        "config": {
            "deploy_profile": str(settings.deploy_profile),
            "llm_provider": str(settings.llm.provider),
            "llm_model": settings.llm.main_model,
            "auth_mode": str(settings.app.auth_mode),
            "reranker_mode": str(settings.reranker.mode),
        },
        "runtime": get_metrics().snapshot(),
    }


@router.post("/reload-skills")
async def reload_skills(request: Request, _: AdminDep) -> dict[str, Any]:
    """Re-discover skill declarations from disk without restarting the process.

    ``SkillRegistry.load()`` reloads in place and the SkillRouter holds a live
    reference to the same registry, so both see the new catalogue immediately.
    """
    skill_registry: SkillRegistry = request.app.state.skill_registry
    skill_registry.load()
    return {"status": "reloaded", "count": len(skill_registry), "skills": skill_registry.names()}


@router.post("/cache-invalidate")
async def cache_invalidate(request: Request, _: AdminDep) -> dict[str, Any]:
    """Flush the cache logical Redis DB (embedding + tool-result caches)."""
    redis: RedisClient = request.app.state.redis
    await redis.cache.flushdb()
    return {"status": "invalidated", "scope": "cache"}
