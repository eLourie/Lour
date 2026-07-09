"""
app/gateway/routes/skills.py

/v1/skills — the public catalogue of high-level capabilities (§5.1).

    GET  /v1/skills                 — list every registered skill and its inputs.
    POST /v1/skills/{name}/invoke   — run a skill with structured inputs.
    POST /v1/skills/auto            — classify free text to a skill (§5.3) and,
                                      by default, run it with the text as query.

Unlike /v1/chat, invocation is synchronous JSON (not SSE): a skill is a named
scenario the caller drives to a single result. Each invocation opens a Session
tagged with the skill name, giving per-skill traceability for free.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Body, Depends, Request

from app.core.di import get_state
from app.infra.db.models.session import Session
from app.infra.db.unit_of_work import UnitOfWork
from app.schemas.skills import (
    AutoRouteRequest,
    AutoRouteResponse,
    InputFieldInfo,
    SkillCatalogResponse,
    SkillInfo,
    SkillInvokeResponse,
)
from app.skills.base import SkillContext

if TYPE_CHECKING:
    from app.infra.clients.postgres import PostgresClient
    from app.skills.base import Skill, SkillResult
    from app.skills.registry import SkillRegistry
    from app.skills.router import SkillRouter

router = APIRouter(prefix="/skills", tags=["skills"])

RegistryDep = Annotated["SkillRegistry", Depends(get_state("skill_registry"))]
RouterDep = Annotated["SkillRouter", Depends(get_state("skill_router"))]
PostgresDep = Annotated["PostgresClient", Depends(get_state("postgres"))]


def _skill_info(skill: Skill) -> SkillInfo:
    return SkillInfo(
        name=skill.name,
        description=skill.description,
        agent=skill.agent,
        tools_allowed=skill.spec.tools_allowed,
        input_schema={
            field_name: InputFieldInfo(
                type=field.type,
                required=field.required,
                default=field.default,
                enum=field.enum,
                description=field.description,
            )
            for field_name, field in skill.spec.input_schema.items()
        },
        requires_confirmation=skill.spec.requires_confirmation,
    )


def _to_response(result: SkillResult) -> SkillInvokeResponse:
    return SkillInvokeResponse(**result.model_dump())


async def _open_session(postgres: PostgresClient, thread_id: str, skill_name: str) -> None:
    """Create the Session row for this invocation, tagged with the skill name."""
    async with UnitOfWork(postgres) as uow:
        if await uow.sessions.get_by_thread_id(thread_id) is None:
            await uow.sessions.add(Session(thread_id=thread_id, skill_name=skill_name))


@router.get("", response_model=SkillCatalogResponse)
async def list_skills(registry: RegistryDep) -> SkillCatalogResponse:
    """List the public catalogue of skills."""
    skills = sorted(registry.all(), key=lambda s: s.name)
    return SkillCatalogResponse(count=len(skills), skills=[_skill_info(s) for s in skills])


@router.post("/{name}/invoke", response_model=SkillInvokeResponse)
async def invoke_skill(
    name: str,
    request: Request,
    registry: RegistryDep,
    postgres: PostgresDep,
    inputs: Annotated[dict[str, Any], Body(default_factory=dict)],
) -> SkillInvokeResponse:
    """Run a skill with the given inputs and return its structured result."""
    skill = registry.get(name)  # 404 if unknown
    graph: Any = request.app.state.agent_graph

    thread_id = uuid.uuid4().hex
    await _open_session(postgres, thread_id, name)

    ctx = SkillContext(graph=graph, thread_id=thread_id, inputs=inputs)
    result = await skill.invoke(ctx)  # validation errors → 422
    return _to_response(result)


@router.post("/auto", response_model=AutoRouteResponse)
async def auto_route(
    req: AutoRouteRequest,
    request: Request,
    registry: RegistryDep,
    skill_router: RouterDep,
    postgres: PostgresDep,
) -> AutoRouteResponse:
    """Classify free text to the best skill, and (by default) run it."""
    decision = await skill_router.classify(req.text)

    result_dto: SkillInvokeResponse | None = None
    if req.invoke:
        skill = registry.get(decision.skill)
        graph: Any = request.app.state.agent_graph
        thread_id = uuid.uuid4().hex
        await _open_session(postgres, thread_id, decision.skill)
        # Free text has no structured inputs → feed it straight in as the query.
        ctx = SkillContext(graph=graph, thread_id=thread_id, raw_query=req.text)
        result_dto = _to_response(await skill.invoke(ctx))

    return AutoRouteResponse(skill=decision.skill, reasoning=decision.reasoning, result=result_dto)
