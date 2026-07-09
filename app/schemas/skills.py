"""
app/schemas/skills.py

API DTOs for the skills routes (/v1/skills catalog, /invoke, /auto).

The catalog is the *public* face of what the agent can do (§5.1), so these
models expose each skill's inputs and the agent it drives — enough for a client
to build an invocation form — while keeping the internal Policy/prompt details
out of the wire contract.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InputFieldInfo(BaseModel):
    """One input field of a skill, as advertised in the catalog."""

    type: str
    required: bool
    default: Any = None
    enum: list[str] | None = None
    description: str = ""


class SkillInfo(BaseModel):
    """A skill's public description in the catalog."""

    name: str
    description: str
    agent: str
    tools_allowed: list[str]
    input_schema: dict[str, InputFieldInfo]
    requires_confirmation: bool


class SkillCatalogResponse(BaseModel):
    count: int
    skills: list[SkillInfo]


class SkillInvokeResponse(BaseModel):
    """The outcome of a skill invocation (mirrors skills.base.SkillResult)."""

    skill: str
    agent: str
    thread_id: str
    answer: str
    tools_called: list[str] = Field(default_factory=list)
    tokens_used: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class AutoRouteRequest(BaseModel):
    """Free-text request for /auto: classify (and optionally run) the best skill."""

    text: str = Field(min_length=1, description="The free-form user request.")
    invoke: bool = Field(
        default=True,
        description="If true, also run the chosen skill with the text as its query.",
    )


class AutoRouteResponse(BaseModel):
    skill: str
    reasoning: str = ""
    result: SkillInvokeResponse | None = None
