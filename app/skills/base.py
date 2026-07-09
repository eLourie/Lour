"""
app/skills/base.py

The Skill abstraction: a declarative business scenario that drives the Phase-5
supervisor graph towards one chosen agent, under a per-skill policy.

  - ``SkillSpec``   — the validated declaration parsed from a ``*.yaml`` file:
    name, description, the agent it routes to, its tool allowlist, its input
    schema, a prompt template and its Policy block. ``effective_policy`` folds
    ``tools_allowed`` into ``Policy.allowed_tools`` so the act node / ToolGate
    enforce the skill's ACL for free (§5.4).
  - ``SkillContext`` — everything ``invoke`` needs at call time: the compiled
    graph, a thread id, the caller's inputs (or a raw free-text query for the
    ``/auto`` path) and an optional request-level policy override.
  - ``SkillResult``  — the structured outcome returned to the API.
  - ``Skill``        — the base class. ``invoke`` seeds the graph with a *forced*
    route (skill declares the agent → the supervisor does not route again, §5.3)
    and drives it to a final answer. ``postprocess`` is the override hook a
    Python implementation subclasses (e.g. review_code).

Skills declare policy (data); the graph and tools layer enforce it (code).
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.graphs.builder import initial_state

# AgentName is a Pydantic field annotation below → must resolve at runtime.
from app.agents.state import AgentName, Route
from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.core.policy import (
    ApprovalRules,
    BudgetPolicy,
    Policy,
    PolicyResolver,
    default_policy,
)

logger = get_logger(__name__)

# Supported input field types (kept small on purpose — skills take simple args).
_PY_TYPES: dict[str, type] = {"str": str, "int": int, "float": float, "bool": bool}
InputType = Literal["str", "int", "float", "bool"]


# Declaration


class InputField(BaseModel):
    """One field of a skill's input schema (parsed from YAML)."""

    type: InputType = "str"
    required: bool = True
    default: Any = None
    enum: list[str] | None = Field(
        default=None, description="If set, the value must be one of these choices."
    )
    description: str = ""


class SkillSpec(BaseModel):
    """The declarative definition of a skill (one ``definitions/*.yaml`` file)."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    description: str
    agent: AgentName
    tools_allowed: list[str] = Field(default_factory=list)
    input_schema: dict[str, InputField] = Field(default_factory=dict)
    output_schema: dict[str, str] = Field(default_factory=dict)
    # Template rendered with the validated inputs (str.format placeholders).
    prompt: str
    # Raw policy block from YAML; allowed_tools is derived from tools_allowed.
    budget: BudgetPolicy = Field(default_factory=BudgetPolicy)
    approval_rules: ApprovalRules = Field(default_factory=ApprovalRules)
    requires_confirmation: bool = False

    def effective_policy(self) -> Policy:
        """The skill's own Policy — its budget + allowlist derived from tools_allowed."""
        return Policy(
            budget=self.budget,
            allowed_tools=set(self.tools_allowed),
            approval_rules=self.approval_rules,
            requires_confirmation=self.requires_confirmation,
        )


# Runtime context + result


class SkillContext(BaseModel):
    """Inputs the runner hands to ``Skill.invoke``."""

    graph: Any = Field(description="Compiled supervisor graph (app.state.agent_graph).")
    thread_id: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    # Free-text query for the /auto path — bypasses input-schema templating.
    raw_query: str | None = None
    request_policy: Policy | None = None

    model_config = {"arbitrary_types_allowed": True}


class SkillResult(BaseModel):
    """The structured outcome of a skill invocation."""

    skill: str
    agent: str
    thread_id: str
    answer: str
    tools_called: list[str] = Field(default_factory=list)
    tokens_used: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


# Skill base class


class Skill:
    """
    Base skill: drives the supervisor graph towards ``spec.agent``.

    Subclass and override :meth:`postprocess` for skill-specific result shaping;
    the registry auto-discovers such subclasses in ``implementations/``.
    """

    def __init__(self, spec: SkillSpec) -> None:
        self.spec = spec

    # Convenience accessors

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def description(self) -> str:
        return self.spec.description

    @property
    def agent(self) -> AgentName:
        return self.spec.agent

    # Input handling

    def validate_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Fill defaults and check required / enum constraints. Raises ValidationError."""
        resolved: dict[str, Any] = {}
        for field_name, field in self.spec.input_schema.items():
            if field_name in inputs and inputs[field_name] is not None:
                value = inputs[field_name]
            elif field.default is not None:
                value = field.default
            elif field.required:
                raise ValidationError(
                    f"Skill {self.name!r} requires input {field_name!r}",
                    detail={"field": field_name},
                )
            else:
                continue

            if field.enum is not None and str(value) not in field.enum:
                raise ValidationError(
                    f"Input {field_name!r} must be one of {field.enum}, got {value!r}",
                    detail={"field": field_name, "allowed": field.enum},
                )
            resolved[field_name] = _PY_TYPES[field.type](value)
        return resolved

    def build_query(self, inputs: dict[str, Any]) -> str:
        """Render the skill's prompt template with the validated inputs."""
        resolved = self.validate_inputs(inputs)
        try:
            return self.spec.prompt.format(**resolved)
        except KeyError as exc:  # a placeholder without a matching input
            raise ValidationError(
                f"Skill {self.name!r} prompt references missing input {exc}",
                detail={"missing": str(exc)},
            ) from exc

    # Invocation

    async def invoke(self, ctx: SkillContext) -> SkillResult:
        """Run the skill end-to-end: seed a forced-route graph run, collect the answer."""
        query = ctx.raw_query if ctx.raw_query is not None else self.build_query(ctx.inputs)

        policy = PolicyResolver.resolve(
            defaults=default_policy(),
            skill=self.spec.effective_policy(),
            request=ctx.request_policy,
        )

        seed = initial_state(
            session_id=ctx.thread_id,
            thread_id=ctx.thread_id,
            query=query,
            policy=policy,
        )
        # Skill declares the agent → force the route so the supervisor's route
        # node short-circuits its LLM classification (§5.3, no double routing).
        seed["route"] = Route(agent=self.spec.agent, reasoning=f"skill:{self.name}").model_dump()

        started = time.perf_counter()
        config = {"configurable": {"thread_id": ctx.thread_id}}
        state = await ctx.graph.ainvoke(seed, config=config)
        elapsed_ms = (time.perf_counter() - started) * 1000

        result = self._result_from_state(ctx.thread_id, state)
        logger.info(
            "skill_invoke",
            skill=self.name,
            agent=result.agent,
            thread_id=ctx.thread_id,
            tokens=result.tokens_used,
            tool_calls=len(result.tools_called),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return await self.postprocess(result, state)

    def _result_from_state(self, thread_id: str, state: dict[str, Any]) -> SkillResult:
        route = state.get("route")
        budget = state.get("budget")
        tools_called = [getattr(r, "name", "") for r in state.get("tools_called") or []]
        return SkillResult(
            skill=self.name,
            agent=getattr(route, "agent", None) or self.spec.agent,
            thread_id=thread_id,
            answer=state.get("final_answer") or "",
            tools_called=tools_called,
            tokens_used=getattr(budget, "tokens_used", 0),
        )

    async def postprocess(self, result: SkillResult, state: dict[str, Any]) -> SkillResult:
        """Override hook for Python implementations. Default: return unchanged."""
        return result
