"""
tests/unit/test_skills.py

Unit coverage for the skills layer (Phase 6) with *stubbed* LLM/memory and a real
supervisor graph, registry and router — no backing services. Validates the wiring
the live DoD depends on: a skill forces its declared agent (no re-routing, §5.3),
derives its tool allowlist into the effective policy, validates inputs, discovers
Python overrides, and the router recovers from an out-of-catalogue LLM answer.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from app.agents.deps import GraphDeps
from app.agents.enforcement.budget import BudgetEnforcer
from app.agents.graphs.builder import build_graph
from app.agents.state import Plan, Reflection, Route
from app.core.config import get_settings
from app.core.exceptions import ValidationError
from app.services.llm.base import LLMResponse
from app.services.memory.base import MemoryContext
from app.skills.base import Skill, SkillContext, SkillSpec
from app.skills.registry import SkillRegistry
from app.skills.router import SkillDecision, SkillRouter
from app.tools.gate import ToolGate
from app.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


# Stubs (mirroring tests/unit/test_agent_graph.py)


class _StubLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]

    def stream(self, messages, *, options=None):  # pragma: no cover
        raise NotImplementedError

    async def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        return [[0.0] for _ in texts]


class _StubStructured:
    def __init__(self, *, reflection: Reflection | None = None) -> None:
        self._reflection = reflection or Reflection(is_complete=True, reasoning="ok")

    async def complete(
        self, messages: list[dict[str, Any]], schema: type[BaseModel], *, context: str = ""
    ) -> Any:
        if schema is Route:  # should never be called for a forced skill route
            raise AssertionError("route node was invoked despite a forced route")
        if schema is Plan:
            return Plan(steps=["gather", "summarise"])
        if schema is Reflection:
            return self._reflection
        raise AssertionError(f"unexpected schema {schema}")


class _StubMemory:
    async def recall(
        self, session_id: str, query: str, *, top_k: int | None = None
    ) -> MemoryContext:
        return MemoryContext()

    async def write(self, session_id: str, role: str, content: str) -> None:
        return None


def _make_deps() -> GraphDeps:
    registry = ToolRegistry()
    return GraphDeps(
        llm=_StubLLM(
            [LLMResponse(content="A concise report.", prompt_tokens=4, completion_tokens=6)]
        ),  # type: ignore[arg-type]
        structured=_StubStructured(),  # type: ignore[arg-type]
        tool_registry=registry,
        tool_gate=ToolGate(registry),
        memory=_StubMemory(),  # type: ignore[arg-type]
        enforcer=BudgetEnforcer(),
        settings=get_settings(),
    )


def _graph() -> Any:
    return build_graph(_make_deps(), checkpointer=InMemorySaver())


# Spec / input handling


def _spec(**over: Any) -> SkillSpec:
    base: dict[str, Any] = {
        "name": "demo_skill",
        "description": "A demo.",
        "agent": "researcher",
        "tools_allowed": ["rag_query", "web_search"],
        "input_schema": {
            "topic": {"type": "str", "required": True},
            "depth": {
                "type": "str",
                "required": False,
                "default": "quick",
                "enum": ["quick", "deep"],
            },
        },
        "prompt": "Research {topic} at {depth} depth.",
        "budget": {"max_cost_tokens": 1000},
    }
    base.update(over)
    return SkillSpec.model_validate(base)


def test_effective_policy_derives_allowlist_from_tools() -> None:
    policy = _spec().effective_policy()
    assert policy.allowed_tools == {"rag_query", "web_search"}
    assert policy.budget.max_cost_tokens == 1000


def test_validate_inputs_fills_defaults() -> None:
    resolved = Skill(_spec()).validate_inputs({"topic": "rust"})
    assert resolved == {"topic": "rust", "depth": "quick"}


def test_validate_inputs_missing_required_raises() -> None:
    with pytest.raises(ValidationError):
        Skill(_spec()).validate_inputs({"depth": "deep"})


def test_validate_inputs_enum_rejects_bad_value() -> None:
    with pytest.raises(ValidationError):
        Skill(_spec()).validate_inputs({"topic": "x", "depth": "shallow"})


def test_build_query_renders_template() -> None:
    assert Skill(_spec()).build_query({"topic": "rust"}) == "Research rust at quick depth."


# Invocation drives the graph with a forced route


async def test_skill_forces_declared_agent() -> None:
    skill = Skill(_spec(agent="researcher"))
    ctx = SkillContext(graph=_graph(), thread_id="sk1", inputs={"topic": "rust"})
    result = await skill.invoke(ctx)

    assert result.skill == "demo_skill"
    assert result.agent == "researcher"  # forced, not re-routed
    assert result.answer == "A concise report."


async def test_skill_auto_raw_query_bypasses_input_schema() -> None:
    # /auto has no structured inputs — a raw query must work even though the
    # skill declares a required 'topic'.
    skill = Skill(_spec())
    ctx = SkillContext(graph=_graph(), thread_id="sk2", raw_query="just answer this")
    result = await skill.invoke(ctx)
    assert result.answer == "A concise report."


# Registry + override discovery


def test_registry_loads_definitions_and_override() -> None:
    reg = SkillRegistry().load()
    assert set(reg.names()) >= {
        "research_topic",
        "review_code",
        "answer_from_kb",
        "summarize_document",
    }
    # review_code ships a Python override subclass.
    assert type(reg.get("review_code")).__name__ == "ReviewCodeSkill"
    assert reg.get("research_topic").agent == "researcher"


def test_registry_get_unknown_raises() -> None:
    from app.core.exceptions import NotFoundError

    with pytest.raises(NotFoundError):
        SkillRegistry().load().get("nope")


# Router


class _RouterStructured:
    def __init__(self, returned: str) -> None:
        self._returned = returned

    async def complete(
        self, messages: list[dict[str, Any]], schema: type[BaseModel], *, context: str = ""
    ) -> Any:
        return SkillDecision(skill=self._returned, reasoning="stub")


async def test_router_returns_registered_skill() -> None:
    reg = SkillRegistry().load()
    router = SkillRouter(reg, _RouterStructured("answer_from_kb"))  # type: ignore[arg-type]
    decision = await router.classify("what does the doc say about X?")
    assert decision.skill == "answer_from_kb"


async def test_router_falls_back_when_llm_names_unknown_skill() -> None:
    reg = SkillRegistry().load()
    router = SkillRouter(reg, _RouterStructured("does_not_exist"))  # type: ignore[arg-type]
    decision = await router.classify("please review this python code for bugs")
    assert decision.skill in reg  # recovered to a real skill
    assert "fallback" in decision.reasoning
