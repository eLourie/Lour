"""
tests/unit/test_agent_graph.py

End-to-end exercise of the supervisor graph with *stubbed* LLM/memory and real
nodes, enforcer, ToolRegistry and ToolGate — no backing services. This validates
the wiring the live DoD depends on: routing to each agent, the act tool loop,
reducers accumulating messages/tool-calls, budget/loop enforcement and the
memory read/write bookends.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from app.agents.deps import GraphDeps
from app.agents.enforcement.budget import BudgetEnforcer
from app.agents.graphs.builder import build_graph, initial_state
from app.agents.state import Plan, Reflection, Route
from app.core.config import get_settings
from app.core.policy import BudgetPolicy, Policy
from app.services.llm.base import LLMResponse
from app.services.memory.base import MemoryContext
from app.tools.base import BaseTool, ToolResult
from app.tools.gate import ToolGate
from app.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


# Stubs


class _StubLLM:
    """Returns scripted chat responses in order (one per act/direct call)."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]

    def stream(self, messages, *, options=None):  # pragma: no cover - unused here
        raise NotImplementedError

    async def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        return [[0.0] for _ in texts]


class _StubStructured:
    """Serves Route / Plan / Reflection instances by requested schema."""

    def __init__(
        self,
        *,
        agent: str = "direct",
        plan: list[str] | None = None,
        reflection: Reflection | None = None,
    ) -> None:
        self._agent = agent
        self._plan = plan or ["gather", "summarise"]
        self._reflection = reflection or Reflection(is_complete=True, reasoning="ok")

    async def complete(
        self, messages: list[dict[str, Any]], schema: type[BaseModel], *, context: str = ""
    ) -> Any:
        if schema is Route:
            return Route(agent=self._agent, reasoning="stub")  # type: ignore[arg-type]
        if schema is Plan:
            return Plan(steps=self._plan)
        if schema is Reflection:
            return self._reflection
        raise AssertionError(f"unexpected schema {schema}")


class _StubMemory:
    def __init__(self) -> None:
        self.writes: list[tuple[str, str]] = []

    async def recall(self, session_id: str, query: str, *, top_k: int | None = None) -> MemoryContext:
        return MemoryContext()

    async def write(self, session_id: str, role: str, content: str) -> None:
        self.writes.append((role, content))


class _EchoCodeExec(BaseTool):
    name = "code_exec"
    description = "Fake sandbox that echoes a fixed result for testing."
    args_schema = type("Args", (BaseModel,), {"__annotations__": {"code": str}})
    side_effects = True

    async def execute(self, args: Any) -> ToolResult:
        return ToolResult.success({"stdout": "42", "exit_code": 0})


def _make_deps(llm: _StubLLM, structured: _StubStructured, memory: _StubMemory) -> GraphDeps:
    registry = ToolRegistry()
    registry.register(_EchoCodeExec())
    return GraphDeps(
        llm=llm,  # type: ignore[arg-type]
        structured=structured,  # type: ignore[arg-type]
        tool_registry=registry,
        tool_gate=ToolGate(registry),
        memory=memory,  # type: ignore[arg-type]
        enforcer=BudgetEnforcer(),
        settings=get_settings(),
    )


def _config(thread: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread}}


# Tests


async def test_direct_route_answers_without_tools() -> None:
    llm = _StubLLM([LLMResponse(content="Hello there!", prompt_tokens=3, completion_tokens=2)])
    memory = _StubMemory()
    deps = _make_deps(llm, _StubStructured(agent="direct"), memory)
    graph = build_graph(deps, checkpointer=InMemorySaver())

    seed = initial_state(session_id="s1", thread_id="t1", query="hi")
    result = await graph.ainvoke(seed, config=_config("t1"))

    assert result["route"].agent == "direct"
    assert result["final_answer"] == "Hello there!"
    assert result["finished"] is True
    assert ("assistant", "Hello there!") in memory.writes


async def test_coder_route_runs_tool_then_answers() -> None:
    llm = _StubLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[{"name": "code_exec", "arguments": {"code": "print(42)"}}],
                prompt_tokens=5,
                completion_tokens=5,
            ),
            LLMResponse(content="The answer is 42.", prompt_tokens=4, completion_tokens=4),
        ]
    )
    deps = _make_deps(llm, _StubStructured(agent="coder"), _StubMemory())
    graph = build_graph(deps, checkpointer=InMemorySaver())

    seed = initial_state(session_id="s2", thread_id="t2", query="compute 6*7 in python")
    result = await graph.ainvoke(seed, config=_config("t2"))

    assert result["route"].agent == "coder"
    assert result["final_answer"] == "The answer is 42."
    names = [r.name for r in result["tools_called"]]
    assert "code_exec" in names
    assert result["budget"].tool_calls_made == 1


async def test_researcher_route_plans_and_finishes() -> None:
    llm = _StubLLM(
        [LLMResponse(content="Rust is a systems language.", prompt_tokens=6, completion_tokens=6)]
    )
    deps = _make_deps(
        llm,
        _StubStructured(agent="researcher", reflection=Reflection(is_complete=True, reasoning="done")),
        _StubMemory(),
    )
    graph = build_graph(deps, checkpointer=InMemorySaver())

    seed = initial_state(session_id="s3", thread_id="t3", query="what is rust")
    result = await graph.ainvoke(seed, config=_config("t3"))

    assert result["route"].agent == "researcher"
    assert result["plan"]  # plan node produced steps
    assert result["final_answer"] == "Rust is a systems language."


async def test_unknown_tool_is_fed_back_not_crash() -> None:
    llm = _StubLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[{"name": "ghost_tool", "arguments": {}}],
                prompt_tokens=2,
                completion_tokens=2,
            ),
            LLMResponse(content="Recovered.", prompt_tokens=2, completion_tokens=2),
        ]
    )
    deps = _make_deps(llm, _StubStructured(agent="coder"), _StubMemory())
    graph = build_graph(deps, checkpointer=InMemorySaver())

    seed = initial_state(session_id="s4", thread_id="t4", query="do a thing")
    result = await graph.ainvoke(seed, config=_config("t4"))

    records = result["tools_called"]
    assert any(r.name == "ghost_tool" and r.error == "unknown_tool" for r in records)
    assert result["final_answer"] == "Recovered."


async def test_budget_iteration_cap_forces_finish() -> None:
    # The model always asks for the same tool → without a cap it would loop
    # forever. A tight max_iterations forces finalisation.
    llm = _StubLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[{"name": "code_exec", "arguments": {"code": "print(1)"}}],
                prompt_tokens=1,
                completion_tokens=1,
            )
        ]
    )
    deps = _make_deps(llm, _StubStructured(agent="coder"), _StubMemory())
    graph = build_graph(deps, checkpointer=InMemorySaver())

    policy = Policy(budget=BudgetPolicy(max_iterations=3))
    seed = initial_state(session_id="s5", thread_id="t5", query="loop", policy=policy)
    result = await graph.ainvoke(seed, config=_config("t5"))

    assert result["finished"] is True
    assert result["final_answer"]  # finalize guaranteed an answer
    assert result["budget"].iterations <= 4
