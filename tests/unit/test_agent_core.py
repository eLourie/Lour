"""
tests/unit/test_agent_core.py

Pure-logic unit coverage for the agent core: the BudgetEnforcer (limits + loop
detection), BudgetState arithmetic, and AgentEvent SSE serialisation. No I/O.
"""

from __future__ import annotations

import json

import pytest

from app.agents.enforcement.budget import BudgetEnforcer
from app.agents.events import AgentEvent, EventType
from app.agents.state import AgentState, BudgetState, ToolCallRecord
from app.core.policy import BudgetPolicy, Policy

pytestmark = pytest.mark.unit


def _state(
    policy: Policy, budget: BudgetState, tools: list[ToolCallRecord] | None = None
) -> AgentState:
    return AgentState(
        session_id="s",
        thread_id="t",
        query="q",
        policy=policy,
        budget=budget,
        tools_called=tools or [],
    )


def test_budget_within_limits_can_continue() -> None:
    enforcer = BudgetEnforcer()
    policy = Policy(budget=BudgetPolicy(max_iterations=5, max_tool_calls=10))
    verdict = enforcer.check(_state(policy, BudgetState(iterations=2, tool_calls_made=3)))
    assert verdict.can_continue
    assert verdict.kind == "ok"


def test_iteration_limit_stops() -> None:
    enforcer = BudgetEnforcer()
    policy = Policy(budget=BudgetPolicy(max_iterations=3))
    verdict = enforcer.check(_state(policy, BudgetState(iterations=3)))
    assert not verdict.can_continue
    assert verdict.kind == "budget"


def test_token_limit_stops() -> None:
    enforcer = BudgetEnforcer()
    policy = Policy(budget=BudgetPolicy(max_cost_tokens=100))
    verdict = enforcer.check(_state(policy, BudgetState(tokens_used=150)))
    assert not verdict.can_continue
    assert "max_cost_tokens" in (verdict.reason or "")


def test_loop_detection_stops_on_identical_calls() -> None:
    enforcer = BudgetEnforcer(loop_window=3)
    same = [ToolCallRecord(name="rag_query", arguments={"query": "x"}) for _ in range(3)]
    policy = Policy(budget=BudgetPolicy(max_iterations=100))
    verdict = enforcer.check(_state(policy, BudgetState(iterations=3), tools=same))
    assert not verdict.can_continue
    assert verdict.kind == "loop"


def test_loop_detection_ignores_varied_calls() -> None:
    enforcer = BudgetEnforcer(loop_window=3)
    varied = [ToolCallRecord(name="rag_query", arguments={"query": str(i)}) for i in range(3)]
    assert enforcer.detect_loop(varied) is False


def test_budget_state_with_delta_and_elapsed() -> None:
    b = BudgetState(tokens_used=10, tool_calls_made=1, iterations=1)
    nb = b.with_delta(tokens=5, tool_calls=2, iterations=1)
    assert (nb.tokens_used, nb.tool_calls_made, nb.iterations) == (15, 3, 2)
    # original untouched (nodes never mutate in place)
    assert b.tokens_used == 10
    assert nb.elapsed_s() >= 0.0


def test_event_to_sse_shape() -> None:
    frame = AgentEvent.route_decided("coder", "because code").to_sse()
    assert frame.startswith("event: route_decided\n")
    body = frame.split("data: ", 1)[1].strip()
    payload = json.loads(body)
    assert payload["agent"] == "coder"
    assert payload["node"] == "route"
    assert frame.endswith("\n\n")


def test_event_types_serialise_by_value() -> None:
    assert AgentEvent.token("hi", node="direct").type == EventType.TOKEN
    tool = AgentEvent.tool_result("code_exec", ok=False, error="boom")
    payload = json.loads(tool.to_sse().split("data: ", 1)[1].strip())
    assert payload["ok"] is False
    assert payload["error"] == "boom"
