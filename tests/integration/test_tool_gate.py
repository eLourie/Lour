"""
tests/integration/test_tool_gate.py

ToolGate enforcement: allowlist + HITL approval (ADR-011). Pure policy logic —
no backing services, so it runs in the default suite too.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.core.exceptions import PolicyViolationError
from app.core.policy import ApprovalRules, Policy
from app.tools.base import BaseTool, ToolResult
from app.tools.gate import ToolGate
from app.tools.registry import ToolRegistry


class _NoArgs(BaseModel):
    pass


class _ReadTool(BaseTool[_NoArgs]):
    name = "safe_read"
    description = "A read-only tool for tests."
    args_schema = _NoArgs

    async def execute(self, args: _NoArgs) -> ToolResult:
        return ToolResult.success()


class _WriteTool(BaseTool[_NoArgs]):
    name = "danger_write"
    description = "A side-effecting tool for tests."
    args_schema = _NoArgs
    side_effects = True

    async def execute(self, args: _NoArgs) -> ToolResult:
        return ToolResult.success()


@pytest.fixture
def gate() -> ToolGate:
    reg = ToolRegistry()
    reg.register(_ReadTool())
    reg.register(_WriteTool())
    return ToolGate(reg)


def test_allowlist_permits_listed_tool(gate: ToolGate) -> None:
    policy = Policy(allowed_tools={"safe_read"})
    decision = gate.check("safe_read", policy)
    assert decision.allowed
    assert not decision.requires_approval


def test_allowlist_blocks_unlisted_tool(gate: ToolGate) -> None:
    policy = Policy(allowed_tools={"safe_read"})
    decision = gate.check("danger_write", policy)
    assert not decision.allowed
    assert "not permitted" in (decision.reason or "")


def test_unknown_tool_is_blocked(gate: ToolGate) -> None:
    decision = gate.check("ghost", Policy())
    assert not decision.allowed
    assert "not registered" in (decision.reason or "")


def test_authorize_raises_on_blocked_tool(gate: ToolGate) -> None:
    policy = Policy(allowed_tools=set())  # empty allowlist = nothing allowed
    with pytest.raises(PolicyViolationError):
        gate.authorize("safe_read", policy)


def test_explicit_approval_rule_flags_tool(gate: ToolGate) -> None:
    policy = Policy(approval_rules=ApprovalRules(tools_requiring_approval={"safe_read"}))
    decision = gate.check("safe_read", policy)
    assert decision.allowed
    assert decision.requires_approval


def test_side_effect_tool_requires_approval_when_configured(gate: ToolGate) -> None:
    policy = Policy(approval_rules=ApprovalRules(require_approval_on_side_effects=True))
    # read-only tool: no approval; side-effecting tool: approval required
    assert not gate.check("safe_read", policy).requires_approval
    assert gate.check("danger_write", policy).requires_approval


def test_none_allowlist_permits_everything(gate: ToolGate) -> None:
    # Policy with allowed_tools=None means "all registered tools".
    assert gate.check("safe_read", Policy()).allowed
    assert gate.check("danger_write", Policy()).allowed
