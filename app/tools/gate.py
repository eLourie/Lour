"""
app/tools/gate.py

ToolGate — the Tools-layer enforcement point for the unified Policy (ADR-011,
PROJECT_CONTEXT §5.4).

Policy is *declared* once (config ← skill ← agent ← request, resolved by
PolicyResolver) and *enforced* in two places:
  * BudgetEnforcer (Orchestration) — iterations / tokens / wall-time.
  * ToolGate      (Tools)          — allowlist + HITL approval, HERE.

The gate is deliberately pure and synchronous where it can be: it decides
whether a call may proceed and whether it needs human approval. The actual
HITL interrupt (pausing the graph) lives in the Orchestration layer (Phase 5);
the gate only tells it *that* approval is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from app.core.exceptions import PolicyViolationError
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.core.policy import Policy
    from app.tools.registry import ToolRegistry

logger = get_logger(__name__)


class GateDecision(BaseModel):
    """Outcome of a ToolGate check for one tool call."""

    tool: str
    allowed: bool
    requires_approval: bool = False
    reason: str | None = None


class ToolGate:
    """Enforces the allowlist and approval rules of the active Policy."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def check(self, tool_name: str, policy: Policy) -> GateDecision:
        """
        Decide whether ``tool_name`` may run under ``policy``.

        Blocks (allowed=False) when the tool is unknown or not on the policy's
        allowlist. Flags ``requires_approval`` when the tool is named in the
        approval rules, or when it has side effects and the policy requires
        approval on side effects.
        """
        if tool_name not in self._registry:
            return GateDecision(
                tool=tool_name, allowed=False, reason=f"Tool {tool_name!r} is not registered"
            )

        if not policy.is_tool_allowed(tool_name):
            return GateDecision(
                tool=tool_name,
                allowed=False,
                reason=f"Tool {tool_name!r} is not permitted by the active policy",
            )

        tool = self._registry.get(tool_name)
        rules = policy.approval_rules
        requires_approval = tool_name in rules.tools_requiring_approval or (
            rules.require_approval_on_side_effects and tool.side_effects
        )
        return GateDecision(tool=tool_name, allowed=True, requires_approval=requires_approval)

    def authorize(self, tool_name: str, policy: Policy) -> GateDecision:
        """
        Like ``check`` but raise PolicyViolationError when the call is blocked.

        Use this at the tool-call boundary when a denied call is an error
        (rather than a routing signal). The returned decision still carries
        ``requires_approval`` for the caller to act on.
        """
        decision = self.check(tool_name, policy)
        if not decision.allowed:
            logger.warning("tool_gate_blocked", tool=tool_name, reason=decision.reason)
            raise PolicyViolationError(
                decision.reason or f"Tool {tool_name!r} blocked by policy",
                detail={"tool": tool_name},
            )
        return decision
