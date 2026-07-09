"""
app/agents/enforcement/budget.py

BudgetEnforcer — the Orchestration-layer enforcement point of the unified Policy
(ADR-011). It answers one question for the graph: *may this run take another
step?* The ToolGate (Tools layer) enforces the other half — allowlist + approval.

Two failure modes are guarded:

  - budget   — the run has spent its iteration / tool-call / token / wall-time
               allowance (limits come from the resolved ``policy.budget``).
  - loop     — the agent keeps issuing the *same* tool call, making no progress.
               A 14B model on a long tool chain can get stuck this way (ADR-007);
               detecting it lets the graph force finalisation instead of burning
               the whole budget.

The enforcer is pure and synchronous: it reads ``AgentState`` and returns a
verdict. Acting on the verdict (routing to a finish node) is the graph's job.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.agents.state import AgentState, ToolCallRecord

logger = get_logger(__name__)

# How many identical trailing tool calls constitute a loop.
_DEFAULT_LOOP_WINDOW = 3


class BudgetVerdict(BaseModel):
    """Whether the run may continue, and why not if it may not."""

    can_continue: bool
    kind: Literal["ok", "budget", "loop"] = "ok"
    reason: str | None = None


class BudgetEnforcer:
    """Enforces budget limits and detects non-progressing loops."""

    def __init__(self, loop_window: int = _DEFAULT_LOOP_WINDOW) -> None:
        self._loop_window = loop_window

    def check(self, state: AgentState) -> BudgetVerdict:
        """Return whether the graph may take another step under ``state``."""
        b = state.budget
        limits = state.policy.budget

        if limits.max_iterations is not None and b.iterations >= limits.max_iterations:
            return self._stop("budget", f"max_iterations reached ({limits.max_iterations})")
        if limits.max_tool_calls is not None and b.tool_calls_made >= limits.max_tool_calls:
            return self._stop("budget", f"max_tool_calls reached ({limits.max_tool_calls})")
        if limits.max_cost_tokens is not None and b.tokens_used >= limits.max_cost_tokens:
            return self._stop("budget", f"max_cost_tokens reached ({limits.max_cost_tokens})")
        if limits.max_duration_s is not None and b.elapsed_s() >= limits.max_duration_s:
            return self._stop("budget", f"max_duration_s reached ({limits.max_duration_s}s)")

        if self.detect_loop(state.tools_called):
            return self._stop(
                "loop", f"same tool call repeated {self._loop_window}x without progress"
            )

        return BudgetVerdict(can_continue=True)

    def detect_loop(self, tools_called: list[ToolCallRecord]) -> bool:
        """True when the last ``loop_window`` tool calls are byte-identical."""
        if len(tools_called) < self._loop_window:
            return False
        tail = tools_called[-self._loop_window :]
        signatures = {self._signature(r.name, r.arguments) for r in tail}
        return len(signatures) == 1

    @staticmethod
    def _signature(name: str, arguments: dict[str, object]) -> str:
        return name + "::" + json.dumps(arguments, sort_keys=True, default=str)

    def _stop(self, kind: Literal["budget", "loop"], reason: str) -> BudgetVerdict:
        logger.info("budget_enforced", kind=kind, reason=reason)
        return BudgetVerdict(can_continue=False, kind=kind, reason=reason)
