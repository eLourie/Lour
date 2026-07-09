"""
app/agents/state.py

AgentState — the single Pydantic model that flows through every node of the
supervisor graph (and its subgraphs), plus the small structured-decision models
the nodes read and write.

Design notes:
  - LangGraph merges each node's partial return into the state. Fields that
    *accumulate* across nodes (the message transcript, the tool-call ledger)
    carry an ``Annotated[..., _extend]`` reducer so returns are appended, not
    replaced. Every other field is last-writer-wins, which is correct because
    the graph runs its nodes sequentially (no parallel writers to a field).
  - ``budget`` is a serialisable counter object (epoch ``started_at``, not a
    monotonic clock) so wall-time enforcement keeps working after a run is
    resumed from a Postgres checkpoint.
  - ``policy`` is resolved once at the graph entry and stored on the state so
    the BudgetEnforcer (limits) and the act node (tool allowlist/approval) read
    a single source of truth — and so the resolved policy survives a resume.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from app.core.policy import Policy
from app.services.llm.base import LLMMessage  # noqa: TC001 — Pydantic runtime field
from app.services.memory.base import MemoryContext  # noqa: TC001 — Pydantic runtime field

# Reducers


def _extend[T](left: list[T] | None, right: list[T] | None) -> list[T]:
    """Append reducer: concatenate the accumulated list with a node's return."""
    return [*(left or []), *(right or [])]


# Structured decisions (produced by nodes via structured output)


AgentName = Literal["researcher", "coder", "direct"]


class Route(BaseModel):
    """The supervisor's routing decision (structured, never string-parsed)."""

    agent: AgentName = Field(description="Which agent should handle the request.")
    reasoning: str = Field(default="", description="Why this agent was chosen.")


class Plan(BaseModel):
    """A researcher's multi-step plan."""

    steps: list[str] = Field(default_factory=list, description="Ordered plan steps.")


class Reflection(BaseModel):
    """A researcher's self-assessment of whether the task is complete."""

    is_complete: bool = Field(description="True if the gathered information answers the query.")
    reasoning: str = Field(default="", description="Justification for the completion verdict.")
    missing: str | None = Field(
        default=None, description="What is still missing, if not yet complete."
    )


class ToolCallRecord(BaseModel):
    """One executed tool call, appended to the ledger for tracing + loop detection."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error: str | None = None


class PendingApproval(BaseModel):
    """Surfaced when a side-effecting tool call is paused for HITL approval."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = "Tool requires human approval before execution."


# Budget counters


class BudgetState(BaseModel):
    """Mutable run counters, checkpoint-serialisable (epoch clock)."""

    tokens_used: int = 0
    tool_calls_made: int = 0
    iterations: int = 0
    started_at: float = Field(default_factory=lambda: time.time())

    def with_delta(
        self, *, tokens: int = 0, tool_calls: int = 0, iterations: int = 0
    ) -> BudgetState:
        """Return a copy advanced by the given deltas (nodes never mutate in place)."""
        return self.model_copy(
            update={
                "tokens_used": self.tokens_used + tokens,
                "tool_calls_made": self.tool_calls_made + tool_calls,
                "iterations": self.iterations + iterations,
            }
        )

    def elapsed_s(self) -> float:
        return max(0.0, time.time() - self.started_at)


# The graph state


class AgentState(BaseModel):
    """State threaded through the supervisor graph and every subgraph."""

    # Identity
    session_id: str
    thread_id: str
    query: str = ""

    # Conversation the LLM sees (Ollama/OpenAI message dicts). Accumulates.
    messages: Annotated[list[LLMMessage], _extend] = Field(default_factory=list)

    # Routing + planning + reflection (last-writer-wins)
    route: Route | None = None
    plan: list[str] = Field(default_factory=list)
    reflection: Reflection | None = None

    # Tool ledger — accumulates across act iterations.
    tools_called: Annotated[list[ToolCallRecord], _extend] = Field(default_factory=list)

    # Injected memory (short-term window/summary + long-term facts).
    memory_context: MemoryContext | None = None

    # Enforcement inputs/outputs
    policy: Policy = Field(default_factory=Policy)
    budget: BudgetState = Field(default_factory=BudgetState)

    # Terminal outputs
    final_answer: str | None = None
    finished: bool = False
    pending_approval: PendingApproval | None = None

    model_config = {"arbitrary_types_allowed": True}
