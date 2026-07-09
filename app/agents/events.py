"""
app/agents/events.py

AgentEvent — the taxonomy of things the client sees while a run unfolds.

Two levels of granularity flow over the same SSE channel (PROJECT_CONTEXT §7,
Phase 5 "двухуровневый streaming"):

  - node level   — NODE_STARTED / NODE_FINISHED / ROUTE_DECIDED / TOOL_CALLED /
                   TOOL_RESULT: coarse progress a UI can render as a timeline.
  - token level  — TOKEN: incremental answer text for a typing effect.

Plus terminal signals: APPROVAL_REQUIRED (HITL pause), FINAL (the answer),
ERROR (a run-ending failure) and DONE (stream close).

Events are Pydantic models so they serialise deterministically; ``to_sse``
renders one as a Server-Sent Event frame.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EventType(StrEnum):
    NODE_STARTED = "node_started"
    NODE_FINISHED = "node_finished"
    ROUTE_DECIDED = "route_decided"
    TOKEN = "token"
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    APPROVAL_REQUIRED = "approval_required"
    FINAL = "final"
    ERROR = "error"
    DONE = "done"


class AgentEvent(BaseModel):
    """One streamed event. ``data`` carries the type-specific payload."""

    type: EventType
    node: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    def to_sse(self) -> str:
        """Render as an SSE frame: an ``event:`` line + a JSON ``data:`` line."""
        payload = {"node": self.node, **self.data}
        body = json.dumps(payload, default=str, ensure_ascii=False)
        return f"event: {self.type.value}\ndata: {body}\n\n"

    # Constructors — keep call sites terse and consistent.

    @classmethod
    def node_started(cls, node: str) -> AgentEvent:
        return cls(type=EventType.NODE_STARTED, node=node)

    @classmethod
    def node_finished(cls, node: str, **data: Any) -> AgentEvent:
        return cls(type=EventType.NODE_FINISHED, node=node, data=data)

    @classmethod
    def route_decided(cls, agent: str, reasoning: str = "") -> AgentEvent:
        return cls(
            type=EventType.ROUTE_DECIDED,
            node="route",
            data={"agent": agent, "reasoning": reasoning},
        )

    @classmethod
    def token(cls, text: str, *, node: str | None = None) -> AgentEvent:
        return cls(type=EventType.TOKEN, node=node, data={"text": text})

    @classmethod
    def tool_called(
        cls, name: str, arguments: dict[str, Any], *, node: str | None = None
    ) -> AgentEvent:
        return cls(
            type=EventType.TOOL_CALLED,
            node=node,
            data={"name": name, "arguments": arguments},
        )

    @classmethod
    def tool_result(
        cls, name: str, *, ok: bool, error: str | None = None, node: str | None = None
    ) -> AgentEvent:
        return cls(
            type=EventType.TOOL_RESULT,
            node=node,
            data={"name": name, "ok": ok, "error": error},
        )

    @classmethod
    def approval_required(cls, tool: str, arguments: dict[str, Any], reason: str) -> AgentEvent:
        return cls(
            type=EventType.APPROVAL_REQUIRED,
            data={"tool": tool, "arguments": arguments, "reason": reason},
        )

    @classmethod
    def final(cls, answer: str, **data: Any) -> AgentEvent:
        return cls(type=EventType.FINAL, data={"answer": answer, **data})

    @classmethod
    def error(cls, message: str, *, code: str = "agent_error") -> AgentEvent:
        return cls(type=EventType.ERROR, data={"message": message, "code": code})

    @classmethod
    def done(cls) -> AgentEvent:
        return cls(type=EventType.DONE)
