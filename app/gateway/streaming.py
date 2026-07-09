"""
app/gateway/streaming.py

SSE helpers: turn a LangGraph run into a stream of ``AgentEvent`` frames.

The run is streamed in two modes at once (``subgraphs=True`` so the researcher /
coder inner nodes surface too):

  - ``updates`` — one entry per finished node; the supervisor's ``route`` update
    also yields a ROUTE_DECIDED, and a ``__interrupt__`` entry yields
    APPROVAL_REQUIRED (a HITL pause).
  - ``custom``  — the fine-grained events nodes emit via ``app.agents.stream``:
    NODE_STARTED, TOKEN, TOOL_CALLED, TOOL_RESULT.

When the stream ends without a pause, the final state's ``final_answer`` is sent
as FINAL. Every stream terminates with DONE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.agents.events import AgentEvent
from app.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger(__name__)


def _custom_event(chunk: dict[str, Any]) -> AgentEvent | None:
    """Map a node's custom-stream dict to an AgentEvent."""
    kind = chunk.get("event")
    node = chunk.get("node")
    if kind == "node":
        return AgentEvent.node_started(node or "")
    if kind == "token":
        return AgentEvent.token(chunk.get("text", ""), node=node)
    if kind == "tool_called":
        return AgentEvent.tool_called(
            str(chunk.get("name", "")), chunk.get("arguments", {}) or {}, node=node
        )
    if kind == "tool_result":
        return AgentEvent.tool_result(
            str(chunk.get("name", "")),
            ok=bool(chunk.get("ok", True)),
            error=chunk.get("error"),
            node=node,
        )
    return None


def _update_events(chunk: dict[str, Any]) -> list[AgentEvent]:
    """Map an ``updates`` chunk ({node: update}) to node-level AgentEvents."""
    events: list[AgentEvent] = []
    for node, update in chunk.items():
        if node == "__interrupt__":
            continue
        if isinstance(update, dict):
            route = update.get("route")
            if route is not None:
                agent = getattr(route, "agent", None) or route.get("agent", "")
                reasoning = getattr(route, "reasoning", "") or ""
                events.append(AgentEvent.route_decided(str(agent), str(reasoning)))
        events.append(AgentEvent.node_finished(str(node)))
    return events


def _interrupt_event(interrupts: Any) -> AgentEvent:
    """Build APPROVAL_REQUIRED from a ``__interrupt__`` update payload."""
    try:
        value = getattr(interrupts[0], "value", {}) or {}
        pending = value.get("pending") or [{}]
        first = pending[0]
    except (IndexError, AttributeError, TypeError):
        first = {}
    return AgentEvent.approval_required(
        tool=str(first.get("tool", "")),
        arguments=first.get("arguments", {}) or {},
        reason=str(first.get("reason", "Approval required")),
    )


async def stream_run(graph: Any, payload: Any, config: dict[str, Any]) -> AsyncIterator[str]:
    """
    Drive ``graph`` and yield SSE frames.

    ``payload`` is the seed state dict for a new run, or a ``langgraph.types``
    ``Command`` to resume a paused (HITL) run.
    """
    interrupted = False
    try:
        async for item in graph.astream(
            payload, config=config, stream_mode=["updates", "custom"], subgraphs=True
        ):
            mode, chunk = (item[1], item[2]) if len(item) == 3 else item
            if mode == "custom":
                event = _custom_event(chunk)
                if event is not None:
                    yield event.to_sse()
            elif mode == "updates":
                for event in _update_events(chunk):
                    yield event.to_sse()
                if "__interrupt__" in chunk:
                    interrupted = True
                    yield _interrupt_event(chunk["__interrupt__"]).to_sse()
    except Exception as exc:  # a run-ending failure — report and close cleanly
        logger.exception("agent_stream_error")
        yield AgentEvent.error(str(exc)).to_sse()
        yield AgentEvent.done().to_sse()
        return

    if not interrupted:
        snapshot = await graph.aget_state(config)
        if snapshot.next:  # paused for some other reason — treat as not-final
            interrupted = True
        else:
            final = (snapshot.values or {}).get("final_answer")
            if final:
                yield AgentEvent.final(str(final)).to_sse()

    yield AgentEvent.done().to_sse()
