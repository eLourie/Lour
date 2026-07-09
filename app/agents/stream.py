"""
app/agents/stream.py

``emit`` — publish a fine-grained custom event from inside a graph node.

LangGraph's ``get_stream_writer()`` returns the active run's custom-event sink
(and a harmless no-op when the graph is invoked without ``stream_mode="custom"``,
e.g. in tests). Nodes emit token / tool progress through it; the SSE layer
(app/gateway/streaming.py) turns those dicts into ``AgentEvent`` frames.
"""

from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer


def emit(event: str, **data: Any) -> None:
    """Best-effort custom stream event; silently a no-op off-stream."""
    try:
        writer = get_stream_writer()
    except Exception:  # pragma: no cover - defensive; writer absent off-stream
        return
    writer({"event": event, **data})
