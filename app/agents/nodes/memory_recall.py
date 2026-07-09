"""
app/agents/nodes/memory_recall.py

memory_recall — the supervisor graph's first node. It asks the MemoryManager for
everything relevant to the incoming query (short-term working window + rolling
summary + top long-term facts) and parks it on ``state.memory_context`` so the
downstream nodes (route / plan / act) can weave it into their prompts.

Memory is a service, not baked into the agent (ADR-005): this node only *reads*
through the facade; it writes nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.agents.deps import GraphDeps
    from app.agents.state import AgentState

logger = get_logger(__name__)


def make_memory_recall_node(
    deps: GraphDeps,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def memory_recall(state: AgentState) -> dict[str, Any]:
        context = await deps.memory.recall(state.session_id, state.query)
        logger.debug(
            "node_memory_recall",
            session_id=state.session_id,
            long_term=len(context.long_term),
            has_summary=context.summary is not None,
        )
        return {"memory_context": context}

    return memory_recall
