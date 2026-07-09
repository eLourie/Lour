"""
app/agents/nodes/memory_write.py

memory_write — the supervisor graph's exit node. It persists the turn's outcome
through the MemoryManager facade: the user's query and the agent's final answer
land in short-term + episodic memory. Long-term *distillation* is not done here —
that is consolidation's asynchronous job (ADR-012), keeping this hot path cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.agents.deps import GraphDeps
    from app.agents.state import AgentState

logger = get_logger(__name__)


def make_memory_write_node(
    deps: GraphDeps,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def memory_write(state: AgentState) -> dict[str, Any]:
        if state.query:
            await deps.memory.write(state.session_id, "user", state.query)
        if state.final_answer:
            await deps.memory.write(state.session_id, "assistant", state.final_answer)
        logger.debug(
            "node_memory_write",
            session_id=state.session_id,
            wrote_answer=bool(state.final_answer),
        )
        return {"finished": True}

    return memory_write
