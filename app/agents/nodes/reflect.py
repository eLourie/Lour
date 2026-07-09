"""
app/agents/nodes/reflect.py

reflect — the researcher's self-assessment. After an ``act`` iteration it judges
(via structured output) whether the information gathered is enough to answer the
request. Its verdict drives the subgraph's loop-or-finish edge; when complete it
promotes the latest assistant answer to ``final_answer``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.agents.prompts import render
from app.agents.state import Reflection
from app.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.agents.deps import GraphDeps
    from app.agents.state import AgentState
    from app.services.llm.base import LLMMessage

logger = get_logger(__name__)


def _last_assistant_content(messages: list[LLMMessage]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return str(msg["content"])
    return ""


def make_reflect_node(
    deps: GraphDeps,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def reflect(state: AgentState) -> dict[str, Any]:
        last_content = _last_assistant_content(state.messages)
        tools_summary = ", ".join(r.name for r in state.tools_called) or "none"
        prompt = render(
            "reflection.j2",
            query=state.query,
            plan=state.plan,
            tools_summary=tools_summary,
            last_content=last_content,
        )
        verdict = await deps.structured.complete(
            [{"role": "user", "content": prompt}], schema=Reflection
        )
        logger.info(
            "node_reflect",
            is_complete=verdict.is_complete,
            session_id=state.session_id,
        )
        updates: dict[str, Any] = {"reflection": verdict}
        if verdict.is_complete and last_content:
            updates["final_answer"] = last_content
        return updates

    return reflect
