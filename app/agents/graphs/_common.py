"""
app/agents/graphs/_common.py

Small helpers shared by the subgraph builders: reading the latest assistant turn
out of the transcript, and a ``finalize`` node that guarantees the run ends with
a ``final_answer`` set — even when the BudgetEnforcer forced the stop mid-task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.agents.state import AgentState
    from app.services.llm.base import LLMMessage

_BUDGET_STOP_ANSWER = (
    "I wasn't able to finish this within the allotted budget. "
    "Here is what I gathered so far."
)


def last_assistant_message(messages: list[LLMMessage]) -> LLMMessage | None:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg
    return None


def last_assistant_content(messages: list[LLMMessage]) -> str:
    msg = last_assistant_message(messages)
    return str(msg["content"]) if msg and msg.get("content") else ""


def make_finalize_node() -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Ensure ``final_answer`` is populated before the subgraph reaches END."""

    async def finalize(state: AgentState) -> dict[str, Any]:
        if state.final_answer:
            return {}
        content = last_assistant_content(state.messages)
        return {"final_answer": content or _BUDGET_STOP_ANSWER}

    return finalize
