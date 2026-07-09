"""
app/agents/graphs/coder.py

Coder subgraph: ``setup → act ↻ finish``.

``setup`` seeds the transcript with the coder system prompt (which steers the
model toward the ``code_exec`` sandbox tool) and the user's request. ``act`` then
runs reason-act iterations: the model writes code, runs it, reads stdout/stderr,
and fixes it. The loop continues while the model keeps calling tools and the
budget allows; it finishes when the model answers without a tool call, the budget
is exhausted, or a loop is detected. ``finalize`` guarantees a ``final_answer``.

Simpler than the researcher: no explicit plan/reflect — a coder's feedback loop
is the sandbox's stdout, read back on the next iteration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langgraph.graph import END, START, StateGraph

from app.agents.graphs._common import last_assistant_message, make_finalize_node
from app.agents.nodes.act import make_act_node
from app.agents.prompts import render
from app.agents.state import AgentState

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.graph.state import CompiledStateGraph

    from app.agents.deps import GraphDeps


def _make_setup_node(deps: GraphDeps) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def setup(state: AgentState) -> dict[str, Any]:
        system = render("coder.j2", memory_context=state.memory_context)
        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": state.query},
            ]
        }

    return setup


def build_coder_graph(deps: GraphDeps) -> CompiledStateGraph:
    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("setup", _make_setup_node(deps))
    graph.add_node("act", make_act_node(deps))
    graph.add_node("finalize", make_finalize_node())

    graph.add_edge(START, "setup")
    graph.add_edge("setup", "act")

    def loop_or_finish(state: AgentState) -> Literal["act", "finalize"]:
        if not deps.enforcer.check(state).can_continue:
            return "finalize"
        # If the model's last turn requested tools, we appended their results —
        # loop so it can read them. If it answered outright, we're done.
        last = last_assistant_message(state.messages)
        if last is not None and last.get("tool_calls"):
            return "act"
        return "finalize"

    graph.add_conditional_edges(
        "act", loop_or_finish, {"act": "act", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)

    return graph.compile()
