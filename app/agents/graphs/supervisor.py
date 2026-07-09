"""
app/agents/graphs/supervisor.py

Supervisor graph: ``memory_recall → route → {researcher | coder | direct} → memory_write → END``.

The supervisor recalls memory, makes exactly one structured routing decision, and
hands the request to the chosen agent — the researcher and coder are embedded as
compiled subgraphs (they inherit the parent's checkpointer), while ``direct`` is a
single-shot answer node for requests no tool would help with. All three converge
on ``memory_write``, which persists the outcome before the run ends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langgraph.graph import END, START, StateGraph

from app.agents.graphs.coder import build_coder_graph
from app.agents.graphs.researcher import build_researcher_graph
from app.agents.nodes.memory_recall import make_memory_recall_node
from app.agents.nodes.memory_write import make_memory_write_node
from app.agents.nodes.route import make_route_node
from app.agents.state import AgentState
from app.agents.stream import emit as _emit

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.graph.state import CompiledStateGraph

    from app.agents.deps import GraphDeps

_DIRECT_SYSTEM = (
    "You are a helpful, concise assistant. Answer the user directly. "
    "If you genuinely do not know, say so rather than guessing."
)
_DIRECT_OPTIONS = {"temperature": 0.7}


def _make_direct_node(deps: GraphDeps) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def direct(state: AgentState) -> dict[str, Any]:
        _emit("node", node="direct")
        messages: list[dict[str, Any]] = [{"role": "system", "content": _DIRECT_SYSTEM}]
        ctx = state.memory_context
        if ctx is not None and not ctx.is_empty:
            if ctx.summary:
                messages.append({"role": "system", "content": f"Context: {ctx.summary}"})
            for item in ctx.long_term:
                messages.append({"role": "system", "content": f"Known: {item.content}"})
        messages.append({"role": "user", "content": state.query})

        response = await deps.llm.chat(messages, options=_DIRECT_OPTIONS)
        if response.content:
            _emit("token", text=response.content, node="direct")
        tokens = response.prompt_tokens + response.completion_tokens
        return {
            "messages": [{"role": "assistant", "content": response.content}],
            "final_answer": response.content,
            "budget": state.budget.with_delta(tokens=tokens, iterations=1),
        }

    return direct


def build_supervisor_graph(deps: GraphDeps) -> StateGraph:
    """Assemble (but do not compile) the supervisor graph."""
    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("memory_recall", make_memory_recall_node(deps))
    graph.add_node("route", make_route_node(deps))
    graph.add_node("researcher", build_researcher_graph(deps))
    graph.add_node("coder", build_coder_graph(deps))
    graph.add_node("direct", _make_direct_node(deps))
    graph.add_node("memory_write", make_memory_write_node(deps))

    graph.add_edge(START, "memory_recall")
    graph.add_edge("memory_recall", "route")

    def dispatch(state: AgentState) -> Literal["researcher", "coder", "direct"]:
        return state.route.agent if state.route is not None else "direct"

    graph.add_conditional_edges(
        "route",
        dispatch,
        {"researcher": "researcher", "coder": "coder", "direct": "direct"},
    )

    for agent in ("researcher", "coder", "direct"):
        graph.add_edge(agent, "memory_write")
    graph.add_edge("memory_write", END)

    return graph


def compile_supervisor(deps: GraphDeps, checkpointer: Any = None) -> CompiledStateGraph:
    """Compile the supervisor graph with an optional checkpointer."""
    return build_supervisor_graph(deps).compile(checkpointer=checkpointer)
