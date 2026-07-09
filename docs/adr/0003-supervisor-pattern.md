# ADR-003: LangGraph supervisor pattern

**Status:** Accepted
**Date:** 2026-07-09

## Context

The agent must handle qualitatively different requests — open-ended research,
code writing/execution, and simple direct answers — each needing a different set
of tools, prompts and control flow. We need an orchestration topology that routes
a request to the right specialist without collapsing into one giant do-everything
prompt.

## Decision

Adopt the **supervisor pattern**: a single routing agent (the supervisor) makes
exactly one structured decision and hands the request to a specialised subagent.

- **Supervisor graph:** `memory_recall → route → {researcher | coder | direct} → memory_write → END`.
- **Routing is structured**, not string-parsed: the `route` node returns a
  validated Pydantic `Route{agent, reasoning}` via the structured-output service.
  Routing happens once — the chosen subagent owns the request; there is no second
  round of supervision (PROJECT_CONTEXT §5.3).
- **Subagents are LangGraph subgraphs** embedded as nodes, so they inherit the
  parent's checkpointer:
  - *researcher*: `plan → act ↻ reflect → finalize` — gather information with
    web/RAG tools and self-assess completeness.
  - *coder*: `setup → act ↻ finalize` — write and run code in the sandbox,
    reading stdout back on each iteration.
  - *direct*: a single-shot answer for requests no tool would help.

Alternatives considered and rejected: **Swarm** (agents hand off freely — harder
to debug, prone to loops), **Hierarchical** (supervisors of supervisors — overkill
for three agents), **Flat sequential** (no routing — does not scale to distinct
capabilities).

## Consequences

- One clear place where intent is decided (the `route` node) and one place where
  each capability lives (its subgraph). Easy to add a fourth agent: a subgraph +
  a `Literal` route option.
- Structured routing removes a whole class of brittle string-parsing bugs and
  makes the decision inspectable/traceable.
- Memory recall/write bracket every run uniformly, regardless of the chosen
  agent.

## Trade-offs

- One extra LLM call per request for routing.
- Mitigation: a single resident model does the routing (no separate fast model,
  PROJECT_CONTEXT §5.3); a rule-based fast-path remains a possible showcase
  optimisation if routing latency ever matters.
