# ADR-008: PostgresSaver for checkpointing

**Status:** Accepted
**Date:** 2026-07-09

## Context

A multi-step agent run is long-lived: it may span many LLM/tool calls, pause for
human approval (HITL), or be interrupted by a process restart. To survive any of
that, the graph's state must be durably persisted after every step — not held only
in memory.

## Decision

Use LangGraph's **`AsyncPostgresSaver`** as the checkpointer, backed by a
dedicated `agent_checkpoint` database (provisioned in the Postgres init script,
Phase 0). Postgres is already in the stack, so this adds no new backing service.

- A `CheckpointerManager` (app/agents/checkpointing.py) owns the saver's
  connection-pool lifecycle across the app lifespan — `start()` opens the pool and
  runs `setup()` (idempotent `CREATE TABLE IF NOT EXISTS`), `aclose()` closes it.
- The compiled supervisor graph is given the saver; its embedded subgraphs inherit
  it, so the whole run checkpoints under one thread.
- The `thread_id` (also the Session row's key) is the resume handle: a killed run
  is re-driven from its last checkpoint, and a HITL-paused run resumes via
  `Command(resume=...)` from the `/v1/sessions/{id}/approve` endpoint.
- `delete_thread(thread_id)` gives retention/privacy a concrete hook.

Rejected alternatives: **in-memory / SQLite** savers — no durability across
restarts, no shared state for HITL resume from a separate request.

## Consequences

- Runs resume after a restart, can be replayed for debugging, and support
  time-travel over their step history.
- HITL is a first-class flow: pause on a checkpoint, resume from it later.
- Checkpoints are isolated in their own database, never colliding with app tables.

## Trade-offs

- A write to Postgres on every node step.
- Mitigation: writes are async and off the critical path; the checkpoint DB is
  separate so its load does not contend with application queries.
