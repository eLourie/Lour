# ADR-005: Custom memory layer instead of Mem0 / Letta

**Status:** Accepted  
**Date:** 2026-07-09

## Context

The agent needs memory that survives beyond a single request: the current
session's working context, durable cross-session facts, and a chronological
record of what happened. Turnkey libraries (Mem0, Letta/MemGPT) offer this as a
black box. The project's backing stack — Redis, Qdrant, PostgreSQL — already
provides every primitive such a layer needs.

## Decision

Implement memory ourselves as three explicit layers behind a `MemoryManager`
facade (memory is a **service**, not something baked into the agent):

- **Short-term** (Redis) — per-session sliding window of verbatim turns, with
  LLM tail-summarisation when the window overflows so context is bounded but
  never silently lost.
- **Long-term** (Qdrant) — semantic facts ranked by a blended score
  `alpha·cosine + beta·recency + gamma·importance`, not cosine alone. Recency
  decays exponentially by a configurable half-life; importance is assigned by an
  LLM-as-judge at write time.
- **Episodic** (PostgreSQL) — an append-only chronological ledger, written
  through the Repository + Unit of Work stack (ADR-009).

Distillation of raw session activity into long-term facts is asynchronous and
owned by consolidation (ADR-012), keeping the write hot-path cheap.

## Consequences

- Importance scoring, recency decay and consolidation are visible, tunable code
  (via `MEMORY_*` config) — the portfolio value is in the mechanism, not a
  dependency.
- Each layer has a single responsibility and a small surface (< ~300 LOC),
  swappable independently.
- Single-user instance (variant A, §1.3): no `user_id` isolation — short-term is
  keyed by session, long-term/episodic are instance-global.

## Trade-offs

- More code than adopting a library.
- Mitigation: the shared Redis/Qdrant/PG clients and Repository/UoW already
  exist, so each layer is thin; the facade keeps callers decoupled from the
  three stores.
