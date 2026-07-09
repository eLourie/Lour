# ADR-012: APScheduler for consolidation, arq as scale-up path

**Status:** Accepted  
**Date:** 2026-07-09

## Context

Long-term memory is populated by *consolidation*: periodically distilling salient
facts out of session activity, deduplicating them, and persisting the important
ones. This is background work that must run on a schedule — it is not triggered
by a user request. The question is what runs it.

## Decision

Use **APScheduler** (`AsyncIOScheduler`), in-process, as the consolidation
runner. A single interval job (`MEMORY_CONSOLIDATION_INTERVAL_S`) sweeps every
session with a live working window and consolidates it; the job is started in the
FastAPI lifespan and stopped on shutdown. The same `consolidate_session` method
the job calls is also invokable directly (used by tests and any manual trigger).

`arq` (a separate worker process backed by a Redis queue) is documented as the
**scale-up path**, not the default.

## Consequences

- Zero extra processes or infrastructure: consolidation lives inside the app.
- The scheduled run is resilient — a failure consolidating one session is logged
  and skipped, never taking down the sweep (`max_instances=1`, `coalesce=True`).
- Consolidation can be disabled entirely via `MEMORY_CONSOLIDATION_ENABLED=false`.

## Trade-offs

- In-process scheduling does not survive being spread across multiple app
  instances, and heavy consolidation competes with request handling for the
  event loop.
- Mitigation: for a single-user instance this is a non-issue; when it becomes
  one, `arq` moves the work to a dedicated worker without changing the
  `ConsolidationService` logic — only who calls it.
