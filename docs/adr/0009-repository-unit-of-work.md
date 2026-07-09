# ADR-009: Repository pattern + Unit of Work

**Status:** Accepted  
**Date:** 2026-07-03

## Context

Direct SQLAlchemy session usage in service/domain code couples business
logic to the ORM, making unit testing require a real DB connection.

## Decision

Implement generic `Repository[T]` over SQLAlchemy 2.0 async, orchestrated
by `UnitOfWork` which owns the session and controls commit/rollback.

This is **core** (not showcase) — it provides genuine testability
(mock the repo, not the DB) and explicit transactional boundaries.

## Consequences

- Services depend on `Repository` abstractions, not `AsyncSession` directly.
- Tests mock `Repository` without a DB container.
- `UnitOfWork` as a context manager makes transaction scope explicit.

## Trade-offs

- Boilerplate for each new model (`SessionRepository`, etc.).
- Mitigation: generic `Repository[T]` keeps concrete repos to ~10 lines.