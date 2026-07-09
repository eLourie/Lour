"""
app/gateway/routes/sessions.py

/v1/sessions — inspect sessions and drive Human-In-The-Loop approvals.

  - GET  /v1/sessions               — list known sessions (newest first).
  - GET  /v1/sessions/{thread_id}   — the live checkpoint state: status,
                                      final answer, and any pending approval.
  - POST /v1/sessions/{thread_id}/approve — resume a run paused on a tool call,
                                      streaming the continuation as SSE.

The approve endpoint is the resume half of the act node's HITL interrupt: it
feeds the human's decision back with ``Command(resume=...)`` and the graph picks
up exactly where it paused, from its Postgres checkpoint (ADR-008).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from app.core.di import get_state
from app.core.exceptions import NotFoundError
from app.gateway.streaming import stream_run
from app.infra.db.unit_of_work import UnitOfWork
from app.schemas.agent import (
    ApproveRequest,
    SessionDetail,
    SessionListResponse,
    SessionSummary,
)

if TYPE_CHECKING:
    from app.infra.clients.postgres import PostgresClient

router = APIRouter(prefix="/sessions", tags=["sessions"])

PostgresDep = Annotated["PostgresClient", Depends(get_state("postgres"))]


@router.get("", response_model=SessionListResponse)
async def list_sessions(postgres: PostgresDep) -> SessionListResponse:
    async with UnitOfWork(postgres) as uow:
        rows = await uow.sessions.list_all()
    rows.sort(key=lambda s: s.created_at, reverse=True)
    items = [
        SessionSummary(
            thread_id=s.thread_id,
            skill_name=s.skill_name,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in rows
    ]
    return SessionListResponse(items=items, total=len(items))


def _pending_approval(snapshot: Any) -> dict[str, Any] | None:
    """Extract the first pending HITL approval from a checkpoint snapshot."""
    for task in getattr(snapshot, "tasks", ()) or ():
        interrupts = getattr(task, "interrupts", None)
        if interrupts:
            value = getattr(interrupts[0], "value", {}) or {}
            pending = value.get("pending") or []
            if pending:
                return dict(pending[0])
    return None


@router.get("/{thread_id}", response_model=SessionDetail)
async def get_session(thread_id: str, request: Request) -> SessionDetail:
    graph: Any = request.app.state.agent_graph
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)

    if not snapshot.created_at and not snapshot.values:
        raise NotFoundError(f"No session found for thread {thread_id!r}", code="session_not_found")

    values = snapshot.values or {}
    pending = _pending_approval(snapshot)
    if pending is not None:
        status = "awaiting_approval"
    elif snapshot.next:
        status = "running"
    else:
        status = "done"

    return SessionDetail(
        thread_id=thread_id,
        status=status,
        final_answer=values.get("final_answer"),
        pending_approval=pending,
        next_nodes=list(snapshot.next),
    )


@router.post("/{thread_id}/approve")
async def approve(
    thread_id: str, req: ApproveRequest, request: Request
) -> StreamingResponse:
    """Resume a HITL-paused run with the user's approve/deny decision."""
    graph: Any = request.app.state.agent_graph
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = await graph.aget_state(config)
    if not snapshot.next:
        raise NotFoundError(
            f"Session {thread_id!r} is not awaiting approval", code="not_awaiting_approval"
        )

    command: Any = Command(resume={"approved": req.approved})
    return StreamingResponse(
        stream_run(graph, command, config),
        media_type="text/event-stream",
        headers={"X-Thread-Id": thread_id, "Cache-Control": "no-cache"},
    )
