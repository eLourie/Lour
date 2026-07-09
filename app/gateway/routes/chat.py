"""
app/gateway/routes/chat.py

/v1/chat — the free-form conversational entry point.

A POST streams the run back as Server-Sent Events (node progress + tokens +
tool calls, see app/gateway/streaming.py). The ``thread_id`` is the LangGraph
checkpoint key: omit it to start a new session (a Session row is created and the
generated id returned in the ``X-Thread-Id`` header), or pass an existing one to
continue — the graph resumes from its Postgres checkpoint (ADR-008).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from app.agents.graphs.builder import initial_state
from app.core.di import get_state
from app.gateway.middleware.rate_limit import chat_limit, limiter
from app.gateway.streaming import stream_run
from app.infra.db.models.session import Session
from app.infra.db.unit_of_work import UnitOfWork
from app.schemas.agent import ChatRequest

if TYPE_CHECKING:
    from app.infra.clients.postgres import PostgresClient

router = APIRouter(prefix="/chat", tags=["chat"])

PostgresDep = Annotated["PostgresClient", Depends(get_state("postgres"))]


async def _ensure_session(postgres: PostgresClient, thread_id: str) -> None:
    async with UnitOfWork(postgres) as uow:
        if await uow.sessions.get_by_thread_id(thread_id) is None:
            await uow.sessions.add(Session(thread_id=thread_id))


@router.post("")
@limiter.limit(chat_limit)
async def chat(
    req: ChatRequest,
    request: Request,
    response: Response,  # slowapi injects rate-limit headers here (headers_enabled)
    postgres: PostgresDep,
) -> StreamingResponse:
    """Start (or continue) a chat turn and stream the agent's progress as SSE."""
    graph: Any = request.app.state.agent_graph

    thread_id = req.thread_id or uuid.uuid4().hex
    await _ensure_session(postgres, thread_id)

    seed = initial_state(session_id=thread_id, thread_id=thread_id, query=req.message)
    config = {"configurable": {"thread_id": thread_id}}

    return StreamingResponse(
        stream_run(graph, seed, config),
        media_type="text/event-stream",
        headers={"X-Thread-Id": thread_id, "Cache-Control": "no-cache"},
    )
