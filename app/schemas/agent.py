"""
app/schemas/agent.py

API DTOs for the agent routes (/v1/chat, /v1/sessions).
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — runtime type needed by Pydantic
from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """A free-form turn. Omit thread_id to start a new session."""

    message: str = Field(min_length=1, description="The user's message.")
    thread_id: str | None = Field(
        default=None, description="Existing session/checkpoint thread to continue."
    )


class SessionSummary(BaseModel):
    thread_id: str
    skill_name: str | None = None
    created_at: datetime
    updated_at: datetime


class SessionListResponse(BaseModel):
    items: list[SessionSummary]
    total: int


class SessionDetail(BaseModel):
    thread_id: str
    status: str = Field(description="running | awaiting_approval | done | unknown")
    final_answer: str | None = None
    pending_approval: dict[str, Any] | None = None
    next_nodes: list[str] = Field(default_factory=list)


class ApproveRequest(BaseModel):
    approved: bool = Field(default=True, description="Approve or deny the paused tool call.")
