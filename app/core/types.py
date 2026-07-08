"""
app/core/types.py

Shared type aliases, sentinel values, and lightweight base models.
Imported by virtually every module — keep it minimal and free of app-level imports.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

# Type aliases

JSON = dict[str, Any]
"""JSON-compatible dictionary."""

RequestID = str
"""Hex UUID string identifying a single HTTP request."""

SessionID = str
"""Identifier for an agent conversation session."""

TraceID = str
"""Distributed trace identifier (e.g. Langfuse trace ID)."""

DocumentID = str
"""Opaque document identifier in the RAG pipeline."""

ChunkID = str
"""Identifier for a single RAG chunk."""

ToolName = str
"""Registered tool name — must match ToolRegistry key."""

SkillName = str
"""Registered skill name — must match SkillRegistry key."""

T = TypeVar("T")
"""Generic type variable."""

ModelT = TypeVar("ModelT", bound=BaseModel)
"""Pydantic model type variable."""


# Base Pydantic model


class AppBaseModel(BaseModel):
    """
    Project-wide Pydantic base model.
    Enforces:
      - No extra fields (strict contract between callers)
      - Assignment validation enabled
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
        use_enum_values=True,
    )


def _utcnow() -> datetime:
    return datetime.utcnow()


class TimestampedModel(AppBaseModel):
    """Adds created_at / updated_at tracking to Pydantic models."""

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


# Result wrapper


class Result(AppBaseModel):
    """
    Generic success/failure wrapper.
    Prefer typed return values for domain objects; use this for
    generic operations where the caller only needs ok/error.
    """

    ok: bool
    error: str | None = None
    detail: Any = None

    @classmethod
    def success(cls, detail: Any = None) -> Result:
        return cls(ok=True, detail=detail)

    @classmethod
    def failure(cls, error: str, detail: Any = None) -> Result:
        return cls(ok=False, error=error, detail=detail)


# Pagination


class PageParams(AppBaseModel):
    """Common pagination parameters."""

    limit: int = Field(default=20, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class Page(AppBaseModel):
    """Generic paginated response envelope."""

    items: list[Any]
    total: int
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        return self.offset + self.limit < self.total


# Utilities


def new_uuid() -> str:
    """Generate a new UUID4 as a hex string (no dashes)."""
    return uuid4().hex


def new_uuid_str() -> str:
    """Generate a new UUID4 as a standard dash-separated string."""
    return str(uuid4())
