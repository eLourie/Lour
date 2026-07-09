"""
app/services/sandbox/base.py

SandboxService Protocol + shared result model.

The sandbox isolates untrusted code (LLM-generated, run by ``code_exec``). The
Protocol keeps the tool layer decoupled from the concrete backend — the Docker
implementation can be swapped for e2b/Firecracker without touching callers.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class SandboxLanguage(StrEnum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"


class SandboxResult(BaseModel):
    """Outcome of one sandboxed execution."""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    duration_ms: float = 0.0


@runtime_checkable
class SandboxService(Protocol):
    """Contract for isolated code execution backends."""

    async def run(
        self,
        code: str,
        *,
        language: SandboxLanguage = SandboxLanguage.PYTHON,
        timeout_s: int | None = None,
    ) -> SandboxResult:
        """Execute ``code`` in isolation and return its captured result."""
        ...
