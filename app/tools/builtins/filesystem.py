"""
app/tools/builtins/filesystem.py

fs_ops — read / write / list files, confined to a single workspace directory.

Every path is resolved and checked to stay inside the workspace root, so
``../`` traversal and absolute paths cannot escape. Writes mutate state, so the
tool declares ``side_effects=True`` (eligible for HITL approval via ToolGate).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.tools.base import BaseTool, ToolResult
from app.tools.decorators import audited
from app.tools.registry import tool

logger = get_logger(__name__)

_MAX_READ_BYTES = 200_000


class FileSystemArgs(BaseModel):
    op: Literal["read", "write", "list"] = Field(description="Operation to perform.")
    path: str = Field(description="Path relative to the workspace root.")
    content: str | None = Field(
        default=None, description="Content to write (required when op='write')."
    )


@tool
class FileSystemTool(BaseTool[FileSystemArgs]):
    name = "fs_ops"
    description = (
        "Read, write or list files inside the agent workspace. Paths are relative "
        "to the workspace root; escaping it is not allowed. Use for scratch files "
        "and reading local documents. Do NOT use to reach arbitrary system paths."
    )
    args_schema = FileSystemArgs
    side_effects = True  # write mutates the workspace

    def __init__(self, workspace_dir: str) -> None:
        self._root = Path(workspace_dir).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, rel: str) -> Path | None:
        """Resolve ``rel`` under the root; return None if it escapes."""
        candidate = (self._root / rel).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            return None
        return candidate

    @audited
    async def execute(self, args: FileSystemArgs) -> ToolResult:
        target = self._resolve(args.path)
        if target is None:
            return ToolResult.failure(f"Path escapes the workspace: {args.path!r}")
        return await asyncio.to_thread(self._dispatch, args, target)

    def _dispatch(self, args: FileSystemArgs, target: Path) -> ToolResult:
        if args.op == "read":
            if not target.is_file():
                return ToolResult.failure(f"Not a file: {args.path!r}")
            data = target.read_bytes()[:_MAX_READ_BYTES]
            return ToolResult.success(
                {"path": args.path, "content": data.decode("utf-8", errors="replace")}
            )
        if args.op == "list":
            base = target if target.is_dir() else self._root
            entries = sorted(
                p.name + ("/" if p.is_dir() else "") for p in base.iterdir()
            )
            return ToolResult.success({"path": args.path, "entries": entries})
        # write
        if args.content is None:
            return ToolResult.failure("content is required for op='write'")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.content, encoding="utf-8")
        return ToolResult.success({"path": args.path, "bytes_written": len(args.content)})
