"""
app/tools/builtins/code_exec.py

code_exec — run a short Python or JavaScript snippet in the isolated sandbox and
return its stdout/stderr. Dispatches to the SandboxService (Docker); the tool
itself holds no execution logic. Declares ``side_effects=True`` — running
arbitrary code is inherently side-effecting and approval-eligible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from app.services.sandbox.base import SandboxLanguage
from app.tools.base import BaseTool, ToolResult
from app.tools.decorators import audited
from app.tools.registry import tool

if TYPE_CHECKING:
    from app.services.sandbox.base import SandboxService

_LANGUAGE_MAP = {
    "python": SandboxLanguage.PYTHON,
    "javascript": SandboxLanguage.JAVASCRIPT,
}


class CodeExecArgs(BaseModel):
    code: str = Field(description="Source code to execute. Print results to stdout.")
    language: Literal["python", "javascript"] = Field(default="python")
    timeout_s: int | None = Field(
        default=None, ge=1, le=120, description="Optional wall-clock timeout override."
    )


@tool
class CodeExecTool(BaseTool[CodeExecArgs]):
    name = "code_exec"
    description = (
        "Execute a short Python or JavaScript snippet in a network-isolated "
        "sandbox and return stdout/stderr. Use for calculations, data wrangling "
        "or quick scripts. Do NOT expect network access or filesystem persistence."
    )
    args_schema = CodeExecArgs
    side_effects = True  # runs arbitrary code

    def __init__(self, sandbox: SandboxService) -> None:
        self._sandbox = sandbox

    @audited
    async def execute(self, args: CodeExecArgs) -> ToolResult:
        result = await self._sandbox.run(
            args.code,
            language=_LANGUAGE_MAP[args.language],
            timeout_s=args.timeout_s,
        )
        payload = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
        }
        if result.ok:
            return ToolResult.success(payload, duration_ms=result.duration_ms)
        reason = "timed out" if result.timed_out else f"exited with code {result.exit_code}"
        return ToolResult(ok=False, data=payload, error=f"code_exec {reason}")
