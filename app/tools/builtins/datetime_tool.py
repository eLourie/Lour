"""
app/tools/builtins/datetime_tool.py

get_datetime — return the current timezone-aware timestamp.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from app.tools.base import BaseTool, ToolResult
from app.tools.registry import tool


class DateTimeArgs(BaseModel):
    timezone: str = Field(
        default="UTC",
        description="IANA timezone name, e.g. 'UTC', 'Europe/Berlin'. Defaults to UTC.",
    )


@tool
class DateTimeTool(BaseTool[DateTimeArgs]):
    name = "get_datetime"
    description = (
        "Get the current date and time as a timezone-aware ISO-8601 string. "
        "Use when the user asks what time or date it is now. "
        "Do NOT use for parsing or formatting arbitrary dates."
    )
    args_schema = DateTimeArgs

    async def execute(self, args: DateTimeArgs) -> ToolResult:
        try:
            tz = UTC if args.timezone.upper() == "UTC" else ZoneInfo(args.timezone)
        except (ZoneInfoNotFoundError, ValueError):
            return ToolResult.failure(f"Unknown timezone: {args.timezone!r}")
        now = datetime.now(tz)
        return ToolResult.success(
            {"iso": now.isoformat(), "timezone": args.timezone, "epoch": now.timestamp()}
        )
