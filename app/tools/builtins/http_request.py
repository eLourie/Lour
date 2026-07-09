"""
app/tools/builtins/http_request.py

http_request — make an HTTP request to a host on the configured allowlist.

The allowlist is the security boundary: an empty allowlist means the tool is
disabled (fails closed). Because a request may mutate remote state (POST/PUT/
DELETE), the tool declares ``side_effects=True``.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from app.core.exceptions import ExternalServiceError
from app.tools.base import BaseTool, ToolResult
from app.tools.decorators import audited
from app.tools.registry import tool

_TIMEOUT_S = 20.0
_MAX_BODY_CHARS = 20_000


class HttpRequestArgs(BaseModel):
    url: str = Field(description="Absolute http(s) URL. Host must be on the allowlist.")
    method: Literal["GET", "POST", "PUT", "DELETE"] = Field(default="GET")
    headers: dict[str, str] | None = Field(default=None)
    json_body: dict[str, object] | None = Field(
        default=None, description="JSON body for POST/PUT."
    )


@tool
class HttpRequestTool(BaseTool[HttpRequestArgs]):
    name = "http_request"
    description = (
        "Make an HTTP request to an allowlisted host and return status, headers "
        "and body. Use to call known JSON APIs. Do NOT use to fetch readable web "
        "pages (use web_fetch) or to reach hosts outside the allowlist."
    )
    args_schema = HttpRequestArgs
    side_effects = True  # may POST/PUT/DELETE to remote services

    def __init__(self, allowlist: list[str]) -> None:
        self._allowlist = {h.lower() for h in allowlist}

    @audited
    async def execute(self, args: HttpRequestArgs) -> ToolResult:
        parsed = urlparse(args.url)
        if parsed.scheme not in ("http", "https"):
            return ToolResult.failure("url must be an absolute http(s) URL")
        host = (parsed.hostname or "").lower()
        if host not in self._allowlist:
            return ToolResult.failure(
                f"Host {host!r} is not on the http_request allowlist",
                allowlist=sorted(self._allowlist),
            )
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_S, follow_redirects=False) as client:
                resp = await client.request(
                    args.method,
                    args.url,
                    headers=args.headers,
                    json=args.json_body,
                )
        except httpx.HTTPError as exc:
            raise ExternalServiceError(f"http_request failed: {exc}") from exc

        body = resp.text[:_MAX_BODY_CHARS]
        return ToolResult.success(
            {
                "status_code": resp.status_code,
                "headers": dict(resp.headers),
                "body": body,
            },
            truncated=len(resp.text) > _MAX_BODY_CHARS,
        )
