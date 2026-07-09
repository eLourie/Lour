"""
app/gateway/routes/tools.py

Tools introspection API (internal / debugging, PROJECT_CONTEXT §5.1 — tools are
private to the LLM, unlike the public skills catalogue).

    GET /v1/tools        — list every registered tool with its native schema.
    GET /v1/tools/{name} — one tool's schema, or 404.

Used to verify the registry (incl. MCP-adapter tools) after startup and to
inspect the exact schema the model sees.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.core.di import get_state
from app.core.exceptions import NotFoundError
from app.tools.schema import to_ollama_schema

if TYPE_CHECKING:
    from app.tools.registry import ToolRegistry

router = APIRouter(prefix="/tools", tags=["tools"])

RegistryDep = Annotated["ToolRegistry", Depends(get_state("tool_registry"))]


class ToolInfo(BaseModel):
    # `schema` is a reserved Pydantic method name → store as schema_, emit as "schema".
    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str
    side_effects: bool
    schema_: dict[str, Any] = Field(serialization_alias="schema")


class ToolListResponse(BaseModel):
    count: int
    tools: list[ToolInfo]


def _info(tool: Any) -> ToolInfo:
    return ToolInfo(
        name=tool.name,
        description=tool.description,
        side_effects=tool.side_effects,
        schema_=to_ollama_schema(tool),
    )


@router.get("", response_model=ToolListResponse)
async def list_tools(registry: RegistryDep) -> ToolListResponse:
    """List all registered tools (builtins + MCP adapters)."""
    tools = sorted(registry.all(), key=lambda t: t.name)
    return ToolListResponse(count=len(tools), tools=[_info(t) for t in tools])


@router.get("/{name}", response_model=ToolInfo)
async def get_tool(name: str, registry: RegistryDep) -> ToolInfo:
    """Return one tool's schema, or 404 if it is not registered."""
    if name not in registry:
        raise NotFoundError(f"Tool {name!r} is not registered", code="tool_not_found")
    return _info(registry.get(name))
