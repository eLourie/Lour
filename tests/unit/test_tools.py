"""
tests/unit/test_tools.py

Unit coverage for the tools core: BaseTool.run validation, the @tool decorator,
ToolRegistry, the Ollama schema builder and the result cache key. Pure logic —
no backing services.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from app.core.exceptions import NotFoundError
from app.tools.base import BaseTool, ToolResult
from app.tools.cache import cache_key
from app.tools.registry import ToolRegistry, tool
from app.tools.schema import to_ollama_schema

pytestmark = pytest.mark.unit


class AddArgs(BaseModel):
    a: int = Field(description="first addend")
    b: int = 0


class AddTool(BaseTool[AddArgs]):
    name = "add_numbers"
    description = "Add two integers. Do not use for floats."
    args_schema = AddArgs

    async def execute(self, args: AddArgs) -> ToolResult:
        return ToolResult.success(args.a + args.b)


async def test_run_validates_and_executes() -> None:
    result = await AddTool().run({"a": 2, "b": 3})
    assert result.ok
    assert result.data == 5


async def test_run_returns_failure_on_bad_args() -> None:
    result = await AddTool().run({"b": 3})  # missing required `a`
    assert not result.ok
    assert result.metadata["error_type"] == "validation"


async def test_run_never_raises_on_unexpected_error() -> None:
    class Boom(BaseTool[AddArgs]):
        name = "boom"
        description = "Always explodes for testing."
        args_schema = AddArgs

        async def execute(self, args: AddArgs) -> ToolResult:
            raise RuntimeError("kaboom")

    result = await Boom().run({"a": 1})
    assert not result.ok
    assert result.metadata["error_type"] == "unhandled"


def test_ollama_schema_shape() -> None:
    schema = to_ollama_schema(AddTool())
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "add_numbers"
    assert fn["parameters"]["required"] == ["a"]
    # cosmetic `title` keys are stripped
    assert "title" not in fn["parameters"]


def test_tool_decorator_validates_metadata() -> None:
    with pytest.raises(TypeError):

        @tool
        class NoName(BaseTool[AddArgs]):  # missing name
            description = "x"
            args_schema = AddArgs

            async def execute(self, args: AddArgs) -> ToolResult:  # pragma: no cover
                return ToolResult.success()


def test_registry_register_get_and_duplicate() -> None:
    reg = ToolRegistry()
    reg.register(AddTool())
    assert "add_numbers" in reg
    assert reg.get("add_numbers").name == "add_numbers"
    with pytest.raises(ValueError, match="already registered"):
        reg.register(AddTool())
    reg.register(AddTool(), replace=True)  # replace is allowed
    assert len(reg) == 1


def test_registry_get_missing_raises_not_found() -> None:
    reg = ToolRegistry()
    with pytest.raises(NotFoundError):
        reg.get("ghost")


def test_registry_schema_allowlist_filters() -> None:
    reg = ToolRegistry()
    reg.register(AddTool())
    assert reg.to_ollama_schemas(allowed={"other"}) == []
    assert len(reg.to_ollama_schemas(allowed={"add_numbers"})) == 1
    assert len(reg.to_ollama_schemas()) == 1


def test_cache_key_is_order_independent() -> None:
    k1 = cache_key("t", {"a": 1, "b": 2})
    k2 = cache_key("t", {"b": 2, "a": 1})
    assert k1 == k2
    assert cache_key("t", {"a": 1}) != cache_key("t", {"a": 2})
