"""
app/tools/schema.py

Convert a tool's Pydantic ``args_schema`` into the Ollama / OpenAI-compatible
function-calling schema:

    {
      "type": "function",
      "function": {
        "name": ...,
        "description": ...,
        "parameters": { <JSON Schema of the args model> }
      }
    }

Ollama passes ``parameters`` straight to the model as native tool definitions
(ADR-007). We strip Pydantic's ``title`` noise and inline any ``$defs`` so the
schema the model sees is flat and minimal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.tools.base import BaseTool


def to_ollama_schema(tool: BaseTool[Any]) -> dict[str, Any]:
    """Return the native tool-calling schema for a single tool."""
    parameters = _clean_json_schema(tool.ollama_parameters())
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        },
    }


def _clean_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Inline ``$defs`` references and drop cosmetic ``title`` keys so the model
    receives a compact, self-contained parameter schema.
    """
    defs = schema.pop("$defs", {})
    cleaned = _resolve(schema, defs)
    if not isinstance(cleaned, dict):  # pragma: no cover — top level is always object
        return schema
    return cleaned


def _resolve(node: Any, defs: dict[str, Any]) -> Any:
    """Recursively inline $ref targets and strip ``title`` keys."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            target = defs.get(ref.split("/")[-1], {})
            return _resolve(target, defs)
        return {k: _resolve(v, defs) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_resolve(item, defs) for item in node]
    return node
