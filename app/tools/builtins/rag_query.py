"""
app/tools/builtins/rag_query.py

rag_query — the bridge from the Tools layer to the Phase-2 HybridRetriever.
Lets an agent search the personal knowledge base as a tool call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from app.tools.base import BaseTool, ToolResult
from app.tools.decorators import audited
from app.tools.registry import tool

if TYPE_CHECKING:
    from app.services.rag.retrieval import HybridRetriever


class RagQueryArgs(BaseModel):
    query: str = Field(description="Natural-language question to search the knowledge base.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to return.")
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata equality filters, e.g. {'doc_type': 'markdown'}.",
    )


@tool
class RagQueryTool(BaseTool[RagQueryArgs]):
    name = "rag_query"
    description = (
        "Search the personal knowledge base and return the most relevant chunks "
        "with their source. Use to answer questions about ingested documents. "
        "Do NOT use for general web knowledge (use web_search)."
    )
    args_schema = RagQueryArgs

    def __init__(self, retriever: HybridRetriever) -> None:
        self._retriever = retriever

    @audited
    async def execute(self, args: RagQueryArgs) -> ToolResult:
        chunks = await self._retriever.retrieve(
            args.query, top_k=args.top_k, filters=args.filters
        )
        results = [
            {
                "content": c.content,
                "score": c.score,
                "source_uri": c.source_uri,
                "title": c.title,
                "document_id": c.document_id,
            }
            for c in chunks
        ]
        return ToolResult.success(results, count=len(results))
