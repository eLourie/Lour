"""
app/services/rag/query_transform.py

Query transformations that improve recall before retrieval.

  - HyDE (Hypothetical Document Embeddings): the LLM drafts a plausible answer
    to the query; we embed *that* and search with it. A hypothetical answer
    lives closer to real answer passages in embedding space than the question.
  - Multi-query expansion: the LLM rewrites the query several ways; retrieving
    for each and merging widens coverage of differently-phrased passages.

Both are optional enhancements — the retriever works without them. They depend
only on the LLMProvider Protocol, so any provider works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.logging import get_logger
from app.core.telemetry import traced

if TYPE_CHECKING:
    from app.services.llm.base import LLMProvider

logger = get_logger(__name__)

_HYDE_PROMPT = (
    "Write a short, factual paragraph that directly answers the question below, "
    "as if quoting an authoritative document. Do not add caveats or say you are "
    "unsure — just write the passage.\n\nQuestion: {query}\n\nPassage:"
)

_MULTIQUERY_PROMPT = (
    "Rewrite the following search query in {n} different ways to maximise the "
    "chance of matching relevant documents. Vary wording and specificity. "
    "Return one rewrite per line, no numbering, no extra text.\n\nQuery: {query}"
)


class QueryTransformer:
    """LLM-backed query rewriting (HyDE + multi-query)."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    @traced("rag_hyde")
    async def hyde(self, query: str) -> str:
        """Return a hypothetical answer passage for *query*."""
        response = await self._llm.chat(
            [{"role": "user", "content": _HYDE_PROMPT.format(query=query)}]
        )
        passage = response.content.strip()
        return passage or query

    @traced("rag_multi_query")
    async def multi_query(self, query: str, *, n: int = 3) -> list[str]:
        """Return *query* plus up to ``n`` rewrites (deduplicated, order-stable)."""
        response = await self._llm.chat(
            [{"role": "user", "content": _MULTIQUERY_PROMPT.format(query=query, n=n)}]
        )
        rewrites = self._parse_lines(response.content)

        seen: set[str] = set()
        out: list[str] = []
        for candidate in [query, *rewrites]:
            key = candidate.strip().lower()
            if candidate.strip() and key not in seen:
                seen.add(key)
                out.append(candidate.strip())
            if len(out) >= n + 1:
                break
        return out

    @staticmethod
    def _parse_lines(text: str) -> list[str]:
        lines: list[str] = []
        for raw in text.splitlines():
            line = raw.strip().lstrip("-*0123456789.) ").strip()
            if line:
                lines.append(line)
        return lines
