"""
tests/unit/test_query_transform.py

Unit tests for QueryTransformer with a fake LLM (no I/O).
"""

from __future__ import annotations

from typing import Any

import pytest

from app.services.llm.base import LLMResponse
from app.services.rag.query_transform import QueryTransformer

pytestmark = pytest.mark.unit


class FakeLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    async def chat(self, messages: list[dict[str, Any]], **_: Any) -> LLMResponse:
        return LLMResponse(content=self._content)

    def stream(self, messages: list[dict[str, Any]], **_: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError


async def test_hyde_returns_passage() -> None:
    transformer = QueryTransformer(FakeLLM("The tower was designed by Gustave Eiffel."))
    passage = await transformer.hyde("who designed it?")
    assert "Gustave Eiffel" in passage


async def test_hyde_falls_back_to_query_when_empty() -> None:
    transformer = QueryTransformer(FakeLLM("   "))
    assert await transformer.hyde("original query") == "original query"


async def test_multi_query_includes_original_and_dedups() -> None:
    llm_output = "1. rewrite one\n2. rewrite two\n- rewrite one\n"
    transformer = QueryTransformer(FakeLLM(llm_output))
    queries = await transformer.multi_query("original", n=3)

    assert queries[0] == "original"
    assert "rewrite one" in queries
    assert "rewrite two" in queries
    # deduped (rewrite one appears once) and capped at n+1
    assert len(queries) == len({q.lower() for q in queries})
    assert len(queries) <= 4
