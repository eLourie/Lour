"""
app/core/telemetry.py

Decorators and helpers for LLM / tool observability.

@traced  — wraps any async function in a Langfuse span (or no-op).
@observe — same but intended for LLM calls (logs input/output tokens).

structlog JSON logging is always active (offline baseline).
Langfuse is the optional cloud layer, enabled by TELEMETRY_BACKEND.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

from app.core.logging import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

# Module-level reference; set once during lifespan startup.
# If None, all decorators are no-ops.
_telemetry_client: Any = None  # type: ignore[assignment]


def set_telemetry_client(client: Any) -> None:  # noqa: ANN401
    """Called from lifespan after the client is built."""
    global _telemetry_client  # noqa: PLW0603
    _telemetry_client = client


def traced(
    name: str | None = None,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """
    Decorator that wraps an async function in a telemetry span.

    Usage::

        @traced("rag_retrieval")
        async def retrieve(query: str) -> list[Chunk]:
            ...
    """

    def decorator(
        func: Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        span_name = name or func.__qualname__

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.debug("span_ok", span=span_name, elapsed_ms=round(elapsed * 1000, 1))
                if _telemetry_client is not None:
                    _telemetry_client.trace(span_name, latency_ms=elapsed * 1000)
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                logger.warning(
                    "span_error",
                    span=span_name,
                    elapsed_ms=round(elapsed * 1000, 1),
                    error=str(exc),
                )
                raise

        return wrapper

    return decorator


# observe is an alias with the same logic; distinction is semantic (LLM calls)
observe = traced