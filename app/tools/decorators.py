"""
app/tools/decorators.py

Cross-cutting decorators for tool ``execute`` methods:

  @retried — retry transient failures with exponential backoff (tenacity).
  @audited — structured before/after log line with timing and ok/err outcome.
  @cached  — memoise idempotent results in-process for a short TTL.

These wrap a tool's ``async def execute(self, args)`` method. Redis-backed
result caching for the whole registry lives in ``cache.py``; ``@cached`` here is
a lightweight per-instance memoiser useful for tools without a Redis handle
(e.g. datetime is intentionally *not* cached — it must stay fresh).
"""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.core.metrics import get_metrics

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.tools.base import ToolResult

logger = get_logger(__name__)


def retried[F: Callable[..., Awaitable[ToolResult]]](
    *,
    attempts: int = 3,
    exceptions: tuple[type[Exception], ...] = (ExternalServiceError,),
) -> Callable[[F], F]:
    """Retry ``execute`` on transient exceptions with exponential backoff."""

    def decorator(func: F) -> F:
        retrying = retry(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=0.2, max=2.0),
            retry=retry_if_exception_type(exceptions),
            reraise=True,
        )
        return retrying(func)

    return decorator


def audited[F: Callable[..., Awaitable[ToolResult]]](func: F) -> F:
    """Emit a structured log line with timing and outcome around ``execute``."""

    @functools.wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> ToolResult:
        name = getattr(self, "name", func.__qualname__)
        start = time.perf_counter()
        result = await func(self, *args, **kwargs)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        get_metrics().record_tool(name, result.ok)
        logger.info(
            "tool_executed",
            tool=name,
            ok=result.ok,
            elapsed_ms=elapsed_ms,
            error=result.error,
        )
        return result

    return wrapper  # type: ignore[return-value]


def cached[F: Callable[..., Awaitable[ToolResult]]](
    *, ttl_s: float = 60.0, maxsize: int = 256
) -> Callable[[F], F]:
    """
    Memoise ``execute`` results per-instance for ``ttl_s`` seconds.

    Keyed by the JSON of the validated args model. Only successful results are
    cached. Intended for cheap, dependency-free tools; Redis-backed caching for
    the registry lives in ``cache.py``.
    """

    def decorator(func: F) -> F:
        store: dict[str, tuple[float, ToolResult]] = {}

        @functools.wraps(func)
        async def wrapper(self: Any, args: Any, *rest: Any, **kwargs: Any) -> ToolResult:
            try:
                key = args.model_dump_json()
            except AttributeError:
                key = repr(args)
            now = time.monotonic()
            hit = store.get(key)
            if hit is not None and now - hit[0] < ttl_s:
                return hit[1]

            result = await func(self, args, *rest, **kwargs)
            if result.ok:
                if len(store) >= maxsize:
                    store.pop(next(iter(store)))
                store[key] = (now, result)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
