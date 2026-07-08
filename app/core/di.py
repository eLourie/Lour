"""
app/core/di.py

Dependency-injection helpers for FastAPI.

Pattern: singletons are created once in app lifespan (app/main.py) and stored
on app.state. FastAPI Depends() functions retrieve them from request.app.state.

This keeps the DI surface minimal (no magic container) while remaining
fully testable (swap app.state in tests).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request

# Generic state accessor


def get_state(attr: str) -> Any:
    """
    Return a FastAPI Depends()-compatible callable that retrieves
    `request.app.state.<attr>`.

    Usage:
        async def my_route(
            llm: Annotated[LLMService, Depends(get_state("llm_service"))],
        ) -> ...:
            ...
    """

    def _dependency(request: Request) -> Any:
        try:
            return getattr(request.app.state, attr)
        except AttributeError as exc:
            raise RuntimeError(
                f"app.state.{attr!r} is not set — make sure it is initialised in the lifespan."
            ) from exc

    return _dependency


# Typed accessor factories (avoids string typos in route signatures)


def state_dep[T](attr: str, *, annotation: type[T]) -> Any:
    """
    Like get_state() but returns a properly typed FastAPI dependency.

    Usage:
        GetSettings = state_dep("settings", annotation=Settings)

        async def my_route(settings: Annotated[Settings, Depends(GetSettings)]) -> ...:
            ...
    """

    # We return a plain callable; FastAPI resolves the annotation from Annotated[].
    def _dependency(request: Request) -> T:
        try:
            return getattr(request.app.state, attr)  # type: ignore[no-any-return]
        except AttributeError as exc:
            raise RuntimeError(
                f"app.state.{attr!r} is not set — make sure it is initialised in the lifespan."
            ) from exc

    return _dependency


# Request-context helpers (not stored on app.state)


async def get_request_id(request: Request) -> str:
    """Return the request_id attached by RequestIDMiddleware."""
    return getattr(request.state, "request_id", "")


RequestIDDep = Annotated[str, Depends(get_request_id)]
