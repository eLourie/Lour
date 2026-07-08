"""
app/core/logging.py

structlog-based JSON logging — the always-on observability baseline.
Works fully offline (no external services required).

Context variables (request_id, session_id, trace_id) are bound per-request
via contextvars and automatically included in every log record within that scope.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import structlog

# Context variables — populated by request-id middleware

_request_id_var: ContextVar[str] = ContextVar("request_id", default="")
_session_id_var: ContextVar[str] = ContextVar("session_id", default="")
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def set_request_context(
    *,
    request_id: str | None = None,
    session_id: str | None = None,
    trace_id: str | None = None,
) -> None:
    """Bind context values for the current async task."""
    if request_id is not None:
        _request_id_var.set(request_id)
    if session_id is not None:
        _session_id_var.set(session_id)
    if trace_id is not None:
        _trace_id_var.set(trace_id)


def get_request_id() -> str:
    return _request_id_var.get()


def get_session_id() -> str:
    return _session_id_var.get()


def get_trace_id() -> str:
    return _trace_id_var.get()


# Processors


def _inject_context(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add request/session/trace IDs from contextvars to every log record."""
    if rid := _request_id_var.get():
        event_dict["request_id"] = rid
    if sid := _session_id_var.get():
        event_dict["session_id"] = sid
    if tid := _trace_id_var.get():
        event_dict["trace_id"] = tid
    return event_dict


def _drop_color_message(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Remove uvicorn's 'color_message' key — redundant in JSON output."""
    event_dict.pop("color_message", None)
    return event_dict


# Setup


def configure_logging(log_level: str = "INFO", *, json_logs: bool = True) -> None:
    """
    Configure structlog for the entire application.

    Call once at startup (in app lifespan or main.py).
    json_logs=False produces pretty dev output; True (default) produces
    machine-readable JSON for production / log aggregators.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _inject_context,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        _drop_color_message,
    ]

    if json_logs:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.upper())

    # Silence noisy libraries
    for noisy in ("httpx", "httpcore", "asyncio", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> Any:
    """Return a structlog bound logger. Use at module level."""
    return structlog.get_logger(name)


def new_request_id() -> str:
    """Generate a new unique request ID."""
    return uuid4().hex
