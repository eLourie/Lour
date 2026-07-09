"""
app/infra/clients/telemetry.py

Thin adapter that initialises the Langfuse SDK (or a no-op stub)
based on TELEMETRY_BACKEND.

structlog JSON logging is always active (the offline baseline).
This module adds the optional Langfuse Cloud layer on top.

Note on trace():
    The Langfuse SDK (v3+) does not expose a .trace() method on the
    client directly. Real LLM/tool tracing is done via the @observe
    decorator in app/core/telemetry.py (Phase 8). This client's role
    is lifecycle management only: init, flush, shutdown.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.config import TelemetryBackend, TelemetrySettings
from app.core.logging import get_logger

if TYPE_CHECKING:
    from langfuse import Langfuse

logger = get_logger(__name__)


class NoOpTelemetryClient:
    """Fulfils the TelemetryClient interface when Langfuse is disabled."""

    def trace(self, name: str, **kwargs: Any) -> None:
        pass

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class LangfuseTelemetryClient:
    """
    Wraps the Langfuse SDK for lifecycle management.

    Real tracing (LLM calls, tool executions) is handled by the
    @observe decorator in app/core/telemetry.py which uses the
    Langfuse SDK's context manager API directly.

    This class is responsible for: init, flush, shutdown.
    """

    def __init__(self, settings: TelemetrySettings) -> None:
        self._lf: Langfuse | None = None
        try:
            from langfuse import Langfuse as LangfuseSDK

            self._lf = LangfuseSDK(
                public_key=settings.public_key,
                secret_key=settings.secret_key,
                host=settings.host,
            )
            logger.info(
                "telemetry_initialised",
                backend="langfuse",
                host=settings.host,
            )
        except Exception as exc:
            logger.warning("telemetry_init_failed", error=str(exc))

    def trace(self, name: str, **kwargs: Any) -> None:
        """No-op at client level — real tracing via @observe decorator."""
        pass

    def flush(self) -> None:
        if self._lf is not None:
            self._lf.flush()

    def shutdown(self) -> None:
        if self._lf is not None:
            self._lf.shutdown()


def build_telemetry_client(
    settings: TelemetrySettings,
) -> LangfuseTelemetryClient | NoOpTelemetryClient:
    """Factory: return the right client based on TELEMETRY_BACKEND."""
    if settings.backend == TelemetryBackend.NONE:
        logger.info("telemetry_disabled")
        return NoOpTelemetryClient()
    return LangfuseTelemetryClient(settings)
