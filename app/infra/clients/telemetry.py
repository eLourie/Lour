"""
app/infra/clients/telemetry.py

Thin adapter that initialises the Langfuse SDK (or a no-op stub)
based on TELEMETRY_BACKEND.

structlog JSON logging is always active (the offline baseline).
This module adds the optional Langfuse Cloud layer on top.
"""

from __future__ import annotations

from typing import Any

from app.core.config import TelemetryBackend, TelemetrySettings
from app.core.logging import get_logger

logger = get_logger(__name__)


class NoOpTelemetryClient:
    """Fulfils the TelemetryClient interface when Langfuse is disabled."""

    def trace(self, name: str, **kwargs: Any) -> None:  # noqa: ANN401
        pass

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class LangfuseTelemetryClient:
    """
    Wraps the Langfuse SDK.

    Only instantiated when TELEMETRY_BACKEND != none and the SDK is
    importable. Falls back to NoOpTelemetryClient on import error.
    """

    def __init__(self, settings: TelemetrySettings) -> None:
        try:
            from langfuse import Langfuse  # type: ignore[import-untyped]

            self._lf = Langfuse(
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
            self._lf = None

    def trace(self, name: str, **kwargs: Any) -> None:  # noqa: ANN401
        if self._lf is not None:
            self._lf.trace(name=name, **kwargs)

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