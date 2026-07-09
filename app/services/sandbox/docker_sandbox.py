"""
app/services/sandbox/docker_sandbox.py

Docker-backed sandbox for untrusted (LLM-generated) code.

Isolation (PROJECT_CONTEXT §7, Phase 3):
  - ``--network=none``      — no egress (network_disabled).
  - read-only root fs       — only a small tmpfs at /tmp is writable.
  - cpu / memory / pids caps — resource exhaustion is bounded.
  - cap_drop=ALL + no-new-privileges + non-root user.
  - wall-clock timeout      — container is killed if it overruns.

The code is delivered as a base64 blob decoded inside the container, so there
are no bind mounts and quoting is never an issue — the read-only rootfs stays
intact. docker-py is synchronous, so every daemon call runs in a worker thread.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import time
from typing import TYPE_CHECKING, Any

from app.core.exceptions import SandboxError
from app.core.logging import get_logger
from app.services.sandbox.base import SandboxLanguage, SandboxResult

if TYPE_CHECKING:
    from app.core.config import SandboxSettings

logger = get_logger(__name__)


def _python_cmd(b64: str) -> list[str]:
    return [
        "python",
        "-c",
        f"import base64;exec(compile(base64.b64decode('{b64}'),'<sandbox>','exec'))",
    ]


def _node_cmd(b64: str) -> list[str]:
    return [
        "node",
        "-e",
        f"eval(Buffer.from('{b64}','base64').toString('utf8'))",
    ]


class DockerSandbox:
    """Runs code in a locked-down, single-use Docker container."""

    def __init__(self, settings: SandboxSettings) -> None:
        self._settings = settings
        self._client: Any = None  # lazy docker.from_env() — daemon may be absent

    def _docker(self) -> Any:
        if self._client is None:
            import docker

            try:
                self._client = docker.from_env()  # type: ignore[attr-defined]
                self._client.ping()
            except Exception as exc:
                raise SandboxError(f"Docker daemon is not reachable: {exc}") from exc
        return self._client

    async def run(
        self,
        code: str,
        *,
        language: SandboxLanguage = SandboxLanguage.PYTHON,
        timeout_s: int | None = None,
    ) -> SandboxResult:
        timeout = timeout_s or self._settings.timeout_s
        return await asyncio.to_thread(self._run_sync, code, language, timeout)

    def _run_sync(self, code: str, language: SandboxLanguage, timeout: int) -> SandboxResult:
        client = self._docker()
        b64 = base64.b64encode(code.encode()).decode()

        if language is SandboxLanguage.PYTHON:
            image, command = self._settings.python_image, _python_cmd(b64)
        elif language is SandboxLanguage.JAVASCRIPT:
            image, command = self._settings.node_image, _node_cmd(b64)
        else:  # pragma: no cover — enum is exhaustive
            raise SandboxError(f"Unsupported sandbox language: {language}")

        start = time.perf_counter()
        container = client.containers.run(
            image=image,
            command=command,
            detach=True,
            # --- isolation ---
            network_disabled=True,
            read_only=True,
            mem_limit=f"{self._settings.memory_mb}m",
            memswap_limit=f"{self._settings.memory_mb}m",  # no swap headroom
            nano_cpus=int(self._settings.cpu_quota * 1_000_000_000),
            pids_limit=self._settings.pids_limit,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            user="nobody",
            working_dir="/tmp",
            tmpfs={"/tmp": "rw,size=32m,mode=1777"},
        )

        timed_out = False
        exit_code: int | None = None
        try:
            try:
                status = container.wait(timeout=timeout)
                exit_code = int(status.get("StatusCode", -1))
            except Exception:
                timed_out = True
                _safe(container.kill)

            stdout = _logs(container, stdout=True)
            stderr = _logs(container, stdout=False)
        finally:
            _safe(lambda: container.remove(force=True))

        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        ok = (not timed_out) and exit_code == 0
        if timed_out:
            stderr = (stderr + f"\n[sandbox] killed after {timeout}s timeout").strip()
        logger.info(
            "sandbox_run",
            language=language,
            ok=ok,
            exit_code=exit_code,
            timed_out=timed_out,
            duration_ms=duration_ms,
        )
        return SandboxResult(
            ok=ok,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            duration_ms=duration_ms,
        )


def _logs(container: Any, *, stdout: bool) -> str:
    try:
        raw = container.logs(stdout=stdout, stderr=not stdout)
        return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    except Exception:
        return ""


def _safe(fn: Any) -> None:
    with contextlib.suppress(Exception):
        fn()
