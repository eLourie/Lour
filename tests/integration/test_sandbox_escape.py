"""
tests/integration/test_sandbox_escape.py

Isolation guarantees of the Docker sandbox. Requires a running Docker daemon and
the python sandbox image (``docker pull python:3.12-slim``).

Run with: pytest -m integration tests/integration/test_sandbox_escape.py
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.core.config import get_settings
from app.services.sandbox.base import SandboxLanguage
from app.services.sandbox.docker_sandbox import DockerSandbox

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sandbox() -> Iterator[DockerSandbox]:
    sb = DockerSandbox(get_settings().sandbox)
    yield sb


async def test_runs_python_and_captures_stdout(sandbox: DockerSandbox) -> None:
    result = await sandbox.run("print(6 * 7)")
    assert result.ok, result.stderr
    assert result.stdout.strip() == "42"
    assert result.exit_code == 0


async def test_network_egress_is_blocked(sandbox: DockerSandbox) -> None:
    code = (
        "import urllib.request as u\n"
        "u.urlopen('http://example.com', timeout=5)\n"
    )
    result = await sandbox.run(code)
    assert not result.ok
    # DNS/connect must fail because the container has no network.
    assert result.stderr


async def test_wall_clock_timeout_is_enforced(sandbox: DockerSandbox) -> None:
    result = await sandbox.run("while True:\n    pass", timeout_s=3)
    assert result.timed_out
    assert not result.ok


async def test_root_filesystem_is_read_only(sandbox: DockerSandbox) -> None:
    # Writing outside the /tmp tmpfs must fail on the read-only rootfs.
    code = (
        "try:\n"
        "    open('/escape.txt', 'w').write('x')\n"
        "    print('WROTE')\n"
        "except OSError as e:\n"
        "    print('BLOCKED', e.errno)\n"
    )
    result = await sandbox.run(code)
    assert result.ok  # program ran and handled the error
    assert "BLOCKED" in result.stdout
    assert "WROTE" not in result.stdout


async def test_nonzero_exit_is_reported(sandbox: DockerSandbox) -> None:
    result = await sandbox.run("import sys; sys.exit(3)")
    assert not result.ok
    assert result.exit_code == 3
    assert not result.timed_out


async def test_language_enum_dispatch(sandbox: DockerSandbox) -> None:
    # Sanity: the python path is selected for the default language.
    result = await sandbox.run("print('py')", language=SandboxLanguage.PYTHON)
    assert result.stdout.strip() == "py"
