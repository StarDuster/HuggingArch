"""The TypeScript unit tests (`web/src/*.test.ts`) must pass.

Why this lives in Python: the repository's CI lanes and the deploy gate run
pytest, and nothing runs `npm test`. A check nobody runs is a check that
always passes, so the vitest suite is shelled out to from here — the same
shape as `test_web_build.py`, which is why `npm run build` is actually a
gate. Keep it this way: wiring a second CI lane would leave this suite
unread on local runs and on the deploy gate.

A failure here means a shared `web/src` module no longer behaves as its
contract test says — run `npm test` in `web/` to see which property moved.
Locally a missing Node toolchain is a skip; in CI it is a failure, because
there the gate is supposed to exist.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB = PROJECT_ROOT / "web"


def _missing(reason: str) -> None:
    if os.environ.get("CI"):
        pytest.fail(f"web/ unit test gate cannot run in CI: {reason}")
    pytest.skip(reason)


def test_web_unit_tests_pass():
    if shutil.which("node") is None:
        _missing("node not on PATH; the web/ unit tests require Node")
    if shutil.which("npm") is None:
        _missing("npm not on PATH; the web/ unit tests require npm")
    if not (WEB / "node_modules").exists():
        _missing("web/node_modules missing; run `npm ci` in web/")
    # Own process group: npm spawns vitest as a grandchild, and killing npm
    # alone on timeout leaves vitest's workers behind.
    proc = subprocess.Popen(
        ["npm", "test", "--silent"],
        cwd=WEB, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        pytest.fail("web/ unit tests timed out after 180s")
    assert proc.returncode == 0, f"web/ unit tests failed:\n{stdout}\n{stderr}"
    # A suite whose files stopped matching the `include` glob reports no tests
    # at all, which must not read as a green run.
    out = stdout + stderr
    plain = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", out)
    assert "No test files found" not in out, f"vitest collected no test files:\n{out}"
    assert "Tests  " in plain and " passed " in plain, (
        f"vitest did not report a passing run:\n{out}"
    )
