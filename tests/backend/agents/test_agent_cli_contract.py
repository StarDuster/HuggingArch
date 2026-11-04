"""Versioned CLI parameter contracts for Codex/Kimi spawn and resume.

Workspace is Popen cwd (and HOME/TMPDIR/HA_SPEC_OVERLAY_DIR). Resume argv
must only use flags the installed CLI's `--help` actually lists.
Dockerfile pins (`CODEX_VERSION`, `KIMI_VERSION`) are the production
target; the live `--help` check runs against whatever binary is on PATH.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from backend.agents.codex import CodexBackend
from backend.agents.kimi import KimiBackend


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "Dockerfile"
FLAG_RE = re.compile(r"(?<![\w-])(-{1,2}[A-Za-z][\w-]*)")


def _dockerfile_pin(name: str) -> str:
    match = re.search(rf"^ARG {name}=(\S+)", DOCKERFILE.read_text(), re.M)
    assert match, f"Dockerfile missing ARG {name}"
    return match.group(1)


def _run_cli(argv, timeout=25):
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, check=False,
    )


def _help_flags(help_text: str) -> set[str]:
    return set(FLAG_RE.findall(help_text))


def _argv_flags(argv) -> list[str]:
    return [a for a in argv if a.startswith("-") and a != "-"]


def _codex_resume_argv(session_id="thr-1"):
    return [
        "codex", "exec", "resume", "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        session_id, "-",
    ]


def _kimi_resume_flags():
    return ["-p", "--output-format", "--session"]


def test_dockerfile_pins_agent_cli_versions():
    assert re.fullmatch(r"\d+\.\d+\.\d+", _dockerfile_pin("CODEX_VERSION"))
    assert re.fullmatch(r"\d+\.\d+\.\d+", _dockerfile_pin("KIMI_VERSION"))


def test_codex_resume_argv_shape_has_no_directory_flag():
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        proc = mock.Mock()
        proc.stdin = mock.Mock()
        return proc

    with mock.patch("backend.agents.codex.subprocess.Popen", side_effect=fake_popen):
        CodexBackend().spawn_resume("thr-1", "prompt", env={})
    argv = captured["cmd"]
    assert argv[:3] == ["codex", "exec", "resume"]
    assert "-C" not in argv
    assert "--cd" not in argv
    assert argv == _codex_resume_argv()
    flags = _argv_flags(argv)
    assert flags == ["--json", "--dangerously-bypass-approvals-and-sandbox"]


def test_kimi_resume_argv_uses_session_long_option():
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return mock.Mock()

    with mock.patch("backend.agents.kimi.subprocess.Popen", side_effect=fake_popen):
        KimiBackend().spawn_resume("k-1", "prompt", env={})
    cmd = captured["cmd"]
    assert cmd[0] == "kimi"
    assert "--session" in cmd and cmd[cmd.index("--session") + 1] == "k-1"
    assert "-r" not in cmd
    assert "--work-dir" not in cmd
    assert set(_argv_flags(cmd)) >= set(_kimi_resume_flags())


@pytest.mark.skipif(shutil.which("codex") is None, reason="codex CLI not installed")
def test_installed_codex_resume_help_matches_spawn_resume_argv():
    pin = _dockerfile_pin("CODEX_VERSION")
    version = _run_cli(["codex", "--version"])
    version_text = (version.stdout + version.stderr).strip()

    help_run = _run_cli([
        "codex", "exec", "resume", "--json",
        "--dangerously-bypass-approvals-and-sandbox", "--help",
    ])
    help_text = help_run.stdout + help_run.stderr
    assert help_run.returncode == 0, (
        f"codex exec resume flags rejected ({version_text}; "
        f"Dockerfile pin {pin}): {help_text[-500:]}"
    )
    flags = _help_flags(help_text)
    for flag in _argv_flags(_codex_resume_argv()):
        assert flag in flags, (
            f"{flag} missing from `codex exec resume --help` "
            f"({version_text}; pin {pin})"
        )
    assert "-C" not in flags
    assert "--cd" not in flags

    rejected = _run_cli([
        "codex", "exec", "resume", "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C", "/tmp", "--help",
    ])
    rejected_text = rejected.stdout + rejected.stderr
    assert rejected.returncode == 2, (
        f"expected exit 2 for resume -C ({version_text}; pin {pin}), "
        f"got {rejected.returncode}: {rejected_text[-500:]}"
    )
    assert "unexpected argument '-C'" in rejected_text

    exec_help = _run_cli(["codex", "exec", "--help"])
    exec_flags = _help_flags(exec_help.stdout + exec_help.stderr)
    assert "-C" in exec_flags or "--cd" in exec_flags


@pytest.mark.skipif(shutil.which("kimi") is None, reason="kimi CLI not installed")
def test_installed_kimi_help_matches_spawn_resume_argv():
    pin = _dockerfile_pin("KIMI_VERSION")
    version = _run_cli(["kimi", "--version"])
    version_text = (version.stdout + version.stderr).strip().splitlines()[-1]

    help_run = _run_cli(["kimi", "--help"])
    help_text = help_run.stdout + help_run.stderr
    assert help_run.returncode == 0, (
        f"kimi --help failed ({version_text}; Dockerfile pin {pin}): "
        f"{help_text[-500:]}"
    )
    flags = _help_flags(help_text)
    for flag in _kimi_resume_flags():
        assert flag in flags, (
            f"{flag} missing from `kimi --help` ({version_text}; pin {pin})"
        )
    assert "--work-dir" not in flags
    assert "-r" not in flags

    missing = _run_cli([
        "kimi", "-p", "hi", "--output-format", "text",
        "--session", "ha-contract-missing-session",
    ], timeout=15)
    missing_text = missing.stdout + missing.stderr
    assert missing.returncode != 0
    assert "not found" in missing_text.lower() or "session" in missing_text.lower(), (
        f"--session was not treated as resume ({version_text}; pin {pin}): "
        f"{missing_text[-500:]}"
    )
