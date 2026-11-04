"""KimiBackend keeps the spec prompt off argv (MAX_ARG_STRLEN / Errno 7).

kimi's CLI has no stdin prompt path (bare `-p` errors, `-p -` is literal,
piped stdin is ignored — verified on 0.34.0), so the prompt goes through a
temp file referenced as `@path`; see the spawn() comment in
backend/agents/kimi.py.
"""
from __future__ import annotations

import os
import subprocess
import time
from unittest import mock

from backend.agents.base import PROJECT_ROOT
from backend.agents.kimi import KimiBackend


def _spawn_capturing(prompt, env=None):
    """Run spawn() against a fake Popen, capturing argv/kwargs and reading the
    prompt file synchronously — the daemon reaper unlinks it once the (fake)
    process "exits", which a Mock's wait() does immediately."""
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        captured["prompt_path"] = cmd[cmd.index("-p") + 1]
        if captured["prompt_path"].startswith("@"):
            with open(captured["prompt_path"][1:], encoding="utf-8") as fh:
                captured["prompt_text"] = fh.read()
        return mock.Mock()

    with mock.patch("backend.agents.kimi.subprocess.Popen", side_effect=fake_popen):
        KimiBackend().spawn(prompt, env=env if env is not None else {})
    return captured


def test_prompt_not_on_argv():
    # Larger than the 128 KB single-argv-entry limit that broke argv passing.
    prompt = "spec prompt body\n" * 10000
    captured = _spawn_capturing(prompt)

    cmd = captured["cmd"]
    assert prompt not in cmd
    assert all(len(arg) < 4096 for arg in cmd)
    assert cmd[cmd.index("-p") + 1].startswith("@")


def test_prompt_delivered_via_referenced_file():
    prompt = "line one\nline two — spec-gen ground truth\n" * 100
    captured = _spawn_capturing(prompt)

    assert captured["prompt_text"] == prompt
    assert os.path.basename(captured["prompt_path"][1:]).startswith("ha-kimi-prompt-")


def test_command_shape_and_cwd_unchanged():
    captured = _spawn_capturing("prompt")

    cmd = captured["cmd"]
    assert cmd[0] == "kimi"
    assert cmd[1] == "-p"
    # kimi-code 0.34+ rejects --print/--work-dir; `-p` alone is already
    # non-interactive and Popen's cwd pins the working directory.
    assert cmd[3:] == ["--output-format", "stream-json"]
    kwargs = captured["kwargs"]
    assert kwargs["cwd"] == PROJECT_ROOT
    assert kwargs["stdout"] == subprocess.PIPE
    assert kwargs["stderr"] == subprocess.PIPE
    assert kwargs["text"] is True


def test_byok_env_passed_through_and_secrets_scrubbed(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "must-not-leak")
    captured = _spawn_capturing("prompt", env={"KIMI_MODEL_API_KEY": "user-key"})

    env = captured["kwargs"]["env"]
    assert env["KIMI_MODEL_API_KEY"] == "user-key"
    assert "SUPABASE_SERVICE_ROLE_KEY" not in env


def test_prompt_file_reaped_after_process_exits():
    captured = _spawn_capturing("prompt")
    path = captured["prompt_path"][1:]

    # The fake proc's wait() returns immediately, so the daemon reaper should
    # unlink the file promptly.
    deadline = time.time() + 5
    while os.path.exists(path) and time.time() < deadline:
        time.sleep(0.01)
    assert not os.path.exists(path)
