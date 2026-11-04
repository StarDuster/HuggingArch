"""Agent output streams are user-visible; deployment paths must not leak.

A spec-gen run's stream echoes the agent's prose and `$ cd <abs> …` commands.
The per-run overlay root embeds the deploy home AND, for signed-in runs, the
triggering user's UUID — both showed up in the Inspector's generation overlay.
The runner therefore emits scrubbed frames (the on-disk trace log stays raw),
the job buffer re-scrubs yields that bypass the runner, and the DB-bound
run-log upload (owner-readable via /api/account/gen-logs/{id}) is scrubbed
wholesale.
"""
from __future__ import annotations

import io
import json
import threading
import time

from backend._agent_runner import runner as R
from backend._agent_runner.scrub import scrub_paths
from backend.agents.base import PROJECT_ROOT, AgentBackend


_OVERLAY = "/srv/ha/spec_runs/07a6e2a7-54b4-4e52-ba0d-21ce44f06b09/zai-org__GLM-5.3-Flash"


def test_overlay_root_stripped_to_relative_tail(monkeypatch):
    monkeypatch.delenv("HA_AGENT_HOME", raising=False)
    out = scrub_paths(f"$ cd {_OVERLAY}/backend/arch && ls quant", _OVERLAY)
    assert out == "$ cd backend/arch && ls quant"


def test_overlay_nested_under_home_leaks_no_uuid(monkeypatch):
    monkeypatch.delenv("HA_AGENT_HOME", raising=False)
    monkeypatch.setenv("HOME", "/home/deploy")
    overlay = "/home/deploy/.huggingarch/spec_runs/USER-UUID/meta__Foo"
    out = scrub_paths(f"cd {overlay}/models && ls {overlay}/backend/arch", overlay)
    assert "USER-UUID" not in out
    assert "/home/deploy" not in out
    assert out == "cd models && ls backend/arch"


def test_home_collapses_to_tilde(monkeypatch):
    monkeypatch.delenv("HA_AGENT_HOME", raising=False)
    monkeypatch.setenv("HOME", "/home/deploy")
    out = scrub_paths("cache at /home/deploy/.cache/huggingface/hub")
    assert out == "cache at ~/.cache/huggingface/hub"


def test_agent_home_collapses_to_tilde(monkeypatch):
    monkeypatch.setenv("HA_AGENT_HOME", "/home/specagent")
    out = scrub_paths("state in /home/specagent/.kimi/session.json")
    assert out == "state in ~/.kimi/session.json"


def test_project_root_stripped(monkeypatch):
    monkeypatch.delenv("HA_AGENT_HOME", raising=False)
    out = scrub_paths(f"cd {PROJECT_ROOT}/backend/arch/models")
    assert out == "cd backend/arch/models"
    assert PROJECT_ROOT not in out


def test_path_boundary_respected(monkeypatch):
    monkeypatch.delenv("HA_AGENT_HOME", raising=False)
    monkeypatch.setenv("HOME", "/home/deploy")
    # "/app" is a substring of "/application" — only a real path match scrubs.
    assert scrub_paths("ls /application/x", "/app") == "ls /application/x"
    assert scrub_paths("cd /app", "/app") == "cd ."
    assert scrub_paths("no /home/deployer here") == "no /home/deployer here"


class _FakeProc:
    def __init__(self, stdout_lines, returncode=0):
        self._lines = list(stdout_lines)
        self.stderr = io.StringIO("")
        self.pid = 4242
        self.returncode = returncode

    @property
    def stdout(self):
        return self

    def readline(self):
        return self._lines.pop(0) if self._lines else ""

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


class _FakeBackend(AgentBackend):
    """A backend that replays fixed events. It inherits the contract (how an
    ending is classified, what the CLI reports) rather than copying pieces of
    it, so the runner's next question of a backend is answered here too."""
    supports_resume = False
    supports_isolated_workspace = True

    def __init__(self, events):
        self._events = events

    def spawn(self, prompt, env=None, workspace=None):
        return _FakeProc(['{"raw": 1}\n'])

    def extract_session_id(self, evt):
        return None

    def parse_event(self, evt):
        return self._events


def _run_frames(monkeypatch, tmp_path, events, target_root):
    monkeypatch.setattr(R, "get_backend", lambda cli: _FakeBackend(events))
    log_path = tmp_path / "log.txt"
    frames = list(R.stream_agent_core(
        key="m", prep_result={"prompt": "p", "target_root": target_root},
        agent_cli="fake", log_path=str(log_path), log_header="h",
        launch_banner="B", procs_dict={}, locks_lock=threading.Lock(),
        release_fn=lambda k: None,
    ))
    parsed = [json.loads(f[len("data: "):].strip())
              for f in frames if f.startswith("data: ")]
    return parsed, log_path.read_text()


def test_runner_frames_scrubbed_log_file_raw(monkeypatch, tmp_path):
    monkeypatch.delenv("HA_AGENT_HOME", raising=False)
    events = [
        {"type": "log", "text": f"$ cd {_OVERLAY}/backend/arch && grep -rh quant"},
        {"type": "done", "success": False,
         "message": f"validate failed: {_OVERLAY}/models/x.yaml: bad dim"},
    ]
    parsed, raw_log = _run_frames(monkeypatch, tmp_path, events, _OVERLAY)

    streamed = "\n".join(
        e.get("text", e.get("message", "")) for e in parsed)
    assert _OVERLAY not in streamed
    assert "07a6e2a7" not in streamed
    assert "$ cd backend/arch && grep -rh quant" in streamed
    assert "validate failed: models/x.yaml: bad dim" in streamed

    # The on-disk trace log is the operator's debug artifact — raw paths kept.
    assert f"$ cd {_OVERLAY}/backend/arch" in raw_log


def test_upload_run_log_scrubs_prompt_echo(monkeypatch, tmp_path):
    from backend.spec_worker import worker as W
    from backend.service import spec_generation_logs as L
    monkeypatch.delenv("HA_AGENT_HOME", raising=False)
    logp = tmp_path / "run.log"
    logp.write_text(
        f"=== PROMPT ===\nwrite to {_OVERLAY}/models/<name>.yaml\n"
        f"[log] $ cd {_OVERLAY}/backend/arch && ls\n"
    )
    calls = []
    monkeypatch.setattr("backend.supabase_env.configured", lambda: True)
    monkeypatch.setattr(L, "record", lambda **kw: calls.append(kw) or True)

    W._upload_run_log(
        "zai-org/GLM-5.3-Flash",
        {"user_id": "u1", "target_root": _OVERLAY},
        "u1|zai-org/GLM-5.3-Flash", str(logp), True,
    )
    assert len(calls) == 1
    uploaded = calls[0]["log"]
    assert _OVERLAY not in uploaded
    assert "07a6e2a7" not in uploaded
    assert "write to models/<name>.yaml" in uploaded
    assert "$ cd backend/arch && ls" in uploaded


def test_runner_starts_backends_that_cannot_enforce_path_policy(monkeypatch, tmp_path):
    """Codex/Kimi do not set supports_isolated_workspace; a target_root must
    still spawn, with that leaf passed through as workspace."""
    spawned = {}

    class _Unscoped(_FakeBackend):
        supports_isolated_workspace = False
        display_name = "Codex"

        def spawn(self, prompt, env=None, workspace=None):
            spawned["workspace"] = workspace
            return super().spawn(prompt, env=env, workspace=workspace)

    leaf = str(tmp_path / "runs" / "__anonymous__" / "generate__m")
    monkeypatch.setattr(R, "get_backend", lambda cli: _Unscoped([
        {"type": "done", "success": True},
    ]))
    list(R.stream_agent_core(
        key="m", prep_result={"prompt": "p", "target_root": leaf},
        agent_cli="codex", log_path=str(tmp_path / "log.txt"), log_header="h",
        launch_banner="B", procs_dict={}, locks_lock=threading.Lock(),
        release_fn=lambda k: None,
    ))
    assert spawned["workspace"] == leaf


def test_job_buffer_scrubs_non_runner_yields(monkeypatch):
    """Pre-prep failures never pass through stream_agent_core, so the job
    buffer applies the generic (repo/home) scrub itself."""
    from backend.spec_worker import runtime as RT
    monkeypatch.delenv("HA_AGENT_HOME", raising=False)
    monkeypatch.setenv("HOME", "/home/deploy")

    def _run_fn():
        yield {"type": "log", "text": "Starting analysis..."}
        yield {"type": "done", "success": False,
               "message": "Metadata fetch failed: /home/deploy/.cache/hf lock"}

    key = "test/scrub-buffer"
    RT.start_job(key, _run_fn)
    try:
        for _ in range(200):
            status = RT.job_status(key, 0)
            if status["state"] == "done":
                break
            time.sleep(0.02)
        assert status["state"] == "done"
        assert status["success"] is False
        assert status["message"] == "Metadata fetch failed: ~/.cache/hf lock"
        assert "/home/deploy" not in json.dumps(status["events"])
    finally:
        with RT._locks_lock:
            RT._jobs.pop(key, None)
