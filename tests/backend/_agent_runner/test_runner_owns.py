"""stream_agent_core(owns=...): a superseded run starts no agent.

A spec run whose lock was reclaimed while it was stalled can resume later.
The runner asks `owns(key)` before spawning and again while registering the
process; a run that no longer holds the key must not leave an agent running
under a key another run now owns.
"""
from __future__ import annotations

import json
import threading

import backend._agent_runner.runner as R
from tests.backend._agent_runner.test_agent_runner_scrub import _FakeBackend


class _CountingBackend(_FakeBackend):
    def __init__(self, events, on_spawn=None):
        super().__init__(events)
        self.spawned = []
        self._on_spawn = on_spawn

    def spawn(self, prompt, env=None, workspace=None):
        proc = super().spawn(prompt, env=env, workspace=workspace)
        proc.returncode = None           # still running until killed
        self.spawned.append(proc)
        if self._on_spawn:
            self._on_spawn()
        return proc


def _run(monkeypatch, tmp_path, backend, owns):
    monkeypatch.setattr(R, "get_backend", lambda cli: backend)
    procs = {}
    frames = list(R.stream_agent_core(
        key="m", prep_result={"prompt": "p", "target_root": str(tmp_path)},
        agent_cli="fake", log_path=str(tmp_path / "log.txt"), log_header="h",
        launch_banner="B", procs_dict=procs, locks_lock=threading.Lock(),
        release_fn=lambda k: None, owns=owns,
    ))
    events = [json.loads(f[len("data: "):].strip()) for f in frames if f.startswith("data: ")]
    return events, procs


def test_a_run_that_no_longer_owns_the_key_spawns_nothing(monkeypatch, tmp_path):
    backend = _CountingBackend([])
    events, procs = _run(monkeypatch, tmp_path, backend, owns=lambda k: False)
    assert backend.spawned == [] and procs == {}
    done = [e for e in events if e.get("type") == "done"]
    assert done and done[-1]["success"] is False and "superseded" in done[-1]["message"]


def test_superseded_while_spawning_kills_the_agent_and_does_not_register(monkeypatch, tmp_path):
    state = {"owner": True}
    backend = _CountingBackend([], on_spawn=lambda: state.update(owner=False))
    events, procs = _run(monkeypatch, tmp_path, backend, owns=lambda k: state["owner"])
    assert len(backend.spawned) == 1 and backend.spawned[0].returncode == -9   # killed
    assert procs == {}
    assert events[-1]["type"] == "done" and events[-1]["success"] is False


class _RecordingDict(dict):
    def __init__(self):
        super().__init__()
        self.registered = []

    def __setitem__(self, key, value):
        self.registered.append((key, value))
        super().__setitem__(key, value)


def test_without_owns_the_runner_behaves_as_before(monkeypatch, tmp_path):
    backend = _CountingBackend([])
    monkeypatch.setattr(R, "get_backend", lambda cli: backend)
    procs = _RecordingDict()
    list(R.stream_agent_core(
        key="m", prep_result={"prompt": "p", "target_root": str(tmp_path)},
        agent_cli="fake", log_path=str(tmp_path / "log.txt"), log_header="h",
        launch_banner="B", procs_dict=procs, locks_lock=threading.Lock(),
        release_fn=lambda k: None, owns=None,
    ))
    assert len(backend.spawned) == 1
    assert procs.registered == [("m", backend.spawned[0])]   # registered while it ran


def test_a_repair_round_starts_only_after_the_last_agent_is_reaped(monkeypatch, tmp_path):
    """A downgrade returns mid-stream while the CLI is still up; the next
    round must not spawn beside it into the same workspace."""
    backend = _CountingBackend([{"type": "done", "success": True}])
    alive_at_spawn = []
    backend._on_spawn = lambda: alive_at_spawn.append(
        [p.returncode is None for p in backend.spawned[:-1]])
    verdicts = iter(["spec invalid", None])
    monkeypatch.setattr(R, "get_backend", lambda cli: backend)
    procs = {}
    list(R.stream_agent_core(
        key="m", prep_result={"prompt": "p", "target_root": str(tmp_path)},
        agent_cli="fake", log_path=str(tmp_path / "log.txt"), log_header="h",
        launch_banner="B", procs_dict=procs, locks_lock=threading.Lock(),
        release_fn=lambda k: None,
        on_success=lambda *a: next(verdicts),
        repair_prompt_fn=lambda reason, prep: "fix it", max_repair_rounds=1,
    ))
    assert len(backend.spawned) == 2
    assert alive_at_spawn == [[], [False]]          # round 1's agent was gone
    assert backend.spawned[0].returncode == -9
    assert procs == {}                               # no dead handle left for stop()
