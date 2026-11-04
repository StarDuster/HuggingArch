"""A clean agent exit is not proof of success.

The agent CLIs report success on a plain exit-0. When the agent writes NO
spec (e.g. codex's Bubblewrap sandbox fails to launch, so every command errors
and it exits 0 having done nothing), the run must NOT surface as success. The
`on_success` hook returns a downgrade reason and `stream_agent_core` rewrites
the emitted `done` event to success=False — covered here for both success
paths (agent-emitted done, and synthesized-from-exit-0), plus the spec_worker
verdict function itself.
"""
from __future__ import annotations

import io
import json
import threading

import pytest

from backend._agent_runner import runner as R
from backend.spec_worker import worker as W


def _overlay_with_identity(tmp_path, model_id: str = "meta/x") -> str:
    """A run overlay holding a fresh `models/<safe(model_id)>.yaml` whose
    identity names `model_id` — the only spec the verdict gate will judge."""
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    stem = model_id.replace("/", "__")
    (models / f"{stem}.yaml").write_text(
        f"match: {{model_id: [{model_id}]}}\n", encoding="utf-8")
    return str(tmp_path)


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


class _FakeBackend:
    supports_resume = False

    def __init__(self, events, stdout_lines, returncode=0):
        self._events, self._lines, self._rc = events, stdout_lines, returncode

    def spawn(self, prompt, env=None):
        return _FakeProc(self._lines, self._rc)

    def extract_session_id(self, evt):
        return None

    def parse_event(self, evt):
        return self._events


def _final_done(monkeypatch, tmp_path, backend, on_success):
    monkeypatch.setattr(R, "get_backend", lambda cli: backend)
    frames = list(R.stream_agent_core(
        key="m", prep_result={"prompt": "p"}, agent_cli="fake",
        log_path=str(tmp_path / "log.txt"), log_header="d_model", launch_banner="B",
        procs_dict={}, locks_lock=threading.Lock(), release_fn=lambda k: None,
        on_success=on_success,
    ))
    dones = [json.loads(f[len("data: "):].strip())
             for f in frames if f.startswith("data: ")]
    dones = [d for d in dones if d.get("type") == "done"]
    assert dones, "no done event emitted"
    return dones[-1]


def test_agent_done_success_downgraded_when_hook_reports_no_spec(monkeypatch, tmp_path):
    backend = _FakeBackend(events=[{"type": "done", "success": True}],
                           stdout_lines=['{"raw": 1}\n'])
    done = _final_done(monkeypatch, tmp_path, backend,
                       on_success=lambda *a: "wrote no model spec")
    assert done["success"] is False
    assert "no model spec" in done["message"]


def test_agent_done_success_kept_when_hook_returns_none(monkeypatch, tmp_path):
    backend = _FakeBackend(events=[{"type": "done", "success": True}],
                           stdout_lines=['{"raw": 1}\n'])
    done = _final_done(monkeypatch, tmp_path, backend, on_success=lambda *a: None)
    assert done["success"] is True


def test_exit0_without_done_downgraded_when_hook_reports_no_spec(monkeypatch, tmp_path):
    # No stdout lines → no agent `done`; proc exits 0 → runner synthesizes one.
    backend = _FakeBackend(events=[], stdout_lines=[], returncode=0)
    done = _final_done(monkeypatch, tmp_path, backend,
                       on_success=lambda *a: "wrote no model spec")
    assert done["success"] is False
    assert "no model spec" in done["message"]


def test_verify_spec_output_flags_missing_yaml(monkeypatch, tmp_path):
    log = io.StringIO()
    # No fresh yaml → downgrade reason, no validate result.
    empty = tmp_path / "empty"
    (empty / "models").mkdir(parents=True)
    monkeypatch.setattr("backend.spec_worker._op_chain.find_fresh_gap",
                        lambda model_id, started_at: None)
    downgrade, val_result = W._verify_spec_output(
        "meta/x", log, started_at=0.0, root=str(empty))
    assert downgrade.kind == "no_output" and "no model spec" in downgrade.reason
    assert val_result is None

    # Fresh, valid yaml present → success kept (None), result handed to publish.
    root = _overlay_with_identity(tmp_path)
    monkeypatch.setattr(W, "_normalize_overlay_bundle", lambda *a, **k: None)
    monkeypatch.setattr(
        W, "_invoke_validate",
        lambda model_id, spec_name, overlay_dir=None, **kw: {"errors": []},
    )
    downgrade, val_result = W._verify_spec_output("meta/x", log, started_at=0.0, root=root)
    assert downgrade is None
    assert val_result == {"errors": []}

    # Any hard validator error downgrades the terminal SSE verdict.
    monkeypatch.setattr(
        W, "_invoke_validate",
        lambda model_id, spec_name, overlay_dir=None, **kw: {
            "errors": ["dim_map_mismatch in [block] delta_rule"],
        },
    )
    downgrade, val_result = W._verify_spec_output("meta/x", log, started_at=0.0, root=root)
    assert downgrade.kind == "validation" and "failed validation" in downgrade.reason
    assert downgrade.errors == ("dim_map_mismatch in [block] delta_rule",)
    assert val_result is None


def test_verify_spec_output_downgrades_when_validate_crashes(monkeypatch, tmp_path):
    # A raise out of `validate` is infrastructure (config / Hub), not a
    # spec-eval finding — those arrive as `errors[]`. The worker still
    # records the raise as a crash so it does not leak into the generator.
    log = io.StringIO()
    root = _overlay_with_identity(tmp_path)
    monkeypatch.setattr(W, "_normalize_overlay_bundle", lambda *a, **k: None)

    def _boom(model_id, spec_name, overlay_dir=None, **kw):
        raise OSError("hub down")

    monkeypatch.setattr(W, "_invoke_validate", _boom)
    downgrade, val_result = W._verify_spec_output("meta/x", log, started_at=0.0, root=root)
    assert downgrade.kind == "crash" and "validation crashed" in downgrade.reason
    assert "OSError" in downgrade.reason
    assert val_result is None


def test_a_spec_eval_exception_is_a_repairable_validation_finding(monkeypatch, tmp_path):
    """Broadcast / KeyError during projection used to skip repair. `validate`
    now puts them in `errors[]`; the worker hands that list to the writer."""
    log = io.StringIO()
    root = _overlay_with_identity(tmp_path)
    monkeypatch.setattr(W, "_normalize_overlay_bundle", lambda *a, **k: None)
    monkeypatch.setattr(
        W, "_invoke_validate",
        lambda model_id, spec_name, overlay_dir=None, **kw: {
            "valid": False,
            "errors": [
                "ValueError: elementwise broadcast axis 3 has incompatible "
                "extents [2560, 4]",
            ],
        },
    )
    downgrade, val_result = W._verify_spec_output(
        "meta/x", log, started_at=0.0, root=root)
    assert downgrade.kind == "validation"
    assert any("broadcast" in e for e in downgrade.errors)
    assert val_result is None
    prompt = W._repair_prompt_fn("meta/x")(downgrade.reason, {"downgrade": downgrade})
    assert prompt and "broadcast" in prompt and "[2560, 4]" in prompt


def test_a_spec_that_fails_to_load_is_a_validation_finding_not_a_crash(monkeypatch, tmp_path):
    """The writer's contract: whatever is wrong with the spec it just
    wrote — a field that does not parse, an op the registry does not know,
    a component slot that resolves nowhere — comes back as the error list
    the repair round hands the agent. The loader refusing to SERVE a broken
    spec is one job; `validate` turning that refusal into a finding is the
    other, and the worker never sees an exception for it (a crash is a
    validator bug, and only that is unrepairable)."""
    from backend.arch.loader import SpecLoadError
    from backend.arch.validate import _orchestrator as O

    monkeypatch.setattr(O, "_get_safetensors_data", lambda model_id, overlay_dir=None: {
        "total_params": 0, "model_storage_bytes": 0, "header": None,
    })
    monkeypatch.setattr(O, "_fetch_config", lambda model_id, overlay_dir=None: {"model_type": "x"})

    def _refuse(*args, **kwargs):
        raise SpecLoadError("spec 'x' exists at /overlay/models/x.yaml but failed to load: "
                            "unknown op 'frobnicate' at graph node 'mixer'", path="/overlay/models/x.yaml")

    monkeypatch.setattr(O, "_resolve_spec", _refuse)
    report = O.validate("meta/x", "x")
    assert report["valid"] is False
    assert any("frobnicate" in e for e in report["errors"])

    log = io.StringIO()
    root = _overlay_with_identity(tmp_path)
    monkeypatch.setattr(W, "_normalize_overlay_bundle", lambda *a, **k: None)
    monkeypatch.setattr(W, "_invoke_validate",
                        lambda model_id, spec_name, overlay_dir=None, **kw: report)
    downgrade, _ = W._verify_spec_output("meta/x", log, started_at=0.0, root=root)
    assert downgrade.kind == "validation"
    assert any("frobnicate" in e for e in downgrade.errors)
    prompt = W._repair_prompt_fn("meta/x")(downgrade.reason, {"downgrade": downgrade})
    assert prompt and "frobnicate" in prompt


def test_invoke_validate_uses_the_validate_cli(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["kw"] = kw

        class P:
            stdout = '{"errors": [], "valid": true}'
            stderr = ""
            returncode = 0
        return P()

    monkeypatch.setattr(W.subprocess, "run", fake_run)
    out = W._invoke_validate("org/m", "org__m", "/tmp/ov")
    assert out == {"errors": [], "valid": True}
    assert seen["cmd"][1:4] == ["-m", "backend.inference.validate", "org/m"]
    assert "--spec" in seen["cmd"] and "org__m" in seen["cmd"]
    assert "--overlay-dir" in seen["cmd"] and "/tmp/ov" in seen["cmd"]
    assert seen["kw"]["env"]["HA_SPEC_OVERLAY_DIR"] == "/tmp/ov"


def test_verify_spec_output_migrate_blocker_is_validation(monkeypatch, tmp_path):
    from backend.arch.loader import SpecLoadError

    log = io.StringIO()
    root = _overlay_with_identity(tmp_path)
    monkeypatch.setattr(
        W, "_normalize_overlay_bundle",
        lambda *a, **k: (_ for _ in ()).throw(
            SpecLoadError("cannot be migrated automatically: retired kv")))
    downgrade, val_result = W._verify_spec_output(
        "meta/x", log, started_at=0.0, root=root)
    assert downgrade.kind == "validation"
    assert downgrade.error_kind == "spec_invalid"
    assert val_result is None
    prompt = W._repair_prompt_fn("meta/x")(downgrade.reason, {"downgrade": downgrade})
    assert prompt and "retired kv" in prompt


def test_config_unavailable_does_not_generate_a_repair_prompt(monkeypatch, tmp_path):
    """Hub / config failure is infrastructure. Worker records crash and
    must not spend a paid repair round asking the writer to fix the spec."""
    from backend.arch.validate import _orchestrator as O

    monkeypatch.setattr(
        O, "_get_safetensors_data",
        lambda model_id, overlay_dir=None: {
            "total_params": 0, "model_storage_bytes": 0,
            "header": None,
        },
    )

    def _no_config(model_id, overlay_dir=None):
        raise O.ConfigUnavailable(
            f"Cannot fetch config for {model_id}: OSError: hub down")

    monkeypatch.setattr(O, "_fetch_config", _no_config)
    try:
        O.validate("meta/x")
        raised = None
    except O.ConfigUnavailable as exc:
        raised = exc
    assert raised is not None
    assert "hub down" in str(raised)

    log = io.StringIO()
    root = _overlay_with_identity(tmp_path)
    monkeypatch.setattr(W, "_normalize_overlay_bundle", lambda *a, **k: None)
    monkeypatch.setattr(
        W, "_invoke_validate",
        lambda *a, **k: (_ for _ in ()).throw(raised))
    downgrade, val_result = W._verify_spec_output(
        "meta/x", log, started_at=0.0, root=root)
    assert downgrade.kind == "crash"
    assert val_result is None
    prompt = W._repair_prompt_fn("meta/x")(
        downgrade.reason, {"downgrade": downgrade})
    assert prompt is None


def test_empty_checkpoint_metadata_is_not_a_writer_keyerror(monkeypatch, tmp_path):
    """fetch_metadata returning {} used to KeyError on model_storage_bytes
    and hand that to the writer as a repairable finding."""
    from backend.arch.validate import _orchestrator as O

    monkeypatch.setattr("backend.hf.metadata.fetch_metadata", lambda model_id, overlay_dir=None: {})
    monkeypatch.setattr(O, "_fetch_config", lambda model_id, overlay_dir=None: {"model_type": "x"})
    try:
        report = O.validate("org/m")
    except O.CheckpointUnavailable as exc:
        report = None
        infra = exc
    else:
        joined = "; ".join(str(e) for e in report.get("errors") or [])
        assert "model_storage_bytes" not in joined
        assert "KeyError" not in joined
        infra = RuntimeError("empty checkpoint metadata")

    log = io.StringIO()
    root = _overlay_with_identity(tmp_path, "org/m")
    monkeypatch.setattr(W, "_normalize_overlay_bundle", lambda *a, **k: None)
    monkeypatch.setattr(
        W, "_invoke_validate",
        lambda *a, **k: (_ for _ in ()).throw(infra))
    downgrade, _ = W._verify_spec_output("org/m", log, started_at=0.0, root=root)
    assert downgrade.kind != "validation"
    prompt = W._repair_prompt_fn("org/m")(
        downgrade.reason, {"downgrade": downgrade})
    assert prompt is None


def test_agent_timeout_reports_actionable_message(monkeypatch, tmp_path):
    class _ImmediateTimer:
        def __init__(self, _seconds, fn):
            self.fn = fn
            self.daemon = False

        def start(self):
            self.fn()

        def cancel(self):
            pass

    backend = _FakeBackend(events=[], stdout_lines=[], returncode=None)
    monkeypatch.setattr(R.threading, "Timer", _ImmediateTimer)
    done = _final_done_with_timeout(monkeypatch, tmp_path, backend)
    assert done == {
        "type": "done",
        "success": False,
        "message": "Agent execution exceeded 1 hour and was stopped",
        "timed_out": True,
        "agent_end": "timed_out",
    }


def test_agent_timeout_names_the_knob(monkeypatch, tmp_path):
    class _ImmediateTimer:
        def __init__(self, _seconds, fn):
            self.fn = fn
            self.daemon = False

        def start(self):
            self.fn()

        def cancel(self):
            pass

    backend = _FakeBackend(events=[], stdout_lines=[], returncode=None)
    monkeypatch.setattr(R, "get_backend", lambda cli: backend)
    monkeypatch.setattr(R.threading, "Timer", _ImmediateTimer)
    frames = list(R.stream_agent_core(
        key="m", prep_result={"prompt": "p"}, agent_cli="fake",
        log_path=str(tmp_path / "timeout.log"), log_header="d_model",
        launch_banner="B", procs_dict={}, locks_lock=threading.Lock(),
        release_fn=lambda k: None, hard_cap_sec=1800,
        timeout_hint="HA_SPEC_FIX_TIMEOUT_SECONDS",
    ))
    done = next(
        event
        for event in (
            json.loads(frame[len("data: "):].strip())
            for frame in frames if frame.startswith("data: ")
        )
        if event.get("type") == "done"
    )
    assert "30 minute" in done["message"]
    assert "HA_SPEC_FIX_TIMEOUT_SECONDS" in done["message"]


def _final_done_with_timeout(monkeypatch, tmp_path, backend):
    monkeypatch.setattr(R, "get_backend", lambda cli: backend)
    frames = list(R.stream_agent_core(
        key="m", prep_result={"prompt": "p"}, agent_cli="fake",
        log_path=str(tmp_path / "timeout.log"), log_header="d_model",
        launch_banner="B", procs_dict={}, locks_lock=threading.Lock(),
        release_fn=lambda k: None, hard_cap_sec=3600,
    ))
    return next(
        event
        for event in (
            json.loads(frame[len("data: "):].strip())
            for frame in frames if frame.startswith("data: ")
        )
        if event.get("type") == "done"
    )


# ---------------------------------------------------------------------------
# Repair rounds — a verdict-gate downgrade is fed back to the agent instead
# of ending the run (spec_worker enables this for validation failures).
# ---------------------------------------------------------------------------
class _RecordingBackend(_FakeBackend):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.prompts = []

    def spawn(self, prompt, env=None):
        self.prompts.append(prompt)
        return _FakeProc(self._lines, self._rc)


def _run_frames(monkeypatch, tmp_path, backend, **kw):
    monkeypatch.setattr(R, "get_backend", lambda cli: backend)
    frames = list(R.stream_agent_core(
        key="m", prep_result={"prompt": "p0"}, agent_cli="fake",
        log_path=str(tmp_path / "log.txt"), log_header="h", launch_banner="B",
        procs_dict={}, locks_lock=threading.Lock(), release_fn=lambda k: None,
        **kw,
    ))
    return [json.loads(f[len("data: "):].strip())
            for f in frames if f.startswith("data: ")]


def test_repair_round_respawns_agent_with_errors(monkeypatch, tmp_path):
    """A downgrade + a repair prompt → the agent is re-spawned with the
    errors; the intermediate failure never reaches the client, and the
    repaired round's success verdict is the one emitted."""
    backend = _RecordingBackend(events=[{"type": "done", "success": True}],
                                stdout_lines=['{"raw": 1}\n'])
    verdicts = iter(["generated spec failed validation: bad dim", None])
    events = _run_frames(
        monkeypatch, tmp_path, backend,
        on_success=lambda *a: next(verdicts),
        repair_prompt_fn=lambda reason, prep: f"FIX: {reason}",
    )
    dones = [e for e in events if e.get("type") == "done"]
    assert dones == [{"type": "done", "success": True}]
    assert backend.prompts == [
        "p0", "FIX: generated spec failed validation: bad dim"]
    # The client sees a repair notice between rounds, not a failure.
    assert any("repair round 1/2" in (e.get("text") or "")
               for e in events if e.get("type") == "log")


def test_repair_rounds_exhausted_emits_failure(monkeypatch, tmp_path):
    """Out of repair rounds → the failure done is emitted, with the verdict
    reason intact."""
    backend = _RecordingBackend(events=[{"type": "done", "success": True}],
                                stdout_lines=['{"raw": 1}\n'])
    events = _run_frames(
        monkeypatch, tmp_path, backend,
        on_success=lambda *a: "generated spec failed validation: still bad",
        repair_prompt_fn=lambda reason, prep: "FIX",
        max_repair_rounds=1,
    )
    dones = [e for e in events if e.get("type") == "done"]
    assert len(dones) == 1
    assert dones[0]["success"] is False
    assert "still bad" in dones[0]["message"]
    assert backend.prompts == ["p0", "FIX"]


def test_no_repair_fn_keeps_single_attempt(monkeypatch, tmp_path):
    """Default wiring (no repair_prompt_fn) is unchanged: one round, the
    downgrade surfaces as the failure message."""
    backend = _RecordingBackend(events=[{"type": "done", "success": True}],
                                stdout_lines=['{"raw": 1}\n'])
    events = _run_frames(
        monkeypatch, tmp_path, backend,
        on_success=lambda *a: "wrote no model spec",
    )
    dones = [e for e in events if e.get("type") == "done"]
    assert dones == [
        {"type": "done", "success": False, "message": "wrote no model spec"}]
    assert backend.prompts == ["p0"]


def test_worker_repair_hook_only_covers_validation_failures():
    make = W._repair_prompt_fn("meta/x")
    no_output = W.Downgrade("no_output", "agent exited cleanly but wrote no model spec")
    crash = W.Downgrade("crash", "generated spec validation crashed: KeyError")
    validation = W.Downgrade("validation", "generated spec failed validation: bad dim", ("bad dim",))
    assert make(no_output.reason, {"downgrade": no_output}) is None
    assert make(crash.reason, {"downgrade": crash}) is None
    prompt = make(validation.reason, {"downgrade": validation})
    assert prompt and "bad dim" in prompt and "meta/x" in prompt


def test_repair_round_resumes_same_session_when_supported(monkeypatch, tmp_path):
    """With a session-capable backend, a repair round re-enters the SAME
    conversation (spawn_resume with the captured session id) instead of
    cold-starting on the repair prompt."""
    class Resumable(_RecordingBackend):
        supports_resume = True

        def extract_session_id(self, evt):
            return "sess-1"

        def spawn_resume(self, session_id, prompt, env=None, workspace=None):
            self.prompts.append(f"resume:{session_id}:{prompt}")
            return _FakeProc(self._lines, self._rc)

    backend = Resumable(events=[{"type": "done", "success": True}],
                        stdout_lines=['{"raw": 1}\n'])
    verdicts = iter(["generated spec failed validation: bad dim", None])
    events = _run_frames(
        monkeypatch, tmp_path, backend,
        on_success=lambda *a: next(verdicts),
        repair_prompt_fn=lambda reason, prep: f"FIX: {reason}",
    )
    dones = [e for e in events if e.get("type") == "done"]
    assert dones == [{"type": "done", "success": True}]
    assert backend.prompts == [
        "p0", "resume:sess-1:FIX: generated spec failed validation: bad dim"]


def test_repair_rounds_share_one_wall_clock_cap(monkeypatch, tmp_path):
    """Each repair round inherits remaining time; it does not reset the cap."""
    armed = []

    class CaptureTimer:
        def __init__(self, seconds, fn):
            armed.append(seconds)
            self.fn = fn
            self.daemon = False

        def start(self):
            pass

        def cancel(self):
            pass

    clock = {"t": 1000.0}
    monkeypatch.setattr(R.time, "time", lambda: clock["t"])
    monkeypatch.setattr(R.threading, "Timer", CaptureTimer)
    backend = _RecordingBackend(events=[{"type": "done", "success": True}],
                                stdout_lines=['{"raw": 1}\n'])

    def on_success(*_a):
        if len(armed) == 1:
            clock["t"] += 40.0
            return "generated spec failed validation: bad dim"
        return None

    events = _run_frames(
        monkeypatch, tmp_path, backend,
        on_success=on_success,
        repair_prompt_fn=lambda reason, prep: "FIX",
        hard_cap_sec=60,
    )
    assert armed == [60, 20]
    dones = [e for e in events if e.get("type") == "done"]
    assert dones == [{"type": "done", "success": True}]


def test_continue_stream_reuses_the_run_deadline(monkeypatch, tmp_path):
    """Op-continue under the same lock consumes leftover time, not a new cap."""
    armed = []

    class CaptureTimer:
        def __init__(self, seconds, fn):
            armed.append(seconds)
            self.fn = fn
            self.daemon = False

        def start(self):
            pass

        def cancel(self):
            pass

    clock = {"t": 1000.0}
    monkeypatch.setattr(R.time, "time", lambda: clock["t"])
    monkeypatch.setattr(R.threading, "Timer", CaptureTimer)
    monkeypatch.setattr(R, "get_backend", lambda cli: _RecordingBackend(
        events=[{"type": "done", "success": True}],
        stdout_lines=['{"raw": 1}\n']))
    prep = {"prompt": "p0"}
    list(R.stream_agent_core(
        key="m", prep_result=prep, agent_cli="fake",
        log_path=str(tmp_path / "first.log"), log_header="h", launch_banner="B",
        procs_dict={}, locks_lock=threading.Lock(), release_fn=lambda k: None,
        hard_cap_sec=60,
    ))
    assert armed == [60]
    assert prep["_run_deadline_at"] == 1060.0
    clock["t"] += 40.0
    armed.clear()
    monkeypatch.setattr(R, "get_backend", lambda cli: _RecordingBackend(
        events=[{"type": "done", "success": True}],
        stdout_lines=['{"raw": 1}\n']))
    list(R.stream_agent_core(
        key="m", prep_result={"prompt": "p1"}, agent_cli="fake",
        log_path=str(tmp_path / "cont.log"), log_header="h", launch_banner="B",
        procs_dict={}, locks_lock=threading.Lock(), release_fn=lambda k: None,
        hard_cap_sec=60, deadline_at=prep["_run_deadline_at"],
    ))
    assert armed == [20]


def test_each_round_gets_its_own_timer_inside_the_run_deadline(monkeypatch, tmp_path):
    """A round is stopped at min(round start + round budget, run deadline):
    a repair round gets a full round again while the deadline allows, and the
    last one only what is left."""
    armed = []

    class CaptureTimer:
        def __init__(self, seconds, fn):
            armed.append(seconds)
            self.fn = fn
            self.daemon = False

        def start(self):
            pass

        def cancel(self):
            pass

    clock = {"t": 1000.0}
    monkeypatch.setattr(R.time, "time", lambda: clock["t"])
    monkeypatch.setattr(R.threading, "Timer", CaptureTimer)
    backend = _RecordingBackend(events=[{"type": "done", "success": True}],
                                stdout_lines=['{"raw": 1}\n'])
    spent = [25.0, 60.0]

    def on_success(*_a):
        if spent:
            clock["t"] += spent.pop(0)
            return "generated spec failed validation: bad dim"
        return None

    events = _run_frames(
        monkeypatch, tmp_path, backend,
        on_success=on_success,
        repair_prompt_fn=lambda reason, prep: "FIX",
        max_repair_rounds=2,
        hard_cap_sec=100, round_budget_sec=30,
    )
    # Round 1 at t=0: 30 of 100. Round 2 at t=25: a full 30 (75 left).
    # Round 3 at t=85: the 15 the run has left.
    assert armed == [30, 30, 15]
    assert [e for e in events if e.get("type") == "done"] == [
        {"type": "done", "success": True}]


def test_a_round_timeout_names_the_round_knob(monkeypatch, tmp_path):
    class _ImmediateTimer:
        def __init__(self, _seconds, fn):
            self.fn = fn
            self.daemon = False

        def start(self):
            self.fn()

        def cancel(self):
            pass

    backend = _FakeBackend(events=[], stdout_lines=[], returncode=None)
    monkeypatch.setattr(R.threading, "Timer", _ImmediateTimer)
    events = _run_frames(
        monkeypatch, tmp_path, backend, hard_cap_sec=7200,
        round_budget_sec=600, timeout_hint="SPEC_AGENT_RUN_TIMEOUT_SECONDS",
        round_timeout_hint="HA_SPEC_ROUND_TIMEOUT_SECONDS")
    done = [e for e in events if e.get("type") == "done"][-1]
    assert done["timed_out"] is True
    assert done["message"] == (
        "An agent round exceeded its budget of 10 minutes and was stopped. "
        "Retry, or raise HA_SPEC_ROUND_TIMEOUT_SECONDS")


def test_extract_session_id_per_backend():
    """Each resume-capable backend pulls its session handle out of its own
    event shape (claude: system/init, codex: thread.started, kimi:
    session.resume_hint)."""
    from backend.agents.claude import ClaudeBackend
    from backend.agents.codex import CodexBackend
    from backend.agents.kimi import KimiBackend

    claude = ClaudeBackend()
    assert claude.extract_session_id(
        {"type": "system", "subtype": "init", "session_id": "c1"}) == "c1"
    assert claude.extract_session_id({"type": "assistant"}) is None

    codex = CodexBackend()
    assert codex.extract_session_id(
        {"type": "thread.started", "thread_id": "x1"}) == "x1"
    assert codex.extract_session_id({"type": "turn.started"}) is None

    kimi = KimiBackend()
    assert kimi.extract_session_id(
        {"role": "meta", "type": "session.resume_hint",
         "session_id": "k1"}) == "k1"
    assert kimi.extract_session_id({"role": "assistant"}) is None


_ENGINE = "analysis engine refusal at node `lm_head`: gather extent"
_SPEC = "analysis refuses node `mask`: no layout transition rule. Fix `op`: use slice"
_OPEN = "analysis refuses node `w`, raised at x.py:1 (f) on tp8 (decode): boom"

_FINDINGS = [
    {"kind": "analysis_engine_refusal", "detail": _ENGINE, "node": "lm_head",
     "site": "closure.py:212", "exception": "SolClosureError",
     "deployments": ["ep8_dpattn", "tp8"], "stages": ["decode"],
     "capability": "gather extent"},
    {"kind": "analysis_refusal", "detail": _SPEC, "node": "mask",
     "site": "layout dispatch", "exception": "SpecRefusal",
     "deployments": ["every multi-GPU deployment"], "stages": ["prefill"],
     "declaration": "op", "fix": "use slice"},
    {"kind": "analysis_refusal_unattributed", "detail": _OPEN, "node": "w",
     "site": "x.py:1", "exception": "ValueError",
     "deployments": ["tp8"], "stages": ["decode"]},
]


def _analysis_report(errors, blocked):
    return {"valid": False, "errors": list(errors), "blocked_by_engine": blocked,
            "analysis_refusals": _FINDINGS}


def _verdict(monkeypatch, tmp_path, report):
    root = _overlay_with_identity(tmp_path)
    monkeypatch.setattr(W, "_normalize_overlay_bundle", lambda *a, **k: None)
    monkeypatch.setattr(W, "_invoke_validate", lambda *a, **k: report)
    downgrade, _ = W._verify_spec_output("meta/x", io.StringIO(), started_at=0.0, root=root)
    return downgrade


def test_only_engine_refusals_block_by_engine_without_repair(monkeypatch, tmp_path):
    """No spec edit resolves an engine refusal: the run is blocked, its
    verdict carries the gaps as data, and the repair hook hands the agent
    nothing."""
    downgrade = _verdict(monkeypatch, tmp_path, _analysis_report([_ENGINE], True))
    assert downgrade.kind == "blocked_by_engine"
    assert downgrade.error_kind == "blocked_by_engine"
    (gap,) = downgrade.engine_gaps
    assert gap == {"capability": "gather extent", "node": "lm_head",
                   "site": "closure.py:212", "exception": "SolClosureError",
                   "deployments": ["ep8_dpattn", "tp8"], "stages": ["decode"],
                   "detail": _ENGINE}
    assert "does not yet support: gather extent" in downgrade.reason
    assert W._repair_prompt_fn("meta/x")(downgrade.reason, {"downgrade": downgrade}) is None


def test_mixed_refusals_repair_only_what_the_spec_resolves(monkeypatch, tmp_path):
    downgrade = _verdict(monkeypatch, tmp_path, _analysis_report([_ENGINE, _SPEC], False))
    assert downgrade.kind == "validation"
    assert downgrade.errors == (_SPEC,)
    assert downgrade.error_kind is None
    assert _ENGINE in downgrade.reason


def test_only_unattributed_refusals_left_is_its_own_error_kind(monkeypatch, tmp_path):
    """An unattributed refusal stays with the writer; a run left with only
    those (engine refusals aside) is classified so Admin can list it."""
    downgrade = _verdict(monkeypatch, tmp_path, _analysis_report([_OPEN, _ENGINE], False))
    assert downgrade.kind == "validation"
    assert downgrade.errors == (_OPEN,)
    assert downgrade.error_kind == "analysis_refusal_unattributed"
    assert _OPEN in W._repair_prompt_fn("meta/x")(downgrade.reason, {"downgrade": downgrade})


def test_unattributed_beside_a_spec_error_is_unclassified(monkeypatch, tmp_path):
    downgrade = _verdict(monkeypatch, tmp_path, _analysis_report([_OPEN, _SPEC], False))
    assert downgrade.kind == "validation"
    assert downgrade.error_kind is None


class _WorkspaceBackend(_RecordingBackend):
    """A backend spawned into a run overlay (`workspace=`)."""

    def spawn(self, prompt, env=None, workspace=None):
        return super().spawn(prompt, env=env)


def _run_generation(monkeypatch, tmp_path, reports, prep=None):
    """Drive the runner with the spec worker's own verdict hook and repair
    hook; each round's validate answers the next report."""
    root = _overlay_with_identity(tmp_path)
    monkeypatch.setattr(W, "_normalize_overlay_bundle", lambda *a, **k: None)
    answers = iter(reports)
    monkeypatch.setattr(W, "_invoke_validate", lambda *a, **k: next(answers))
    backend = _WorkspaceBackend(events=[{"type": "done", "success": True}],
                                stdout_lines=['{"raw": 1}\n'])
    monkeypatch.setattr(R, "get_backend", lambda cli: backend)
    prep_result = {"prompt": "p0", "target_root": root, **(prep or {})}
    frames = list(R.stream_agent_core(
        key="m", prep_result=prep_result, agent_cli="fake",
        log_path=str(tmp_path / "log.txt"), log_header="h", launch_banner="B",
        procs_dict={}, locks_lock=threading.Lock(), release_fn=lambda k: None,
        on_success=lambda _k, log_file, started_at, pr: W._on_success(
            "meta/x", log_file, started_at, pr),
        repair_prompt_fn=W._repair_prompt_fn("meta/x"), max_repair_rounds=2,
    ))
    events = [json.loads(f[len("data: "):].strip())
              for f in frames if f.startswith("data: ")]
    (done,) = [e for e in events if e.get("type") == "done"]
    return done, backend.prompts


def test_a_run_blocked_by_the_engine_stops_without_a_repair_round(monkeypatch, tmp_path):
    done, prompts = _run_generation(
        monkeypatch, tmp_path, [_analysis_report([_ENGINE], True)])
    assert prompts == ["p0"]
    assert done["success"] is False
    assert done["error_kind"] == "blocked_by_engine"
    assert [(g["capability"], g["deployments"]) for g in done["engine_gaps"]] \
        == [("gather extent", ["ep8_dpattn", "tp8"])]
    # an anonymous run keeps no record
    assert done["record"] == {"outcome": "none"}


def test_mixed_errors_repair_once_then_stop_when_only_the_engine_is_left(monkeypatch, tmp_path):
    done, prompts = _run_generation(monkeypatch, tmp_path, [
        _analysis_report([_ENGINE, _SPEC], False),
        _analysis_report([_ENGINE], True),
    ])
    assert len(prompts) == 2
    assert _SPEC in prompts[1] and _ENGINE not in prompts[1]
    assert "blocked_by_engine: true" in prompts[1]
    assert done["error_kind"] == "blocked_by_engine"


def test_a_repaired_round_drops_the_earlier_rounds_error_kind(monkeypatch, tmp_path):
    """Round one fails on unattributed refusals, round two passes: the
    success carries no failure fields from round one."""
    monkeypatch.setattr(W, "_publish_to_cache", lambda *a, **k: k["record_result"].update(
        {"outcome": "recorded", "id": "r9"}))
    done, prompts = _run_generation(monkeypatch, tmp_path, [
        _analysis_report([_OPEN], False),
        {"valid": True, "errors": []},
    ])
    assert len(prompts) == 2
    assert done == {"type": "done", "success": True,
                    "record": {"outcome": "recorded", "id": "r9"}}


def _signed_in(monkeypatch, record_pending):
    from backend import supabase_env
    from backend.service import spec_generations
    monkeypatch.setattr(supabase_env, "configured", lambda: True)
    monkeypatch.setenv("HA_RECORD_BLOCKED_SPECS", "1")
    monkeypatch.setattr(W, "_record_bundle", lambda *a, **k: ("spec", []))
    monkeypatch.setattr(W, "engine_version", lambda: "ev-1")
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return record_pending
    monkeypatch.setattr(spec_generations, "record_pending", fake)
    return seen


def test_a_blocked_signed_in_run_keeps_its_spec_as_the_pending_record(monkeypatch, tmp_path):
    seen = _signed_in(monkeypatch, {"outcome": "recorded", "id": "p1"})
    done, _ = _run_generation(monkeypatch, tmp_path, [_analysis_report([_ENGINE], True)],
                              prep={"user_id": "u1", "run_started_at": "2026-10-01T00:00:00+00:00"})
    assert done["record"] == {"outcome": "recorded", "id": "p1"}
    assert seen["user_id"] == "u1" and seen["engine_version"] == "ev-1"
    assert seen["run_started_at"] == "2026-10-01T00:00:00+00:00"
    assert [g["capability"] for g in seen["engine_gaps"]] == ["gather extent"]
    assert W.BLOCKED_NOT_SAVED not in done["message"]


def test_a_stale_blocked_run_says_so_on_done(monkeypatch, tmp_path):
    _signed_in(monkeypatch, {"outcome": "stale"})
    done, _ = _run_generation(monkeypatch, tmp_path, [_analysis_report([_ENGINE], True)],
                              prep={"user_id": "u1"})
    assert done["record"] == {"outcome": "stale"}


def test_a_blocked_run_whose_record_failed_says_it_was_not_saved(monkeypatch, tmp_path):
    _signed_in(monkeypatch, {"outcome": "failed", "reason": "503"})
    done, _ = _run_generation(monkeypatch, tmp_path, [_analysis_report([_ENGINE], True)],
                              prep={"user_id": "u1"})
    assert done["record"]["outcome"] == "failed"
    assert done["message"].endswith(W.BLOCKED_NOT_SAVED)
    assert done["error_kind"] == "blocked_by_engine"


# ---------------------------------------------------------------------------
# A round stopped by its own timer, with the run's deadline still ahead, is
# judged on what the agent wrote and continues as a repair round.
# ---------------------------------------------------------------------------
class _FiresOnStart:
    """A reaper timer that fires as soon as it is armed, for the first
    `n` rounds only."""
    left = 0

    def __init__(self, _seconds, fn):
        self.fn = fn
        self.daemon = False

    def start(self):
        if _FiresOnStart.left > 0:
            _FiresOnStart.left -= 1
            self.fn()

    def cancel(self):
        pass


class _ScriptedSessions(_FakeBackend):
    """Session-capable backend: the first round's process hangs until the
    reaper kills it; later rounds exit cleanly with a `done`."""
    supports_resume = True

    def __init__(self):
        super().__init__(events=[{"type": "done", "success": True}],
                         stdout_lines=['{"raw": 1}\n'])
        self.calls = []

    def _proc(self):
        if not self.calls[1:]:
            return _FakeProc([], returncode=None)
        return _FakeProc(['{"raw": 1}\n'], returncode=0)

    def spawn(self, prompt, env=None, workspace=None):
        self.calls.append(("spawn", None, prompt))
        return self._proc()

    def spawn_resume(self, session_id, prompt, env=None, workspace=None):
        self.calls.append(("resume", session_id, prompt))
        return self._proc()

    def extract_session_id(self, evt):
        return "sess-1"


def test_a_round_timeout_with_time_left_continues_as_a_repair(monkeypatch, tmp_path):
    monkeypatch.setattr(R.threading, "Timer", _FiresOnStart)
    _FiresOnStart.left = 1
    backend = _ScriptedSessions()
    verdicts = iter(["generated spec failed validation: bad dim", None])
    seen = []

    def repair(reason, prep):
        seen.append(prep.get("_round_timed_out"))
        return f"CONTINUE: {reason}"

    events = _run_frames(
        monkeypatch, tmp_path, backend,
        on_success=lambda *a: next(verdicts), repair_prompt_fn=repair,
        hard_cap_sec=7200, round_budget_sec=600)
    assert [e for e in events if e.get("type") == "done"] == [
        {"type": "done", "success": True}]
    assert seen == [True]
    # The first round never emitted a session id (it hung), so the repair
    # round cold-starts; the second round resumes its own session.
    assert backend.calls[0] == ("spawn", None, "p0")
    assert backend.calls[1][2] == "CONTINUE: generated spec failed validation: bad dim"
    assert any("reached its time budget" in (e.get("text") or "")
               for e in events if e.get("type") == "log")


def test_a_round_timeout_resumes_the_rounds_session(monkeypatch, tmp_path):
    """The timed-out round re-enters the session it ran in."""
    monkeypatch.setattr(R.threading, "Timer", _FiresOnStart)
    _FiresOnStart.left = 1

    class _HangsAfterInit(_ScriptedSessions):
        def _proc(self):
            if not self.calls[1:]:
                # Emits its init event, then hangs until killed.
                return _FakeProc(['{"init": 1}\n'], returncode=None)
            return _FakeProc(['{"raw": 1}\n'], returncode=0)

        def parse_event(self, evt):
            return [] if "init" in evt else self._events

    backend = _HangsAfterInit()
    verdicts = iter(["generated spec failed validation: bad dim", None])
    prep = {"prompt": "p0"}
    monkeypatch.setattr(R, "get_backend", lambda cli: backend)
    list(R.stream_agent_core(
        key="m", prep_result=prep, agent_cli="fake",
        log_path=str(tmp_path / "log.txt"), log_header="h", launch_banner="B",
        procs_dict={}, locks_lock=threading.Lock(), release_fn=lambda k: None,
        on_success=lambda *a: next(verdicts),
        repair_prompt_fn=lambda reason, p: "CONTINUE",
        hard_cap_sec=7200, round_budget_sec=600))
    assert backend.calls == [("spawn", None, "p0"), ("resume", "sess-1", "CONTINUE")]
    assert prep["_session_id"] == "sess-1"


def test_a_timed_out_round_whose_spec_passes_is_a_success(monkeypatch, tmp_path):
    monkeypatch.setattr(R.threading, "Timer", _FiresOnStart)
    _FiresOnStart.left = 1
    backend = _ScriptedSessions()
    events = _run_frames(
        monkeypatch, tmp_path, backend, on_success=lambda *a: None,
        repair_prompt_fn=lambda reason, prep: "CONTINUE",
        hard_cap_sec=7200, round_budget_sec=600)
    assert [e for e in events if e.get("type") == "done"] == [
        {"type": "done", "success": True}]
    assert len(backend.calls) == 1


def test_a_timed_out_round_with_nothing_written_ends_on_the_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(R.threading, "Timer", _FiresOnStart)
    _FiresOnStart.left = 1
    backend = _ScriptedSessions()
    events = _run_frames(
        monkeypatch, tmp_path, backend,
        on_success=lambda *a: "agent exited cleanly but wrote no model spec",
        repair_prompt_fn=lambda reason, prep: None,
        hard_cap_sec=7200, round_budget_sec=600,
        round_timeout_hint="HA_SPEC_ROUND_TIMEOUT_SECONDS")
    done = [e for e in events if e.get("type") == "done"]
    assert done == [{
        "type": "done", "success": False, "timed_out": True,
        "message": ("An agent round exceeded its budget of 10 minutes and was "
                    "stopped. Retry, or raise HA_SPEC_ROUND_TIMEOUT_SECONDS"),
        "agent_end": "timed_out",
    }]
    assert len(backend.calls) == 1


def test_the_run_deadline_ends_the_run_without_a_verdict(monkeypatch, tmp_path):
    monkeypatch.setattr(R.threading, "Timer", _FiresOnStart)
    _FiresOnStart.left = 1
    backend = _ScriptedSessions()
    judged = []
    events = _run_frames(
        monkeypatch, tmp_path, backend,
        on_success=lambda *a: judged.append(a) or "bad",
        repair_prompt_fn=lambda reason, prep: "CONTINUE",
        hard_cap_sec=600, round_budget_sec=7200)
    done = [e for e in events if e.get("type") == "done"]
    assert len(done) == 1 and done[0]["timed_out"] is True
    assert "exceeded 10 minutes" in done[0]["message"]
    assert judged == [] and len(backend.calls) == 1


def test_no_repair_round_left_judges_then_ends_on_the_timeout(monkeypatch, tmp_path):
    """The last round's own timeout is still judged; a downgrade there has no
    repair round to go to, so the run ends on the timeout."""
    monkeypatch.setattr(R.threading, "Timer", _FiresOnStart)
    _FiresOnStart.left = 1
    backend = _ScriptedSessions()
    judged = []
    asked = []
    events = _run_frames(
        monkeypatch, tmp_path, backend,
        on_success=lambda *a: judged.append(a) or "bad",
        repair_prompt_fn=lambda reason, prep: asked.append(reason) or "CONTINUE",
        max_repair_rounds=0, hard_cap_sec=7200, round_budget_sec=600)
    done = [e for e in events if e.get("type") == "done"]
    assert len(done) == 1 and done[0]["timed_out"] is True
    assert len(judged) == 1 and asked == [] and len(backend.calls) == 1


def test_the_last_rounds_timeout_with_a_passing_spec_is_a_success(monkeypatch, tmp_path):
    """Round 1 is repaired; the repair round (the last allowed) is stopped by
    its own timer with a spec that passes — the run succeeds."""
    monkeypatch.setattr(R.threading, "Timer", _FiresOnStart)
    _FiresOnStart.left = 2

    class _AlwaysHangs(_ScriptedSessions):
        def _proc(self):
            return _FakeProc([], returncode=None)

    backend = _AlwaysHangs()
    verdicts = iter(["generated spec failed validation: bad dim", None])
    events = _run_frames(
        monkeypatch, tmp_path, backend,
        on_success=lambda *a: next(verdicts),
        repair_prompt_fn=lambda reason, prep: "CONTINUE",
        max_repair_rounds=1, hard_cap_sec=7200, round_budget_sec=600)
    assert [e for e in events if e.get("type") == "done"] == [
        {"type": "done", "success": True}]
    assert len(backend.calls) == 2


def test_no_verdict_hook_ends_a_timed_out_round_unjudged(monkeypatch, tmp_path):
    monkeypatch.setattr(R.threading, "Timer", _FiresOnStart)
    _FiresOnStart.left = 1
    backend = _ScriptedSessions()
    events = _run_frames(monkeypatch, tmp_path, backend,
                         hard_cap_sec=7200, round_budget_sec=600)
    done = [e for e in events if e.get("type") == "done"]
    assert len(done) == 1 and done[0]["timed_out"] is True


def test_a_continued_run_resumes_the_given_session_first(monkeypatch, tmp_path):
    backend = _ScriptedSessions()
    backend.calls.append(("earlier", None, ""))  # every round exits cleanly
    prep = {"prompt": "CONTINUATION"}
    monkeypatch.setattr(R, "get_backend", lambda cli: backend)
    list(R.stream_agent_core(
        key="m", prep_result=prep, agent_cli="fake",
        log_path=str(tmp_path / "log.txt"), log_header="h", launch_banner="B",
        procs_dict={}, locks_lock=threading.Lock(), release_fn=lambda k: None,
        on_success=lambda *a: None, resume_session="sess-0"))
    assert backend.calls[1] == ("resume", "sess-0", "CONTINUATION")


def test_worker_repair_prompt_after_a_round_timeout_continues_in_place():
    make = W._repair_prompt_fn("meta/x")
    validation = W.Downgrade("validation", "failed", ("bad dim",))
    prompt = make(validation.reason,
                  {"downgrade": validation, "_round_timed_out": True})
    assert "reached its time budget" in prompt
    assert "bad dim" in prompt and "continue from them" in prompt
    no_output = W.Downgrade("no_output", "wrote no model spec")
    assert make(no_output.reason,
                {"downgrade": no_output, "_round_timed_out": True}) is None


def test_a_timed_out_round_with_drafts_but_no_model_spec_continues(tmp_path):
    """The likely state of a long generation at a round timeout: draft
    components written, the model spec not yet. It continues in place."""
    make = W._repair_prompt_fn("meta/x")
    (tmp_path / "models").mkdir()
    drafts = tmp_path / "components" / "drafts"
    drafts.mkdir(parents=True)
    (drafts / "x_attn.yaml").write_text("role: attention\n", encoding="utf-8")
    no_output = W.Downgrade("no_output", "wrote no model spec")
    prep = {"downgrade": no_output, "_round_timed_out": True,
            "target_root": str(tmp_path)}
    prompt = make(no_output.reason, prep)
    assert "`components/drafts/x_attn.yaml`" in prompt
    assert "`models/meta__x.yaml`" in prompt
    # A clean exit with nothing written is still not repaired.
    assert make(no_output.reason, {**prep, "_round_timed_out": False}) is None
    # Nor is a timeout that left nothing.
    (drafts / "x_attn.yaml").unlink()
    assert make(no_output.reason, prep) is None


def test_a_deployment_that_does_not_keep_blocked_specs_says_so(monkeypatch, tmp_path):
    """Until every instance knows pending records the write is gated off
    (`HA_RECORD_BLOCKED_SPECS`); the run says its spec was not saved."""
    _signed_in(monkeypatch, {"outcome": "recorded", "id": "p1"})
    monkeypatch.delenv("HA_RECORD_BLOCKED_SPECS")
    from backend.service import spec_generations
    monkeypatch.setattr(spec_generations, "record_pending",
                        lambda **kw: pytest.fail("a pending record was written"))
    done, _ = _run_generation(monkeypatch, tmp_path, [_analysis_report([_ENGINE], True)],
                              prep={"user_id": "u1"})
    assert done["record"]["outcome"] == "failed"
    assert "HA_RECORD_BLOCKED_SPECS" in done["record"]["reason"]
    assert done["message"].endswith(W.BLOCKED_NOT_SAVED)
