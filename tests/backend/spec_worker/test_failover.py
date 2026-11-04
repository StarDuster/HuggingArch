"""A generate run goes on past an agent that, not the spec, stopped it.

An unavailable model service moves the run to the next agent backend with
credentials; a timed-out round with a draft continues from it; both inside
the run's one deadline, under the same lock and log. When nothing is left,
the run's `done` names the agent's ending as its `error_kind`.
"""
from __future__ import annotations

import json
import time

import pytest

from backend.spec_worker import failover as FO
from backend.spec_worker import limits
from backend.spec_worker import overlay
from backend.spec_worker import resume as RS
from backend.spec_worker import worker as W
from backend.spec_worker.outcome import RunErrorKind

MODEL = "meta/Foo"


def test_every_infrastructure_ending_is_a_run_error_kind():
    """The run's error kind for an agent ending is the agent's own word."""
    from backend.agents.base import AgentEnd
    infra = {e.value for e in AgentEnd} - {AgentEnd.OK.value, AgentEnd.FAILED.value}
    assert infra <= {k.value for k in RunErrorKind}
    assert infra == {"agent_unavailable", "timed_out", "killed",
                     "context_exhausted"}


# --- the order ---------------------------------------------------------------


def test_the_runs_own_cli_first_then_the_others(monkeypatch):
    monkeypatch.delenv(limits.FAILOVER_KNOB, raising=False)
    assert limits.failover_order("kimi", ("codex", "claude", "kimi")) == (
        "kimi", "codex", "claude")


def test_the_operator_orders_the_fallbacks(monkeypatch):
    monkeypatch.setenv(limits.FAILOVER_KNOB, "claude, Kimi")
    assert limits.failover_order("kimi", ("codex", "claude", "kimi")) == (
        "kimi", "claude")
    monkeypatch.setenv(limits.FAILOVER_KNOB, "")
    assert limits.failover_order("kimi", ("codex", "claude", "kimi")) == ("kimi",)
    monkeypatch.setenv(limits.FAILOVER_KNOB, "claude,gemini")
    with pytest.raises(ValueError, match="gemini"):
        limits.failover_order("kimi", ("codex", "claude", "kimi"))


def _fallbacks(prep: dict, own: str = "kimi") -> list[tuple[str, object]]:
    failover = FO.Failover(MODEL, target_root=None)
    failover.start(prep, own)
    return [(a.cli, a.env) for a in failover._fallbacks]


def test_a_host_run_falls_back_to_backends_the_host_has_keys_for(monkeypatch):
    monkeypatch.delenv(limits.FAILOVER_KNOB, raising=False)
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "KIMI_MODEL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-host")
    assert _fallbacks({"prompt": "p"}) == [("claude", None)]


def test_a_host_gateway_token_counts_and_an_endpoint_alone_does_not(monkeypatch):
    monkeypatch.delenv(limits.FAILOVER_KNOB, raising=False)
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "KIMI_MODEL_API_KEY",
                 "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example")
    assert _fallbacks({"prompt": "p"}) == []
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "gateway-token")
    assert _fallbacks({"prompt": "p"}) == [("claude", None)]


def test_a_byok_run_falls_back_only_to_what_the_service_resolved(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-host")
    monkeypatch.setenv(limits.FAILOVER_KNOB, "claude,codex")
    # No stamp: a BYOK run never borrows the host's keys.
    assert _fallbacks({"prompt": "p", "agent_env": {"KIMI_MODEL_API_KEY": "u"}}) == []
    stamped = {"codex": {"OPENAI_API_KEY": "sk-user"}, "claude": None}
    assert _fallbacks({"prompt": "p", "agent_env": {"KIMI_MODEL_API_KEY": "u"},
                       "agent_fallbacks": stamped}) == [
        ("claude", None), ("codex", {"OPENAI_API_KEY": "sk-user"})]


# --- the run -----------------------------------------------------------------


def _leaf(tmp_path, monkeypatch, *, draft=True, session=None):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    leaf = overlay.new_run_overlay("user-a", MODEL, kind="generate")
    if draft:
        (leaf / "components" / "drafts" / "foo_attn.yaml").write_text(
            "role: attention\n", encoding="utf-8")
    if session:
        transcript = leaf / ".agent-home" / ".claude" / "projects" / "-leaf"
        transcript.mkdir(parents=True)
        (transcript / f"{session}.jsonl").write_text("{}\n", encoding="utf-8")
    return leaf


class _Agents:
    """Stands in for `stream_agent_core`: each call is one agent attempt,
    which ends on the next scripted `done`."""

    def __init__(self, *dones, deadline_in=3600.0, session=None):
        self.dones = list(dones)
        self.calls: list[dict] = []
        self.deadline_in = deadline_in
        self.session = session

    def __call__(self, **kw):
        prep = kw["prep_result"]
        self.calls.append({**kw, "prompt": prep.get("prompt"),
                           "agent_env_seen": prep.get("agent_env")})
        if kw.get("deadline_at") is None:
            prep["_run_deadline_at"] = time.time() + self.deadline_in
        if self.session:
            prep["_session_id"] = self.session
        done = self.dones.pop(0)
        yield 'data: {"type": "log", "text": "working"}\n\n'
        yield f"data: {json.dumps(done)}\n\n"


def _run(monkeypatch, tmp_path, agents, prep, *, owns=True, failover=True):
    for name in ("bind_job_log", "release", "_stage_snapshot", "_upload_run_log"):
        monkeypatch.setattr(W, name, lambda *a, **k: None)
    monkeypatch.setattr(W, "_code_revision", lambda: "rev")
    monkeypatch.setattr(W, "_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(W, "owns", lambda key: owns)
    monkeypatch.setattr(W, "stream_agent_core", agents)
    fo = (FO.Failover(MODEL, target_root=prep.get("target_root"))
          if failover else None)
    events = [W.frame_event(f) for f in W.stream_agent(
        MODEL, prep, run_key="u|meta/Foo", failover=fo)]
    return [e for e in events if e is not None]


UNAVAILABLE = {"type": "done", "success": False,
               "message": "Exit code 1: error: failed to run prompt: "
                          "provider.auth_error: 401 Invalid Authentication",
               "agent_end": "agent_unavailable"}
ROUND_TIMEOUT = {"type": "done", "success": False, "timed_out": True,
                 "message": "An agent round exceeded its budget of 2 hours",
                 "agent_end": "timed_out", "error_kind": "spec_invalid"}
OK = {"type": "done", "success": True}


def _stub_verdict(monkeypatch):
    monkeypatch.setattr(RS._worker, "_verify_spec_output", lambda *a, **k: (
        W.Downgrade("validation", "failed validation: bad dim", ("bad dim",)),
        None))


def test_an_unavailable_agent_fails_over_to_the_next_one(tmp_path, monkeypatch):
    monkeypatch.setenv(limits.FAILOVER_KNOB, "codex,claude")
    leaf = _leaf(tmp_path, monkeypatch)
    _stub_verdict(monkeypatch)
    agents = _Agents(UNAVAILABLE, OK)
    codex_env = {"OPENAI_API_KEY": "sk-user"}
    prep = {"prompt": "TASK", "target_root": str(leaf), "agent_cli": "kimi",
            "agent_env": {"KIMI_MODEL_API_KEY": "k"},
            "agent_fallbacks": {"codex": codex_env}}
    events = _run(monkeypatch, tmp_path, agents, prep)

    dones = [e for e in events if e["type"] == "done"]
    assert dones == [OK]  # the unavailable attempt's done never leaves the run
    assert any("Kimi Code is unavailable" in e.get("text", "")
               and "continuing with Codex" in e.get("text", "") for e in events)
    first, second = agents.calls
    assert (first["agent_cli"], second["agent_cli"]) == ("kimi", "codex")
    assert second["env"]["OPENAI_API_KEY"] == "sk-user"
    assert second["env"]["HA_SPEC_OVERLAY_DIR"] == str(leaf)
    # The same deadline, the same log.
    assert second["deadline_at"] == prep["_run_deadline_at"]
    assert (first["append_log"], second["append_log"]) == (False, True)
    # A cold start on the draft: the task, then the draft and its verdict.
    assert second["prompt"].startswith("TASK")
    assert "`components/drafts/foo_attn.yaml`" in second["prompt"]
    assert "`models/meta__Foo.yaml` is not written yet" in second["prompt"]
    assert "provider.auth_error: 401" in second["prompt"]  # how it ended
    assert second["resume_session"] is None
    # The record names the agent that ran last.
    assert (prep["agent_cli"], prep["_agent_cli"]) == ("codex", "codex")


def test_with_no_draft_the_next_agent_starts_the_task_afresh(tmp_path, monkeypatch):
    monkeypatch.setenv(limits.FAILOVER_KNOB, "codex")
    leaf = _leaf(tmp_path, monkeypatch, draft=False)
    agents = _Agents(UNAVAILABLE, OK)
    prep = {"prompt": "TASK", "target_root": str(leaf), "agent_cli": "kimi",
            "agent_fallbacks": {"codex": None}}
    _run(monkeypatch, tmp_path, agents, prep)
    assert agents.calls[1]["prompt"] == "TASK"
    assert agents.calls[1]["env"] is None  # the host's own codex credentials


def test_no_backend_left_ends_the_run_agent_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv(limits.FAILOVER_KNOB, "codex")
    leaf = _leaf(tmp_path, monkeypatch, draft=False)
    agents = _Agents(UNAVAILABLE, UNAVAILABLE)
    prep = {"prompt": "TASK", "target_root": str(leaf), "agent_cli": "kimi",
            "agent_fallbacks": {"codex": {"OPENAI_API_KEY": "u"}}}
    done = [e for e in _run(monkeypatch, tmp_path, agents, prep)
            if e["type"] == "done"]
    assert len(agents.calls) == 2  # each backend tried once
    assert len(done) == 1
    assert done[0]["error_kind"] == RunErrorKind.AGENT_UNAVAILABLE.value
    assert done[0]["success"] is False


def test_a_timed_out_round_continues_from_its_draft(tmp_path, monkeypatch):
    leaf = _leaf(tmp_path, monkeypatch, session="sess-1")
    _stub_verdict(monkeypatch)
    agents = _Agents(ROUND_TIMEOUT, OK, session="sess-1")
    prep = {"prompt": "TASK", "target_root": str(leaf), "agent_cli": "claude"}
    events = _run(monkeypatch, tmp_path, agents, prep)
    assert [e for e in events if e["type"] == "done"] == [OK]
    _first, second = agents.calls
    assert second["agent_cli"] == "claude"
    # The same CLI re-enters its own session with the continuation.
    assert second["resume_session"] == "sess-1"
    assert "reached its time budget" in second["prompt"]
    assert "TASK" not in second["prompt"]
    assert second["deadline_at"] == prep["_run_deadline_at"]
    # The earlier round's verdict does not ride on the next attempt.
    assert "terminal" not in prep


EXHAUSTED = {"type": "done", "success": False,
             "message": "Prompt is too long · automatic compaction failed",
             "agent_end": "context_exhausted"}


def test_a_full_session_continues_from_its_draft_in_a_new_session(
        tmp_path, monkeypatch):
    """The session that filled its context window is not re-entered (it
    would overflow again): the same agent cold-starts on the task, the draft
    and its current verdict."""
    monkeypatch.setenv(limits.FAILOVER_KNOB, "codex")
    leaf = _leaf(tmp_path, monkeypatch, session="sess-1")
    _stub_verdict(monkeypatch)
    agents = _Agents(EXHAUSTED, OK, session="sess-1")
    prep = {"prompt": "TASK", "target_root": str(leaf), "agent_cli": "claude",
            "agent_fallbacks": {"codex": None}}
    events = _run(monkeypatch, tmp_path, agents, prep)
    assert [e for e in events if e["type"] == "done"] == [OK]
    assert any("outgrew its context window" in e.get("text", "") for e in events)
    _first, second = agents.calls
    assert second["agent_cli"] == "claude"  # the service answered: no failover
    assert second["resume_session"] is None
    assert second["prompt"].startswith("TASK")
    assert "`components/drafts/foo_attn.yaml`" in second["prompt"]
    assert "`models/meta__Foo.yaml` is not written yet" in second["prompt"]


def test_a_full_session_without_a_draft_ends_the_run_context_exhausted(
        tmp_path, monkeypatch):
    leaf = _leaf(tmp_path, monkeypatch, draft=False)
    agents = _Agents(EXHAUSTED, OK)
    prep = {"prompt": "TASK", "target_root": str(leaf), "agent_cli": "claude"}
    (done,) = [e for e in _run(monkeypatch, tmp_path, agents, prep)
               if e["type"] == "done"]
    assert len(agents.calls) == 1
    assert done["error_kind"] == RunErrorKind.AGENT_CONTEXT_EXHAUSTED.value


def test_a_spent_deadline_ends_the_run_timed_out_with_its_draft_kept(
        tmp_path, monkeypatch):
    leaf = _leaf(tmp_path, monkeypatch)
    agents = _Agents(ROUND_TIMEOUT, deadline_in=-1.0)
    prep = {"prompt": "TASK", "target_root": str(leaf), "agent_cli": "claude"}
    events = _run(monkeypatch, tmp_path, agents, prep)
    (done,) = [e for e in events if e["type"] == "done"]
    assert len(agents.calls) == 1
    assert done["error_kind"] == RunErrorKind.AGENT_TIMED_OUT.value
    assert overlay.authored_yaml(str(leaf))  # the draft stays for a continue


def test_a_timed_out_round_without_a_draft_ends_the_run(tmp_path, monkeypatch):
    leaf = _leaf(tmp_path, monkeypatch, draft=False)
    agents = _Agents(ROUND_TIMEOUT)
    prep = {"prompt": "TASK", "target_root": str(leaf), "agent_cli": "claude"}
    (done,) = [e for e in _run(monkeypatch, tmp_path, agents, prep)
               if e["type"] == "done"]
    assert len(agents.calls) == 1
    assert done["error_kind"] == "timed_out"


def test_a_stopped_run_is_not_continued(tmp_path, monkeypatch):
    monkeypatch.setenv(limits.FAILOVER_KNOB, "codex")
    leaf = _leaf(tmp_path, monkeypatch)
    agents = _Agents(UNAVAILABLE, OK)
    prep = {"prompt": "TASK", "target_root": str(leaf), "agent_cli": "kimi",
            "agent_fallbacks": {"codex": None}}
    (done,) = [e for e in _run(monkeypatch, tmp_path, agents, prep, owns=False)
               if e["type"] == "done"]
    assert len(agents.calls) == 1
    assert done["error_kind"] == "agent_unavailable"


def test_a_killed_agent_ends_the_run_killed(tmp_path, monkeypatch):
    leaf = _leaf(tmp_path, monkeypatch)
    killed = {"type": "done", "success": False, "message": "Exit code -9: ",
              "agent_end": "killed"}
    agents = _Agents(killed, OK)
    prep = {"prompt": "TASK", "target_root": str(leaf), "agent_cli": "claude"}
    (done,) = [e for e in _run(monkeypatch, tmp_path, agents, prep)
               if e["type"] == "done"]
    assert len(agents.calls) == 1
    assert done["error_kind"] == RunErrorKind.AGENT_KILLED.value


def test_a_spec_failure_is_not_the_agents(tmp_path, monkeypatch):
    """A verdict downgrade: the CLI finished, so no `agent_end`, no
    continuation, and the spec's own error kind stands."""
    leaf = _leaf(tmp_path, monkeypatch)
    invalid = {"type": "done", "success": False, "error_kind": "spec_invalid",
               "message": "generated spec failed validation: bad dim"}
    agents = _Agents(invalid, OK)
    prep = {"prompt": "TASK", "target_root": str(leaf), "agent_cli": "claude"}
    (done,) = [e for e in _run(monkeypatch, tmp_path, agents, prep)
               if e["type"] == "done"]
    assert len(agents.calls) == 1
    assert done == invalid


def test_without_failover_the_ending_is_still_named(tmp_path, monkeypatch):
    leaf = _leaf(tmp_path, monkeypatch)
    agents = _Agents(UNAVAILABLE)
    prep = {"prompt": "TASK", "target_root": str(leaf), "agent_cli": "kimi"}
    (done,) = [e for e in _run(monkeypatch, tmp_path, agents, prep, failover=False)
               if e["type"] == "done"]
    assert done["error_kind"] == "agent_unavailable"


def test_the_job_records_the_last_agent_and_a_resumable_draft(tmp_path, monkeypatch):
    """Through `generation_job`: the leaf's run record names the agent that
    ran last, and a run that ended on its deadline is resumable."""
    monkeypatch.setenv(limits.FAILOVER_KNOB, "codex")
    leaf = _leaf(tmp_path, monkeypatch)
    _stub_verdict(monkeypatch)
    agents = _Agents(UNAVAILABLE, {**ROUND_TIMEOUT}, deadline_in=3600.0)
    for name in ("bind_job_log", "release", "_stage_snapshot", "_upload_run_log"):
        monkeypatch.setattr(W, name, lambda *a, **k: None)
    monkeypatch.setattr(W, "_code_revision", lambda: "rev")
    monkeypatch.setattr(W, "_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(W, "owns", lambda key: True)

    def spent(**kw):  # the continuation spends the rest of the deadline
        kw["prep_result"]["_run_deadline_at"] = time.time() - 1
        yield from agents(**kw)

    attempts = iter([agents, spent])

    def core(**kw):
        return next(attempts)(**kw)

    monkeypatch.setattr(W, "stream_agent_core", core)
    prep = {"prompt": "TASK", "target_root": str(leaf), "agent_cli": "kimi",
            "agent_fallbacks": {"codex": None}}
    events = list(W.generation_job(
        MODEL, lambda: prep, run_key="u|meta/Foo", target_root=str(leaf),
        failover=FO.Failover(MODEL, target_root=str(leaf)))())
    done = events[-1]
    assert done["error_kind"] == "timed_out" and done["resumable"] is True
    end = overlay.RunEnd.read(leaf)
    assert (end.agent_cli, end.timed_out, end.success) == ("codex", True, False)
