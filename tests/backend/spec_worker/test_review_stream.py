"""What a review run stores: the agent's own text, and an explicit `error`
when the run is cut off at its time budget.

The DeepSeek-V4.1-Flash review was killed at its 20-minute cap during its
final response. It stored `no_pass` (the validator's verdict standing in for
findings) and a report body made of runner log lines ("Reviewing spec...",
"$ ls ..."). `stream_agent_core` is faked; everything after it is real.
"""
from __future__ import annotations

import json

import pytest

from backend.spec_worker import reviewer


def _frame(**evt) -> str:
    return f"data: {json.dumps(evt)}\n\n"


@pytest.fixture
def run(monkeypatch, tmp_path):
    persisted: dict = {}
    caps: list = []
    monkeypatch.setattr(reviewer, "_persist_review",
                        lambda **kw: persisted.update(kw))
    monkeypatch.setattr(reviewer, "_upload_run_log", lambda *a, **k: None)
    monkeypatch.setattr(reviewer, "bind_job_log", lambda *a, **k: None)
    monkeypatch.setattr(reviewer, "_stage_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(reviewer, "_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(reviewer, "_invoke_validate",
                        lambda *a, **k: {"valid": False, "errors": ["boom"]})

    def _go(frames, *, budget=None):
        def _agent(**kw):
            caps.append(kw["hard_cap_sec"])
            yield from frames

        monkeypatch.setattr(reviewer, "stream_agent_core", _agent)
        prep = {"agent_cli": "claude", "agent_env": {}, "code_revision": "x" * 40,
                "record_id": "r1", "reviewer_id": "u1",
                "target_root": str(tmp_path)}
        if budget is not None:
            prep["review_budget_sec"] = budget
        list(reviewer.stream_review("org/m", prep))
        return persisted, caps

    return _go


_TOOL_LOGS = [
    _frame(type="log", text="Reviewing spec..."),
    _frame(type="log", text="Coding agent connected"),
    _frame(type="log", text="$ ls source/"),
    _frame(type="log", text="[Read] /ws/source/model.py"),
]


def test_the_report_is_the_agents_text_not_the_run_log(run):
    persisted, _ = run([
        *_TOOL_LOGS,
        _frame(type="log", author="agent",
               text="===REVIEW REPORT START===\n- rope is partial\n"
                    "VERDICT: NO_PASS\n===REVIEW REPORT END==="),
        _frame(type="done", success=True),
    ])
    assert persisted["verdict"] == "no_pass"
    report = persisted["report"]
    assert "rope is partial" in report
    for line in ("Reviewing spec", "Coding agent connected", "$ ls", "[Read]"):
        assert line not in report


def test_a_review_cut_off_at_its_budget_is_an_error_that_says_so(run):
    persisted, _ = run([
        *_TOOL_LOGS,
        _frame(type="log", author="agent", text="Checking the attention first."),
        _frame(type="done", success=False, timed_out=True,
               message="Agent execution exceeded 35 minutes and was stopped"),
    ], budget=2100)
    # Not the validator's no_pass standing in for findings nobody wrote.
    assert persisted["verdict"] == "error"
    report = persisted["report"]
    assert "stopped at its time budget (35 min)" in report
    assert "not a verdict on the spec" in report
    assert "Checking the attention first." in report
    assert "$ ls" not in report
    # The deterministic validator block is still part of the stored report.
    assert "boom" in report


def test_the_run_is_capped_at_the_budget_prep_sized(run):
    _, caps = run([_frame(type="done", success=True)], budget=2088)
    assert caps == [2088]


def test_a_prep_without_a_budget_runs_at_the_floor(run):
    _, caps = run([_frame(type="done", success=True)])
    assert caps == [reviewer.REVIEW_AGENT_SEC]
