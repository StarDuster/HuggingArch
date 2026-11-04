"""A commit that lands on the serving checkout while a review / fix runs.

The backend serves from a git checkout that other work commits to. The
verdict of a run depends on the spec and snapshot it judged, not on
unrelated commits: a moved HEAD must not turn a finished run into `error`.
The verdict gate validates against the checkout as it is when the gate runs,
and the run is stamped with both revisions.

`subprocess.run` is the only seam faked: `git` calls read a mutable HEAD,
the validate CLI returns a fixed JSON result. `_code_revision` and
`_invoke_validate` run for real.
"""
from __future__ import annotations

import io
import json
import subprocess
from types import SimpleNamespace

import pytest

from backend.spec_worker import reviewer
from backend.spec_worker import worker as W

_SHA_A = "a" * 40
_SHA_B = "b" * 40


class _Checkout:
    """HEAD of the serving checkout, and the validate CLI's answer."""

    def __init__(self, head: str, validate: dict):
        self.head = head
        self.validate = validate
        self.changed = ["web/src/account.ts"]
        self.validate_heads: list[str] = []

    def run(self, cmd, **_kw):
        if cmd[0] == "git" and cmd[1] == "rev-parse":
            return SimpleNamespace(returncode=0, stdout=self.head + "\n", stderr="")
        if cmd[0] == "git" and cmd[1] == "diff":
            return SimpleNamespace(
                returncode=0, stdout="".join(p + "\n" for p in self.changed),
                stderr="")
        assert cmd[1:3] == ["-m", "backend.inference.validate"], cmd
        self.validate_heads.append(self.head)
        return SimpleNamespace(
            returncode=0, stdout=json.dumps(self.validate), stderr="")


@pytest.fixture
def checkout(monkeypatch):
    monkeypatch.delenv("HA_GIT_SHA", raising=False)
    monkeypatch.delenv("GIT_SHA", raising=False)
    co = _Checkout(_SHA_A, {"valid": True, "errors": []})
    monkeypatch.setattr(subprocess, "run", co.run)
    return co


def _run_review(monkeypatch, tmp_path, checkout, *, commit_mid_run: bool):
    persisted: dict = {}
    monkeypatch.setattr(reviewer, "_persist_review",
                        lambda **kw: persisted.update(kw))
    monkeypatch.setattr(reviewer, "_upload_run_log", lambda *a, **k: None)
    monkeypatch.setattr(reviewer, "bind_job_log", lambda *a, **k: None)
    monkeypatch.setattr(reviewer, "_stage_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(reviewer, "_LOG_DIR", str(tmp_path))

    def _agent(**_kw):
        yield 'data: {"type": "log", "text": "VERDICT: NO_PASS", "author": "agent"}\n\n'
        if commit_mid_run:
            checkout.head = _SHA_B
        yield 'data: {"type": "done", "success": true}\n\n'

    monkeypatch.setattr(reviewer, "stream_agent_core", _agent)
    list(reviewer.stream_review("org/m", {
        "agent_cli": "claude",
        "agent_env": {},
        "code_revision": W._code_revision(),
        "record_id": "r1",
        "reviewer_id": "u1",
        "target_root": str(tmp_path),
    }))
    return persisted


def test_review_verdict_survives_a_commit_mid_run(monkeypatch, tmp_path, checkout):
    persisted = _run_review(monkeypatch, tmp_path, checkout, commit_mid_run=True)
    assert persisted["verdict"] == "no_pass"
    # The final validate ran against the checkout as it is at the verdict.
    assert checkout.validate_heads == [_SHA_B]
    report = persisted["report"]
    assert f"run started at {_SHA_A[:12]}" in report
    assert f"validated at {_SHA_B[:12]}" in report
    assert "web/src/account.ts" in report


def test_review_without_a_commit_is_unchanged(monkeypatch, tmp_path, checkout):
    persisted = _run_review(monkeypatch, tmp_path, checkout, commit_mid_run=False)
    assert persisted["verdict"] == "no_pass"
    assert checkout.validate_heads == [_SHA_A]
    report = persisted["report"]
    assert f"code: {_SHA_A[:12]}" in report
    assert "moved" not in report


def _identity(tmp_path) -> str:
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "org__m.yaml").write_text(
        "match: {model_id: [org/m]}\n", encoding="utf-8")
    return str(tmp_path)


def test_fix_gate_survives_a_commit_mid_run(monkeypatch, tmp_path, checkout):
    """Fix and generate share `_verify_spec_output`."""
    root = _identity(tmp_path)
    monkeypatch.setattr(W, "_normalize_overlay_bundle", lambda *a, **k: None)
    checkout.head = _SHA_B
    log = io.StringIO()
    downgrade, val_result = W._verify_spec_output(
        "org/m", log, started_at=0.0, root=root, seeded=True,
        code_revision=_SHA_A)
    assert downgrade is None
    assert val_result == {"valid": True, "errors": []}
    assert _SHA_A[:12] in log.getvalue() and _SHA_B[:12] in log.getvalue()


def test_new_validator_errors_after_a_commit_stay_repairable(
        monkeypatch, tmp_path, checkout):
    """A commit that makes the validator stricter surfaces as ordinary
    validation errors the repair loop hands back to the agent."""
    root = _identity(tmp_path)
    monkeypatch.setattr(W, "_normalize_overlay_bundle", lambda *a, **k: None)
    checkout.head = _SHA_B
    checkout.validate = {"valid": False, "errors": ["new rule"]}
    downgrade, _ = W._verify_spec_output(
        "org/m", io.StringIO(), started_at=0.0, root=root, seeded=True,
        code_revision=_SHA_A)
    assert downgrade.kind == "validation"
    assert W._repair_prompt_fn("org/m")(
        downgrade.reason, {"downgrade": downgrade})


@pytest.mark.parametrize("value,expected", [
    ("A" * 40, "a" * 40),
    (" " + "c" * 40 + "\n", "c" * 40),
    ("abc1234", "unknown"),
    ("z" * 40, "unknown"),
])
def test_code_revision_names_a_full_commit_or_unknown(monkeypatch, value, expected):
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.setenv("HA_GIT_SHA", value)
    assert W._code_revision() == expected


def test_code_revision_without_build_arg_or_git_is_unknown(monkeypatch):
    # The image built without `--build-arg HA_GIT_SHA`: the ENV is empty and
    # there is no `.git` to ask.
    monkeypatch.setenv("HA_GIT_SHA", "")
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=128, stdout="", stderr="fatal: not a git repository"))
    assert W._code_revision() == "unknown"
