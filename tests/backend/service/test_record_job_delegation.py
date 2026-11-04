"""The service route layer still owns only what a service run adds.

Review and fix share their orchestration with a local, database-less run
(`backend.spec_worker.record_jobs`). What stays here is the per-user hourly
quota, the 503 an overloaded server answers, and the dry-run admission a
PASS starts — these pin that none of it moved out, and that the route really
delegates rather than keeping a second copy.
"""
from __future__ import annotations

import pytest

import backend.spec_worker as _sw
from backend.service.routes import spec as _spec
from backend.spec_worker import record_jobs

RECORD = {"id": "rec-1", "model_id": "org/m", "user_id": "user-uuid-1",
          "spec_yaml": "x", "new_component_yamls": []}


def test_the_route_delegates_the_orchestration(monkeypatch):
    seen = {}

    def spy(verb):
        def launch(record, **kw):
            seen[verb] = kw
            return {"status": "started", "run_key": "k"}
        return launch

    monkeypatch.setattr(record_jobs, "launch_review", spy("review"))
    monkeypatch.setattr(record_jobs, "launch_fix", spy("fix"))

    assert _spec.launch_review(RECORD, reviewer_id="u", reviewer_role="admin"
                               )["status"] == "started"
    assert _spec.launch_fix(RECORD, fixer_id="u", fixer_role="admin",
                            review_report="r")["status"] == "started"
    # The admission a PASS starts is the route's own hook, passed in.
    assert callable(seen["review"].pop("on_pass"))
    assert seen["review"] == {"reviewer_id": "u", "reviewer_role": "admin",
                              "agent_cli": None, "agent_env": None}
    assert seen["fix"] == {"fixer_id": "u", "fixer_role": "admin",
                           "review_report": "r", "agent_cli": None,
                           "agent_env": None}


def test_a_pass_starts_the_admission(monkeypatch):
    """The hook the route hands in is the dry-run admission, on the review row
    the runner stored."""
    admissions = []
    monkeypatch.setattr(_spec, "launch_admission",
                        lambda record, review_id: admissions.append(review_id))
    monkeypatch.setattr(_sw, "acquire", lambda k: "ok")
    jobs = []
    monkeypatch.setattr(_sw, "start_job", lambda k, fn, **kw: jobs.append(fn))
    monkeypatch.setattr(record_jobs, "record_bundle", lambda *a: "BUNDLE")
    monkeypatch.setattr(_sw, "prepare_review", lambda m, **kw: {})

    def stream(model_id, prep_result, **kw):
        prep_result["review_row"] = {"id": "review-9", "verdict": "pass"}
        return iter(())
    monkeypatch.setattr(_sw, "stream_review", stream)

    _spec.launch_review(RECORD, reviewer_id="u", reviewer_role="admin")
    list(jobs[0]())
    assert admissions == ["review-9"]


@pytest.mark.parametrize("verb", ["review", "fix"])
def test_the_quota_gate_runs_before_any_slot_is_taken(monkeypatch, verb):
    monkeypatch.setattr(_spec, "_quota_ok", lambda uid: False)
    monkeypatch.setattr(record_jobs, "launch_review",
                        lambda *a, **k: pytest.fail("ran past the quota"))
    monkeypatch.setattr(record_jobs, "launch_fix",
                        lambda *a, **k: pytest.fail("ran past the quota"))
    out = (_spec.launch_review(RECORD, reviewer_id="u", reviewer_role="user")
           if verb == "review" else
           _spec.launch_fix(RECORD, fixer_id="u", fixer_role="user",
                            review_report=""))
    assert out["status"] == "quota_exceeded"


@pytest.mark.parametrize("verb", ["review", "fix"])
def test_an_overloaded_server_is_a_503(monkeypatch, verb):
    monkeypatch.setattr(_sw, "acquire", lambda k: "overloaded")
    monkeypatch.setattr(record_jobs, "record_bundle", lambda *a: "BUNDLE")
    with pytest.raises(Exception) as exc:
        if verb == "review":
            _spec.launch_review(RECORD, reviewer_id="u", reviewer_role="admin")
        else:
            _spec.launch_fix(RECORD, fixer_id="u", fixer_role="admin",
                             review_report="")
    assert getattr(exc.value, "status_code", None) == 503


def test_the_run_key_vocabulary_has_one_definition():
    """`RecordJob` and the two key helpers are the core module's; the route
    re-exports them so there is no second table of job prefixes."""
    assert _spec.RecordJob is record_jobs.RecordJob
    assert _spec.review_run_key is record_jobs.review_run_key
    assert _spec.fix_run_key is record_jobs.fix_run_key
    assert _spec.pending_refusal is record_jobs.pending_refusal
