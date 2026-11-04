"""Launching review and fix on a record a local, database-less run left.

The orchestration (refuse on status, take the slot, prep → stream under
`start_job`) is shared with the service route; these drive it with no
Supabase configured, which is what the UI's Review / Agent Fix buttons could
not do for an offline run before. The agent subprocess is stubbed — nothing
here starts a real agent.
"""
from __future__ import annotations

import pytest

import backend.spec_worker as _sw
from backend.spec_worker import overlay, record_jobs, records
from backend.spec_worker.outcome import GenerationStatus

MODEL = "stepfun-ai/Step-5-Preview"
SPEC = "match: {model_id: stepfun-ai/Step-5-Preview}\nn_layers: 61\n"


@pytest.fixture
def offline(tmp_path, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setenv("HA_SPEC_RECORDS_ROOT", str(tmp_path / "spec_records"))
    return tmp_path


@pytest.fixture
def stub_agent(monkeypatch):
    """The slot, the prep and the stream, without an agent. Returns the prep
    calls and the started jobs."""
    seen: dict = {"prep": [], "jobs": [], "released": []}
    monkeypatch.setattr(_sw, "acquire", lambda k: "ok")
    monkeypatch.setattr(_sw, "release", seen["released"].append)
    monkeypatch.setattr(_sw, "start_job",
                        lambda k, fn, **kw: seen["jobs"].append((k, fn, kw)))

    def prep(model_id, **kw):
        seen["prep"].append((model_id, kw))
        return {"target_root": "/overlay"}

    def stream(model_id, prep_result, **kw):
        prep_result["review_row"] = {"id": None, "verdict": "no_pass"}
        yield 'data: {"type": "done", "success": true}\n\n'

    monkeypatch.setattr(_sw, "prepare_review", prep)
    monkeypatch.setattr(_sw, "prepare_fix", prep)
    monkeypatch.setattr(_sw, "stream_review", stream)
    monkeypatch.setattr(_sw, "stream_fix", stream)
    return seen


def _local_record(**kw):
    records.record(model_id=MODEL, user_id="", spec_yaml=SPEC,
                   new_component_yamls=[], agent_cli="claude", **kw)
    return records.serving_generation("", MODEL)


def test_a_review_runs_against_a_local_run_s_record(offline, stub_agent,
                                                    monkeypatch):
    monkeypatch.setattr(record_jobs, "record_bundle",
                        lambda model_id, spec_yaml, files: "BUNDLE")
    record = _local_record()
    started = record_jobs.launch_review(record, reviewer_id=record["user_id"])
    assert started["status"] == "started"
    assert started["run_key"] == f"review|{overlay.ANONYMOUS_OWNER}|{record['id']}"

    _key, run_fn, kw = stub_agent["jobs"][0]
    assert kw == {"model_id": MODEL, "user_id": overlay.ANONYMOUS_OWNER,
                  "kind": "review"}
    events = list(run_fn())
    assert events[-1] == {"type": "done", "success": True}
    model_id, prep_kw = stub_agent["prep"][0]
    assert model_id == MODEL
    assert prep_kw["record_id"] == record["id"]
    assert prep_kw["reviewer_id"] == overlay.ANONYMOUS_OWNER
    assert prep_kw["bundle"] == "BUNDLE"


def test_a_fix_runs_against_a_local_run_s_record(offline, stub_agent,
                                                 monkeypatch):
    monkeypatch.setattr(record_jobs, "fix_seed_bundle", lambda record: "BUNDLE")
    record = _local_record()
    started = record_jobs.launch_fix(record, fixer_id=record["user_id"],
                                     review_report="fix the rope base")
    assert started["status"] == "started"
    assert started["run_key"] == f"fix|{overlay.ANONYMOUS_OWNER}|{record['id']}"

    _key, run_fn, kw = stub_agent["jobs"][0]
    assert kw["kind"] == "fix"
    list(run_fn())
    _model_id, prep_kw = stub_agent["prep"][0]
    assert prep_kw["record_id"] == record["id"]
    assert prep_kw["review_report"] == "fix the rope base"
    assert prep_kw["work_order"] == []


def test_the_review_a_local_run_drives_is_keyed_apart_from_its_generate(
        offline, stub_agent, monkeypatch):
    """A record job's key is `{job}|{owner}|{record}`, never the generate
    run's `{owner}|{model}` — so a review cannot collide with the generate
    that produced the record it reviews."""
    monkeypatch.setattr(record_jobs, "record_bundle", lambda *a: "BUNDLE")
    monkeypatch.setattr(record_jobs, "fix_seed_bundle", lambda record: "BUNDLE")
    record = _local_record()
    keys = {record_jobs.launch_review(record, reviewer_id="x")["run_key"],
            record_jobs.launch_fix(record, fixer_id="x")["run_key"]}
    assert keys == {f"review|x|{record['id']}", f"fix|x|{record['id']}"}
    assert f"x|{MODEL}" not in keys


def test_an_overloaded_server_is_reported_not_raised(offline, monkeypatch):
    """Core answers `overloaded`; raising a 503 is the HTTP caller's job, so
    a local driver can decide for itself."""
    monkeypatch.setattr(_sw, "acquire", lambda k: "overloaded")
    record = _local_record()
    out = record_jobs.launch_review(record, reviewer_id="x")
    assert out["status"] == "overloaded" and "review again" in out["message"]
    assert record_jobs.launch_fix(record, fixer_id="x")["status"] == "overloaded"


def test_a_duplicate_run_is_refused(offline, monkeypatch):
    monkeypatch.setattr(_sw, "acquire", lambda k: "duplicate")
    record = _local_record()
    assert record_jobs.launch_review(
        record, reviewer_id="x")["status"] == "already_running"


@pytest.mark.parametrize("status", [GenerationStatus.BLOCKED_BY_ENGINE.value,
                                    GenerationStatus.NEEDS_REPAIR.value])
def test_a_pending_record_refuses_a_review_locally_too(offline, status,
                                                       monkeypatch):
    monkeypatch.setattr(_sw, "acquire",
                        lambda k: pytest.fail("took a slot for a pending record"))
    record = {**_local_record(), "status": status}
    assert record_jobs.launch_review(record, reviewer_id="x")["status"] == status


def test_a_prep_failure_frees_the_slot(offline, stub_agent, monkeypatch):
    monkeypatch.setattr(record_jobs, "record_bundle", lambda *a: "BUNDLE")

    def boom(*a, **k):
        raise RuntimeError("no modeling source")
    monkeypatch.setattr(_sw, "prepare_review", boom)
    record = _local_record()
    record_jobs.launch_review(record, reviewer_id="x")
    done = list(stub_agent["jobs"][0][1]())[-1]
    assert done["success"] is False and "no modeling source" in done["message"]
    assert stub_agent["released"] == [f"review|x|{record['id']}"]
