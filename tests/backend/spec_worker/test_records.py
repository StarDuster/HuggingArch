"""Where a run's generation record is kept, on both backends.

A record is one per (owner, model). With Supabase configured the write is
still the conditional RPC; with none, it is a file in the host-local store —
and review / fix find it either way, which is what makes an offline
generate → review → fix chain possible at all.
"""
from __future__ import annotations

import io
import json

import pytest

from backend.spec_worker import overlay, records, worker
from backend.spec_worker.outcome import GenerationStatus, RecordOutcome

MODEL = "stepfun-ai/Step-5-Preview"
SPEC = "match: {model_id: stepfun-ai/Step-5-Preview}\nn_layers: 61\n"
SPEC2 = SPEC + "# a second run wrote this\n"


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """A deployment with no database: the local backend answers."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setenv("HA_SPEC_RECORDS_ROOT", str(tmp_path / "spec_records"))
    return tmp_path / "spec_records"


def _record(spec_yaml=SPEC, **kw):
    return records.record(model_id=MODEL, user_id="", spec_yaml=spec_yaml,
                          snapshot_json={"config": {"model_type": "step5"}},
                          new_component_yamls=[{"path": "a.yaml", "body": "x"}],
                          agent_cli="claude", **kw)


# -- the local backend ---------------------------------------------------------

def test_an_offline_run_leaves_a_record(offline):
    out = _record()
    assert out["outcome"] == RecordOutcome.RECORDED.value
    assert out["id"] == records.local_record_id(overlay.ANONYMOUS_OWNER, MODEL)

    row = json.loads(
        (offline / overlay.ANONYMOUS_OWNER / overlay._safe(MODEL)
         / "record.json").read_text(encoding="utf-8"))
    assert row["status"] == GenerationStatus.GENERATED.value
    assert row["model_id"] == MODEL and row["spec_yaml"] == SPEC
    assert row["new_component_yamls"] == [{"path": "a.yaml", "body": "x"}]
    assert row["snapshot_json"]["config"]["model_type"] == "step5"
    assert row["revision"] == 0


def test_the_record_owner_is_the_reserved_local_owner(offline):
    """A local run has no auth user, so the record's owner is the one name
    this repo already gives the single user of a database-less deployment."""
    row = records.serving_generation("", MODEL)
    assert row is None
    _record()
    assert records.serving_generation("", MODEL)["user_id"] == (
        overlay.ANONYMOUS_OWNER)


def test_a_second_run_of_the_same_model_overwrites_the_record(offline):
    first = _record()
    second = _record(SPEC2)
    assert second["outcome"] == RecordOutcome.RECORDED.value
    # In place: the same record, so the reviews and fixes pointed at it are
    # not orphaned — the rule `record_generation` keeps for a MUTABLE row.
    assert second["id"] == first["id"]
    leaves = list(offline.glob("*/*/record.json"))
    assert len(leaves) == 1
    row = records.get(first["id"])
    assert row["spec_yaml"] == SPEC2 and row["revision"] == 1


def test_a_frozen_record_is_not_overwritten(offline):
    """`record_generation` inserts the new candidate BESIDE a frozen row. The
    local store has one slot per (owner, model), so it keeps what stands and
    reports that it did not write, rather than clobbering a contributed spec."""
    rid = _record()["id"]
    leaf = offline / overlay.ANONYMOUS_OWNER / overlay._safe(MODEL) / "record.json"
    row = json.loads(leaf.read_text(encoding="utf-8"))
    leaf.write_text(json.dumps(
        {**row, "status": GenerationStatus.CONTRIBUTED.value}), encoding="utf-8")

    out = _record(SPEC2)
    assert out["outcome"] == RecordOutcome.FAILED.value
    assert "contributed" in out["reason"]
    assert records.get(rid)["spec_yaml"] == SPEC


def test_get_and_serving_generation_find_the_same_record(offline):
    rid = _record()["id"]
    assert records.get(rid)["id"] == rid
    assert records.serving_generation("", MODEL)["id"] == rid
    assert records.get("no-such-record") is None
    assert records.serving_generation("", "other/Model") is None


def test_a_fix_writes_its_bundle_back_to_the_local_record(offline):
    rid = _record()["id"]
    assert records.update_spec(
        rid, spec_yaml=SPEC2, new_component_yamls=[],
        snapshot_json={"config": {"model_type": "step5.1"}}) is True
    row = records.get(rid)
    assert row["spec_yaml"] == SPEC2 and row["new_component_yamls"] == []
    assert row["snapshot_json"]["config"]["model_type"] == "step5.1"
    assert row["revision"] == 1
    # Scoped to the owner, as the hosted PATCH is: another owner matches no row.
    assert records.update_spec(rid, spec_yaml=SPEC, new_component_yamls=[],
                              user_id="someone-else") is False
    assert records.update_spec("no-such-record", spec_yaml=SPEC,
                              new_component_yamls=[]) is False


def test_the_local_store_keeps_no_pending_record(offline):
    """Nothing local is ever pending, so there is nothing to promote — and a
    blocked run's spec stays in its workspace, as it does for any run that
    cannot be recorded."""
    assert records.record_pending(
        model_id=MODEL, user_id="", spec_yaml=SPEC, engine_gaps=[],
        engine_version="v1")["outcome"] == RecordOutcome.NONE.value
    assert records.promote_pending(
        "rec-1", expected_revision=0)["outcome"] == RecordOutcome.NONE.value


def test_the_record_root_is_not_inside_a_run_overlay(monkeypatch):
    """Overlay leaves are pure cache and get garbage-collected; a record is
    not. It sits beside them under `~/.huggingarch`."""
    monkeypatch.delenv("HA_SPEC_RECORDS_ROOT", raising=False)
    root = records.spec_records_root()
    assert root.name == "spec_records"
    assert root.parent == overlay.spec_runs_root().parent


# -- the generate hook writes it ----------------------------------------------

def test_the_post_success_hook_of_an_offline_generate_leaves_a_record(
        offline, tmp_path, monkeypatch):
    """The real wiring: an offline run (no user, no Supabase) reaches the
    post-success hook, and the record review and fix are pointed at exists
    when it returns. Before this, such a run ended with `outcome: none` and
    the later steps had nothing to act on."""
    monkeypatch.setenv("HA_SHARED_SPEC_CACHE_ROOT", str(tmp_path / "shared"))
    monkeypatch.setattr("backend.analyzer.fetch_config", lambda *_: object())
    overlay_root = tmp_path / "run"
    spec_path = overlay_root / "models" / f"{overlay._safe(MODEL)}.yaml"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(f"match: {{model_id: [{MODEL}]}}\nn_layers: 61\n",
                         encoding="utf-8")
    log = io.StringIO()

    downgrade = worker._publish_to_cache(
        MODEL, log, started_at=0.0, user_id=None, agent_cli="claude",
        target_root=str(overlay_root), val_result={"valid": True, "errors": []},
        seeded=False, code_revision="pinned",
        run_started_at="2026-10-17T00:00:00+00:00")

    assert downgrade is None
    assert "[spec-record] recorded" in log.getvalue()
    assert f"owner={overlay.ANONYMOUS_OWNER}" in log.getvalue()
    row = records.serving_generation("", MODEL)
    assert row is not None and row["status"] == GenerationStatus.GENERATED.value
    assert "n_layers: 61" in row["spec_yaml"]


# -- the Supabase backend is untouched ----------------------------------------

def test_a_configured_deployment_still_writes_through_the_rpc(
        tmp_path, monkeypatch):
    """With Supabase configured the record is the `spec_generations` row the
    conditional RPC writes. The local store is not touched."""
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("HA_SPEC_RECORDS_ROOT", str(tmp_path / "spec_records"))
    from backend.service import spec_generations as _gens
    called = []
    monkeypatch.setattr(_gens, "_write_rpc", lambda fn, body: called.append(
        (fn, body)) or {"outcome": RecordOutcome.RECORDED.value, "id": "rec-1"})

    out = records.record(model_id=MODEL, user_id="user-uuid-1", spec_yaml=SPEC,
                         run_started_at="2026-10-17T00:00:00+00:00")
    assert out == {"outcome": RecordOutcome.RECORDED.value, "id": "rec-1"}
    assert [fn for fn, _ in called] == ["record_generation"]
    assert called[0][1]["p_user_id"] == "user-uuid-1"
    assert not (tmp_path / "spec_records").exists()


def test_a_configured_deployment_keeps_no_record_for_a_run_with_no_user(
        tmp_path, monkeypatch):
    """Unchanged: on a multi-user deployment an unattributed run keeps no
    record. The local store is for a deployment that has no database at all,
    not a second home for this one's anonymous runs."""
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("HA_SPEC_RECORDS_ROOT", str(tmp_path / "spec_records"))
    out = records.record(model_id=MODEL, user_id="", spec_yaml=SPEC)
    assert out == {"outcome": RecordOutcome.NONE.value}
    assert not (tmp_path / "spec_records").exists()
