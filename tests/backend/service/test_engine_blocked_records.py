"""Engine-blocked generation records: re-check, serving, the Admin filters.

The records themselves are written by SQL functions (migration 0002); here
the routes and the re-check are driven against those functions' answers.
"""
from __future__ import annotations

import threading
import time

import pytest

import backend.spec_worker as _sw
from backend.service import record_recheck as RC
from backend.service import spec_generations as _gens
from backend.service.routes import spec as _spec
from backend.spec_worker import runtime
from backend.spec_worker.outcome import GenerationStatus, RecheckOutcome
from .conftest import mock_admins_yes


def _auth(tok):
    return {"Authorization": f"Bearer {tok}"}


GAP = {"capability": "gather extent", "node": "lm_head", "site": "c.py:1",
       "exception": "EngineGap", "deployments": ["tp8"], "stages": ["decode"],
       "detail": "engine refusal at lm_head"}
SPEC_ERR = "analysis refuses node `mask`. Fix `op`: use slice"


def _pending(status="blocked_by_engine", **kw):
    return {"id": "rec-1", "model_id": "acme/Foo", "user_id": "user-uuid-1",
            "spec_yaml": "match: {model_id: [acme/Foo]}\n",
            "new_component_yamls": [], "snapshot_json": {"config": {}},
            "status": status, "revision": 4, "engine_gaps": [GAP],
            "blocked_engine_version": "ev-old", **kw}


# -- re-check: the six outcomes ------------------------------------------------

@pytest.fixture
def recheck(monkeypatch):
    """A re-check whose validate answers `state["validate"]` and whose
    writes are recorded; the record is `state["record"]`."""
    state = {"record": _pending(), "writes": []}
    monkeypatch.setattr(_gens, "get", lambda rid, strict=False: state["record"])
    monkeypatch.setattr(RC, "normalize_bundle", lambda *a, **k: object())
    monkeypatch.setattr(RC._overlay, "seed_recheck_overlay", lambda *a, **k: "/ov")
    monkeypatch.setattr(RC, "engine_version", lambda: "ev-new")

    def validate(*a, **k):
        answer = state["validate"]
        if isinstance(answer, Exception):
            raise answer
        return answer
    monkeypatch.setattr(RC, "_invoke_validate", validate)

    def transition(rid, **kw):
        state["writes"].append(("transition", kw))
        return state.get("answer", {"outcome": "recorded", "id": rid})

    def promote(rid, **kw):
        state["writes"].append(("promote", kw))
        return state.get("answer", {"outcome": "recorded", "id": rid})
    monkeypatch.setattr(_gens, "transition_pending", transition)
    monkeypatch.setattr(_gens, "promote_pending", promote)
    return state


def _done(state):
    events = list(RC.RecordRecheck("rec-1", actor_id="user-uuid-1").run())
    return events[-1]


def test_recheck_that_passes_promotes_at_the_read_revision(recheck):
    recheck["validate"] = {"valid": True, "errors": []}
    done = _done(recheck)
    assert done["recheck"] == RecheckOutcome.PASSED and done["success"] is True
    assert recheck["writes"] == [("promote", {"expected_revision": 4})]


def test_recheck_still_blocked_refreshes_the_gaps(recheck):
    recheck["validate"] = {"valid": False, "errors": [GAP["detail"]],
                           "blocked_by_engine": True,
                           "analysis_refusals": [{"kind": "analysis_engine_refusal", **GAP}]}
    done = _done(recheck)
    assert done["recheck"] == RecheckOutcome.STILL_BLOCKED
    (kind, kw), = recheck["writes"]
    assert kind == "transition" and kw["to"] == GenerationStatus.BLOCKED_BY_ENGINE
    assert kw["engine_gaps"] == [GAP] and kw["engine_version"] == "ev-new"
    assert kw["expected_revision"] == 4
    assert done["engine_gaps"] == [GAP]


def test_recheck_with_spec_errors_needs_repair_with_only_those(recheck):
    recheck["validate"] = {"valid": False, "errors": [GAP["detail"], SPEC_ERR],
                           "blocked_by_engine": False,
                           "analysis_refusals": [{"kind": "analysis_engine_refusal", **GAP}]}
    done = _done(recheck)
    assert done["recheck"] == RecheckOutcome.NEEDS_REPAIR
    (_, kw), = recheck["writes"]
    assert kw["to"] == GenerationStatus.NEEDS_REPAIR
    assert kw["repairable_errors"] == [SPEC_ERR]
    assert done["errors"] == [SPEC_ERR]


def test_recheck_of_an_unloadable_bundle_needs_repair_with_the_refusal(recheck, monkeypatch):
    from backend.spec_worker.bundle import SpecBundleError

    def refuse(*a, **k):
        raise SpecBundleError("draft component 'gone' is missing")
    monkeypatch.setattr(RC, "normalize_bundle", refuse)
    done = _done(recheck)
    assert done["recheck"] == RecheckOutcome.UNLOADABLE
    (_, kw), = recheck["writes"]
    assert kw["to"] == GenerationStatus.NEEDS_REPAIR
    assert kw["repairable_errors"] == ["draft component 'gone' is missing"]


def test_recheck_whose_record_moved_on_is_a_conflict(recheck):
    recheck["validate"] = {"valid": True, "errors": []}
    recheck["answer"] = {"outcome": "conflict"}
    done = _done(recheck)
    assert done["recheck"] == RecheckOutcome.CONFLICT and done["success"] is False


def test_recheck_of_a_record_no_longer_pending_writes_nothing(recheck):
    recheck["record"] = _pending(status="rejected")
    done = _done(recheck)
    assert done["recheck"] == RecheckOutcome.CONFLICT
    assert recheck["writes"] == []


def test_recheck_whose_validate_broke_writes_nothing(recheck):
    recheck["validate"] = RuntimeError("Hub unreachable")
    done = _done(recheck)
    assert done["recheck"] == RecheckOutcome.FAILED and done["success"] is False
    assert "Hub unreachable" in done["message"]
    assert recheck["writes"] == []


def test_recheck_storage_failure_is_failed(recheck):
    recheck["validate"] = {"valid": True, "errors": []}
    recheck["answer"] = {"outcome": "failed", "reason": "503"}
    assert _done(recheck)["recheck"] == RecheckOutcome.FAILED


# -- re-check: routes and the run key -------------------------------------------

def test_user_recheck_starts_a_job_on_its_own_key(client, user_token, monkeypatch):
    monkeypatch.setattr(_gens, "get", lambda rid: _pending())
    keys = []
    monkeypatch.setattr(_sw, "acquire", lambda k: keys.append(k) or "ok")
    monkeypatch.setattr(_sw, "start_job", lambda k, fn, **kw: None)
    r = client.post("/api/account/generations/rec-1/recheck", headers=_auth(user_token))
    assert r.status_code == 200 and r.json()["status"] == "started"
    assert keys == ["recheck|user-uuid-1|rec-1"]


def test_user_recheck_refuses_a_record_that_is_not_pending(client, user_token, monkeypatch):
    monkeypatch.setattr(_gens, "get", lambda rid: _pending(status="generated"))
    monkeypatch.setattr(_sw, "acquire", lambda k: pytest.fail("no job"))
    r = client.post("/api/account/generations/rec-1/recheck", headers=_auth(user_token))
    assert r.json()["status"] == "not_pending"


def test_user_recheck_of_someone_elses_record_is_403(client, user_token, monkeypatch):
    monkeypatch.setattr(_gens, "get", lambda rid: _pending(user_id="other"))
    r = client.post("/api/account/generations/rec-1/recheck", headers=_auth(user_token))
    assert r.status_code == 403


def test_admin_recheck_runs_under_the_admins_key(client, admin_token, httpx_mock, monkeypatch):
    mock_admins_yes(httpx_mock, "admin-uuid-1")
    monkeypatch.setattr(_gens, "get", lambda rid: _pending())
    keys = []
    monkeypatch.setattr(_sw, "acquire", lambda k: keys.append(k) or "ok")
    monkeypatch.setattr(_sw, "start_job", lambda k, fn, **kw: None)
    r = client.post("/api/admin/spec-generations/rec-1/recheck", headers=_auth(admin_token))
    assert r.json()["status"] == "started"
    assert keys == ["recheck|admin-uuid-1|rec-1"]


@pytest.fixture
def local_locks(monkeypatch):
    """The in-process lock alone (no cross-VM lock, no durability rows)."""
    monkeypatch.setattr(runtime, "_supabase_lock_enabled", lambda: False)
    monkeypatch.setattr(runtime, "_persist_begin", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "_persist_finish", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "invalidate_default_loader", lambda: None)
    for table in (runtime._model_locks, runtime._model_procs, runtime._slot_held,
                  runtime._lock_epoch, runtime._jobs, runtime._job_store):
        table.clear()
    monkeypatch.setattr(runtime, "semaphore", threading.Semaphore(runtime._max_concurrent))
    yield
    for table in (runtime._model_locks, runtime._model_procs, runtime._slot_held,
                  runtime._lock_epoch, runtime._jobs, runtime._job_store):
        table.clear()


def test_a_recheck_released_after_its_lock_was_reclaimed_leaves_the_new_owner(
        monkeypatch, local_locks):
    """The re-check runs in a job thread tied to the epoch it was started
    under. When its key's reservation is reclaimed (TTL) and retaken, the
    old thread's late release frees nothing of the new owner's."""
    woke, judged = threading.Event(), threading.Event()

    def slow_judge(self):
        woke.wait(5)
        judged.set()
        return RecheckOutcome.FAILED, {}
    monkeypatch.setattr(RC.RecordRecheck, "_judge", slow_judge)
    started = _spec.launch_recheck(_pending(), actor_id="user-uuid-1")
    key = started["run_key"]
    assert started["status"] == "started"
    # The reservation outlives its TTL and a new owner takes the key.
    runtime._model_locks[key] = time.monotonic() - runtime._LOCK_TTL_SECONDS - 1
    assert runtime.acquire(key) == "ok"
    woke.set()
    assert judged.wait(5)
    deadline = time.time() + 5
    while runtime._jobs[key]["state"] == "running" and time.time() < deadline:
        time.sleep(0.01)
    assert key in runtime._model_locks          # the new owner still holds it
    assert runtime.acquire(key) == "duplicate"


# -- serving --------------------------------------------------------------------

@pytest.fixture
def served(monkeypatch):
    """Analyze with no library spec, serving the caller's record."""
    import backend.analyzer as analyzer
    monkeypatch.setattr(_spec, "_peruser_active", lambda: True)
    monkeypatch.setattr(_spec, "_normalize_record_bundle", lambda *a, **k: object())
    monkeypatch.setattr(_spec._overlay, "materialize_user_spec", lambda *a, **k: "/ov")
    calls = {"n": 0}

    def analyze_model(**kw):
        calls["n"] += 1
        if "overlay_dir" not in kw:
            raise analyzer.NoSpecMatch("No spec matches 'acme/Foo'. run /generate-spec first.")
        return calls["private"](**kw)
    monkeypatch.setattr(analyzer, "analyze_model", analyze_model)
    return calls


def test_analyze_serves_a_blocked_record_with_its_gaps(client, user_token, monkeypatch, served):
    import backend.api.routes.analyze as A
    monkeypatch.setattr(_gens, "serving_generation", lambda uid, mid: _pending())
    served["private"] = lambda **kw: {"success": True}
    monkeypatch.setattr(A, "build_response", lambda *a, **k: {"success": True})
    r = client.post("/api/spec/analyze", json={"model_id": "acme/Foo"}, headers=_auth(user_token))
    body = r.json()
    assert body["success"] is True
    assert body["served_from"] == "user_generation"
    assert body["record_status"] == "blocked_by_engine"
    assert body["engine_gaps"] == [GAP]


def test_analyze_of_a_declaration_record_is_fixable(client, user_token, monkeypatch, served):
    """A saved spec that cannot load for a named declaration carries
    `fixable` and `record_id` so the Inspector can start Agent Fix."""
    from backend.arch.spec_migrate import SpecMigrationBlocked
    monkeypatch.setattr(_gens, "serving_generation",
                        lambda uid, mid: _pending(status="generated", engine_gaps=None))

    def boom(*a, **k):
        raise SpecMigrationBlocked(
            "needs a declaration", errors=["declare reads_from: { hc_pre_mix: i - 1 }"])
    monkeypatch.setattr(_spec, "_normalize_record_bundle", boom)
    body = client.post("/api/spec/analyze", json={"model_id": "acme/Foo"},
                       headers=_auth(user_token)).json()
    assert body["success"] is False
    assert body["fixable"] is True
    assert body["record_id"] == "rec-1"
    assert body["error_kind"] == "spec_needs_declaration"
    assert body["served_from"] == "user_generation"


def test_analyze_of_a_valid_record_carries_no_gaps(client, user_token, monkeypatch, served):
    import backend.api.routes.analyze as A
    monkeypatch.setattr(_gens, "serving_generation",
                        lambda uid, mid: _pending(status="generated", engine_gaps=None))
    served["private"] = lambda **kw: {"success": True}
    monkeypatch.setattr(A, "build_response", lambda *a, **k: {"success": True})
    body = client.post("/api/spec/analyze", json={"model_id": "acme/Foo"},
                       headers=_auth(user_token)).json()
    assert body["record_status"] == "generated" and "engine_gaps" not in body
    assert body["record_id"] == "rec-1"


def test_analyze_that_hits_an_engine_gap_says_so_as_data(client, user_token, monkeypatch, served):
    from backend.arch.eval.refusal import EngineGap
    monkeypatch.setattr(_gens, "serving_generation", lambda uid, mid: _pending())

    def refuse(**kw):
        raise EngineGap("cannot close the gather", capability="gather extent",
                        node="layers.0.lm_head")
    served["private"] = refuse
    body = client.post("/api/spec/analyze", json={"model_id": "acme/Foo"},
                       headers=_auth(user_token)).json()
    assert body["success"] is False
    assert body["error_kind"] == "blocked_by_engine"
    assert body["engine_gap"] == {"capability": "gather extent", "node": "layers.0.lm_head"}
    assert body["record_status"] == "blocked_by_engine"


@pytest.mark.parametrize("raised,kind,extra", [
    # the spec writer resolves it: the repair, for Refine / Agent Fix
    (lambda: __import__("backend.arch.eval.refusal", fromlist=["SpecRefusal"]).SpecRefusal(
        "dims disagree", declaration="dims", fix="declare d_model"),
     "spec_invalid", {"fixable": True}),
    # Python's own statement that a capability is missing: the engine's
    (lambda: NotImplementedError("ring attention under pp"), "blocked_by_engine", {}),
    # no declared owner
    (lambda: KeyError("d_head"), "analysis_refusal_unattributed", {}),
])
def test_a_private_analysis_failure_is_classified_by_its_owner(
        client, user_token, monkeypatch, served, raised, kind, extra):
    """Every private-spec failure says who resolves it as data, and none of
    them says `no_spec`: the page answers none with a generation."""
    monkeypatch.setattr(_gens, "serving_generation",
                        lambda uid, mid: _pending(status="generated", engine_gaps=None))

    def refuse(**kw):
        raise raised()
    served["private"] = refuse
    body = client.post("/api/spec/analyze", json={"model_id": "acme/Foo"},
                       headers=_auth(user_token)).json()
    assert body["success"] is False and body["error_kind"] == kind
    assert body["served_from"] == "user_generation" and "no_spec" not in body
    for key, value in extra.items():
        assert body[key] == value
    if kind == "spec_invalid":
        assert "declare d_model" in body["errors"][0]


def test_records_that_cannot_be_read_are_no_reason_to_generate(client, user_token, monkeypatch, served):
    def outage(uid, mid):
        raise _gens.SpecGenerationsUnavailable("503")
    monkeypatch.setattr(_gens, "serving_generation", outage)
    body = client.post("/api/spec/analyze", json={"model_id": "acme/Foo"},
                       headers=_auth(user_token)).json()
    assert body["success"] is False and body["status"] == "records_unavailable"
    assert "no_spec" not in body


@pytest.mark.parametrize("status", ["needs_repair", "rejected"])
def test_a_record_that_is_not_served_is_not_a_missing_spec(client, user_token, monkeypatch, served, status):
    """The user's only record needs repair or was rejected: it is not
    analysed, and the model is not spec-less either — nothing generates."""
    monkeypatch.setattr(_gens, "serving_generation", lambda uid, mid: _pending(status=status))
    served["private"] = lambda **kw: pytest.fail("analysed a record that is not served")
    body = client.post("/api/spec/analyze", json={"model_id": "acme/Foo"},
                       headers=_auth(user_token)).json()
    assert body["success"] is False and "no_spec" not in body
    assert body["served_from"] == "user_generation" and body["record_status"] == status


def test_no_spec_at_all_says_so_as_data(client, user_token, monkeypatch, served):
    monkeypatch.setattr(_gens, "serving_generation", lambda uid, mid: None)
    body = client.post("/api/spec/analyze", json={"model_id": "acme/Foo"},
                       headers=_auth(user_token)).json()
    assert body["success"] is False and body["no_spec"] is True
    assert "served_from" not in body


# -- account list ------------------------------------------------------------------

def test_account_list_marks_an_engine_change(client, user_token, monkeypatch):
    from backend.service import spec_generation_logs as _logs
    from backend.service import spec_generation_reviews as _reviews
    monkeypatch.setattr(_spec, "engine_version", lambda: "ev-new")
    monkeypatch.setattr(_spec, "library_spec_for", lambda mid: None)
    monkeypatch.setattr(_gens, "list_for_user", lambda uid: [
        {"id": "a", "model_id": "m", "status": "blocked_by_engine",
         "blocked_engine_version": "ev-old", "created_at": "2026-10-01T00:00:00Z"},
        {"id": "b", "model_id": "n", "status": "blocked_by_engine",
         "blocked_engine_version": "ev-new", "created_at": "2026-10-01T00:00:00Z"},
        {"id": "c", "model_id": "o", "status": "generated",
         "created_at": "2026-10-01T00:00:00Z"},
    ])
    monkeypatch.setattr(_logs, "list_for_user", lambda uid, limit=50: [
        {"id": "l1", "model_id": "z", "success": False, "error_kind": "blocked_by_engine",
         "engine_gaps": [GAP], "created_at": "2026-10-01T00:00:00Z"}])
    monkeypatch.setattr(_reviews, "latest_verdicts_for_records", lambda ids: {})
    body = client.get("/api/account/generations", headers=_auth(user_token)).json()
    changed = {it["id"]: it.get("engine_changed") for it in body["items"]}
    assert changed == {"a": True, "b": False, "c": None}
    (orphan,) = body["orphan_logs"]
    assert orphan["error_kind"] == "blocked_by_engine" and orphan["engine_gaps"] == [GAP]


# -- Admin filters -----------------------------------------------------------------

def _admin_rpc(monkeypatch):
    from backend.service.routes import admin as _admin
    calls = []

    def rpc(fn, params, *, user_jwt):
        calls.append((fn, params))
        return 2 if fn.endswith("_count") else []
    monkeypatch.setattr(_admin, "_call_admin_rpc", rpc)
    return calls


def test_admin_generations_filter_by_status(client, admin_token, httpx_mock, monkeypatch):
    mock_admins_yes(httpx_mock, "admin-uuid-1")
    calls = _admin_rpc(monkeypatch)
    r = client.get("/api/admin/spec-generations?status=blocked_by_engine",
                   headers=_auth(admin_token))
    assert r.status_code == 200 and r.json() == {"items": [], "total": 2}
    assert calls == [
        ("get_spec_generations", {"p_limit": 50, "p_offset": 0, "p_sort": "recent",
                                  "p_status": "blocked_by_engine"}),
        ("get_spec_generations_count", {"p_status": "blocked_by_engine"}),
    ]


def test_admin_runs_filter_by_error_kind(client, admin_token, httpx_mock, monkeypatch):
    mock_admins_yes(httpx_mock, "admin-uuid-1")
    calls = _admin_rpc(monkeypatch)
    r = client.get("/api/admin/spec-generation-runs?error_kind=analysis_refusal_unattributed",
                   headers=_auth(admin_token))
    assert r.status_code == 200
    assert calls[0] == ("get_spec_generation_runs", {
        "p_limit": 50, "p_offset": 0, "p_error_kind": "analysis_refusal_unattributed"})
    assert calls[1] == ("get_spec_generation_runs_count",
                        {"p_error_kind": "analysis_refusal_unattributed"})


@pytest.mark.parametrize("url", [
    "/api/admin/spec-generations?status=blocked",
    "/api/admin/spec-generation-runs?error_kind=engine",
])
def test_admin_filters_refuse_an_unknown_value(client, admin_token, httpx_mock, monkeypatch, url):
    mock_admins_yes(httpx_mock, "admin-uuid-1")
    calls = _admin_rpc(monkeypatch)
    r = client.get(url, headers=_auth(admin_token))
    assert r.status_code == 400
    assert calls == []


def test_vocabulary_lists_every_value(client):
    from backend.spec_worker.outcome import GenerationStatus, RunErrorKind
    body = client.get("/api/spec/generation-vocabulary").json()
    assert body["generation_status"] == [s.value for s in GenerationStatus]
    assert body["run_error_kind"] == [k.value for k in RunErrorKind]
    assert body["pending"] == ["blocked_by_engine", "needs_repair"]


# -- a pending record never becomes a second working candidate --------------------

def test_admin_reject_of_a_pending_record_discards_it(client, admin_token, httpx_mock, monkeypatch):
    """Rejected in place, a pending record would be a second working
    candidate (the database refuses that too); it is deleted instead."""
    import re
    from .conftest import TEST_SUPABASE_URL
    mock_admins_yes(httpx_mock, "admin-uuid-1")
    monkeypatch.setattr(_gens, "get", lambda rid: _pending())
    monkeypatch.setattr(_gens, "set_status", lambda *a, **k: pytest.fail("rejected in place"))
    httpx_mock.add_response(
        method="DELETE", url=re.compile(re.escape(f"{TEST_SUPABASE_URL}/rest/v1/spec_generations") + r".*"),
        json=[{"id": "rec-1"}])
    r = client.post("/api/admin/spec-generations/rec-1/reject", headers=_auth(admin_token))
    assert r.json() == {"rejected": True, "discarded": True, "id": "rec-1"}
    (delete,) = [q for q in httpx_mock.get_requests() if q.method == "DELETE"]
    assert delete.url.params["status"] == "in.(blocked_by_engine,needs_repair)"


def test_plain_status_and_bundle_writes_never_match_a_pending_record(httpx_mock):
    import re
    from .conftest import TEST_SUPABASE_URL
    url = re.compile(re.escape(f"{TEST_SUPABASE_URL}/rest/v1/spec_generations") + r".*")
    httpx_mock.add_response(method="PATCH", url=url, json=[], is_reusable=True)
    assert _gens.set_status("rec-1", "rejected") is False
    assert _gens.update_spec("rec-1", spec_yaml="x", new_component_yamls=[]) is False
    for req in httpx_mock.get_requests():
        assert req.url.params["status"] == "not.in.(blocked_by_engine,needs_repair)"


def test_recheck_that_cannot_read_the_record_is_failed_not_conflict(recheck, monkeypatch):
    def outage(rid, strict=False):
        assert strict
        raise _gens.SpecGenerationsUnavailable("503")
    monkeypatch.setattr(_gens, "get", outage)
    done = _done(recheck)
    assert done["recheck"] == RecheckOutcome.FAILED
    assert recheck["writes"] == []
