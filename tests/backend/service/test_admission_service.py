"""The service side of a record's admission (ADMIT §4.1, §4.2, §6.2):
storage fallbacks around migration 0038, the dry run that follows a PASS,
the base it measures against, staleness, and what Open PR may do with a
report."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from backend.service import admission as adm
from backend.service import spec_generation_reviews as R
from backend.service import spec_pr
from backend.spec_worker.bundle import record_fingerprint

from .conftest import TEST_SUPABASE_URL, mock_admins_yes
from .fake_github import FakeGitHub
from .test_spec_pr_admission import MODEL, RECORD, FakeAdmitter, Store

_REVIEWS = re.compile(re.escape(f"{TEST_SUPABASE_URL}/rest/v1/spec_generation_reviews") + r"(\?.*)?")
_GENS = re.compile(re.escape(f"{TEST_SUPABASE_URL}/rest/v1/spec_generations") + r"(\?.*)?")
_UNKNOWN = '{"code":"42703","message":"column spec_generation_reviews.admission does not exist"}'


def _auth(tok):
    return {"Authorization": f"Bearer {tok}"}


# ── storage around migration 0038 ─────────────────────────────────────────

def test_owner_read_falls_back_without_the_admission_column(httpx_mock):
    httpx_mock.add_response(method="GET", url=_GENS, json=[{"id": "r1"}])
    httpx_mock.add_response(method="GET", url=_REVIEWS, status_code=400, text=_UNKNOWN)
    httpx_mock.add_response(method="GET", url=_REVIEWS, json=[{"id": "rev1", "verdict": "pass"}])
    rows = R.list_for_record_owner("r1", "u1")
    assert rows == [{"id": "rev1", "verdict": "pass", "admission_unavailable": True}]
    first, second = [r for r in httpx_mock.get_requests() if "reviews" in str(r.url)]
    assert "admission" in first.url.params["select"]
    assert "admission" not in second.url.params["select"]
    assert first.url.params["order"] == "created_at.desc,id.desc"


def test_owner_read_carries_the_admission_when_deployed(httpx_mock):
    envelope = {"job": "done", "mode": "dry_run", "report": {"status": "ready"}}
    httpx_mock.add_response(method="GET", url=_GENS, json=[{"id": "r1"}])
    httpx_mock.add_response(method="GET", url=_REVIEWS, json=[{"id": "rev1", "admission": envelope}])
    (row,) = R.list_for_record_owner("r1", "u1")
    assert row["admission"] == envelope and "admission_unavailable" not in row


def test_admission_patch_without_the_column_is_logged_not_raised(httpx_mock, caplog):
    httpx_mock.add_response(method="PATCH", url=_REVIEWS, status_code=400,
                            text='{"code":"PGRST204","message":"Could not find the '
                                 "'admission' column of 'spec_generation_reviews'\"}")
    assert R.set_admission("rev1", {"job": "running"}) is False
    assert "not deployed" in caplog.text
    (req,) = httpx_mock.get_requests()
    assert req.url.params["id"] == "eq.rev1"
    assert json.loads(req.content) == {"admission": {"job": "running"}}


def test_admission_patch_ok(httpx_mock):
    httpx_mock.add_response(method="PATCH", url=_REVIEWS, status_code=204)
    assert R.set_admission("rev1", {"job": "done"}) is True


def test_admin_rpc_before_the_migration_marks_rows_unavailable(
        client, admin_token, httpx_mock, monkeypatch):
    from backend.service.routes import admin
    monkeypatch.setattr(admin._gens, "get", lambda rid: None)
    mock_admins_yes(httpx_mock, "admin-uuid-1")
    httpx_mock.add_response(
        method="POST", url=f"{TEST_SUPABASE_URL}/rest/v1/rpc/get_spec_generation_reviews",
        json=[{"id": "rev1", "verdict": "pass"}])
    r = client.get("/api/admin/spec-generations/rec-1/reviews", headers=_auth(admin_token))
    assert r.json()["items"] == [{"id": "rev1", "verdict": "pass", "admission_unavailable": True,
                                  "bundle_current": False}]


def test_admin_rpc_after_the_migration_passes_the_admission_through(
        client, admin_token, httpx_mock, monkeypatch):
    from backend.service.routes import admin
    monkeypatch.setattr(admin._gens, "get", lambda rid: RECORD)
    mock_admins_yes(httpx_mock, "admin-uuid-1")
    httpx_mock.add_response(
        method="POST", url=f"{TEST_SUPABASE_URL}/rest/v1/rpc/get_spec_generation_reviews",
        json=[{"id": "rev1", "verdict": "pass", "admission": None,
               "bundle_fingerprint": record_fingerprint(RECORD)},
              {"id": "rev0", "verdict": "pass", "admission": None, "bundle_fingerprint": "x"}])
    r = client.get("/api/admin/spec-generations/rec-1/reviews", headers=_auth(admin_token))
    assert [(x["id"], x["admission"], x["bundle_current"]) for x in r.json()["items"]] == [
        ("rev1", None, True), ("rev0", None, False)]


def test_the_baseline_carries_the_admission_column_and_rpc_columns():
    """The 0038 admission shape lives in the folded baseline now: init.sql's
    spec_generation_reviews table and the review RPC both carry the columns.
    (Migrations 0001–0041 were deleted once init.sql folded them; they live
    in git history.)"""
    root = Path(__file__).resolve().parents[3] / "supabase"
    sql = (root / "init.sql").read_text()
    table = re.search(r"CREATE TABLE public\.spec_generation_reviews \((.*?)\n\);",
                      sql, re.S).group(1)
    assert "admission jsonb" in table
    for column in ("bundle_fingerprint", "run_revision", "engine_version", "admission"):
        assert f"sr.{column}" in sql


# ── staleness and the latest review ───────────────────────────────────────

def test_a_report_on_a_review_of_another_bundle_is_stale(client, user_token, monkeypatch):
    from backend.service.routes import account
    current = dict(RECORD, user_id="user-uuid-1")
    rows = [{"id": "new", "verdict": "pass", "bundle_fingerprint": record_fingerprint(current)},
            {"id": "old", "verdict": "pass", "bundle_fingerprint": "f" * 64}]
    monkeypatch.setattr(account._reviews, "list_for_record_owner", lambda rid, uid: rows)
    monkeypatch.setattr(account._gens, "get_for_user", lambda rid, uid: current)
    r = client.get("/api/account/generations/rec-1/reviews", headers=_auth(user_token))
    assert [(x["id"], x["bundle_current"]) for x in r.json()["items"]] == [
        ("new", True), ("old", False)]


def test_newer_than_breaks_a_timestamp_tie_by_id():
    rows = [{"id": "a", "created_at": "2026-10-01T00:00:00Z"},
            {"id": "b", "created_at": "2026-10-01T00:00:00Z"},
            {"id": "c", "created_at": "2026-09-30T00:00:00Z"}]
    assert [r["id"] for r in R.newer_than(rows, "a")] == ["b"]
    assert R.newer_than(rows, "b") == []
    assert R.newer_than(rows, "missing") == rows


def test_latest_pr_report_is_the_bots_last_push():
    rows = [
        {"id": "3", "created_at": "2026-10-03T00:00:00Z",
         "admission": {"report": {"mode": "dry_run"}}},
        {"id": "2", "created_at": "2026-10-02T00:00:00Z",
         "admission": {"report": {"mode": "pr", "pr": {"head_sha": "h2"}}}},
        {"id": "1", "created_at": "2026-10-01T00:00:00Z",
         "admission": {"report": {"mode": "pr", "pr": {"head_sha": "h1"}}}},
    ]
    assert R.latest_pr_report(rows)["pr"]["head_sha"] == "h2"
    assert R.latest_pr_report([]) is None


# ── a PASS starts the dry run; the dry run never changes the PASS ─────────

def _review_job(monkeypatch, verdict):
    import backend.spec_worker as sw
    from backend.service.routes import spec as routes
    record = dict(RECORD, user_id="user-uuid-1", spec_yaml="match: {model_id: acme/Model}\n")
    monkeypatch.setattr(routes, "_normalize_record_bundle", lambda *a: object())
    monkeypatch.setattr(sw, "prepare_review", lambda *a, **kw: {"snapshot_data": {}})

    def stream(model_id, prep, run_key=None):
        prep["review_row"] = {"id": "rev-7", "verdict": verdict}
        yield 'data: {"type": "done", "success": true}\n\n'
    monkeypatch.setattr(sw, "stream_review", stream)
    launched = []
    monkeypatch.setattr(routes, "launch_admission",
                        lambda rec, review_id: launched.append((rec["id"], review_id)))
    monkeypatch.setattr(sw, "acquire", lambda k: "ok")
    jobs = []
    monkeypatch.setattr(sw, "start_job", lambda k, fn, **kw: jobs.append(fn))
    routes.launch_review(record, reviewer_id="user-uuid-1", reviewer_role="user")
    list(jobs[0]())
    return launched


def test_a_pass_starts_the_admission_of_that_review(monkeypatch):
    assert _review_job(monkeypatch, "pass") == [(RECORD["id"], "rev-7")]


@pytest.mark.parametrize("verdict", ["no_pass", "error"])
def test_no_admission_follows_anything_but_a_pass(monkeypatch, verdict):
    assert _review_job(monkeypatch, verdict) == []


@pytest.fixture
def dry(monkeypatch, httpx_mock):
    monkeypatch.setenv("HA_GITHUB_TOKEN", "tok")
    monkeypatch.setenv("HA_GITHUB_REPO", "shenh10/HuggingArch")
    for var in ("HA_GITHUB_APP_ID", "HA_GITHUB_APP_PRIVATE_KEY", "HA_GITHUB_APP_INSTALLATION_ID"):
        monkeypatch.delenv(var, raising=False)
    store = Store(RECORD)
    store.install(monkeypatch)
    admitter = FakeAdmitter()
    monkeypatch.setattr(adm.AdmissionTree, "runner", admitter)
    monkeypatch.setattr(spec_pr, "_prepare", lambda record, source, **kw: spec_pr._github.SpecPR(
        branch="b", files={"backend/arch/models/acme__Model.yaml": "x\n"}, title="t",
        body="b", commit_prefix="p"))
    deployed = (adm.REPO_ROOT / "uv.lock").read_text()
    return httpx_mock, store, admitter, deployed


def _dry(store, revision="d" * 40):
    job = spec_pr.DryRunAdmission(store.record, "rev-1")
    job.orchestrator = revision
    return list(job.run())


def test_dry_run_measures_main_when_its_lock_is_the_deployed_one(dry):
    httpx_mock, store, admitter, deployed = dry
    gh = FakeGitHub({"uv.lock": deployed, "README.md": "x\n",
                     "tests/golden/spec_match_fingerprints.json": "{}\n",
                     "tests/golden/pins/tp_split_decisions.json": "{}\n"})
    gh.install(httpx_mock)
    events = _dry(store)
    assert events[-1]["success"] and events[-1]["admission_status"] == "ready"
    (run,) = admitter.runs
    assert run["mode"] == adm.DRY_RUN and run["base"] == gh.branches["main"]
    envelope = store.reviews[0]["admission"]
    assert envelope["job"] == "done" and envelope["mode"] == "dry_run"
    assert envelope["report"]["base"]["sha"] == gh.branches["main"]
    # a dry run writes nothing anywhere: no branch, no PR
    assert set(gh.branches) == {"main"} and not gh.pulls


def test_dry_run_falls_back_to_the_deployments_commit(dry):
    httpx_mock, store, admitter, deployed = dry
    gh = FakeGitHub({"uv.lock": "a newer lock\n", "tests/golden/spec_match_fingerprints.json": "{}\n",
                     "tests/golden/pins/tp_split_decisions.json": "{}\n"})
    deployment = gh.commit({"uv.lock": deployed,
                            "tests/golden/spec_match_fingerprints.json": "{}\n",
                            "tests/golden/pins/tp_split_decisions.json": "{}\n"},
                           parents=[], message="deployed")
    gh.install(httpx_mock)
    _dry(store, revision=deployment)
    (run,) = admitter.runs
    assert run["base"] == deployment
    assert gh.tarballs[-1] == deployment


def test_dry_run_is_unavailable_when_the_deployment_revision_is_unknown(dry):
    httpx_mock, store, admitter, _deployed = dry
    gh = FakeGitHub({"uv.lock": "a newer lock\n"})
    gh.install(httpx_mock)
    events = _dry(store, revision="unknown")
    assert admitter.runs == []
    report = store.reviews[0]["admission"]["report"]
    assert report["status"] == "unavailable" and report["base"]["kind"] == "deployment"
    assert "revision is unknown" in report["gates"]["G0"][0]
    assert events[-1]["admission_status"] == "unavailable"


def test_a_blocked_admission_leaves_the_pass_standing(dry):
    from backend.service.routes import account
    httpx_mock, store, admitter, deployed = dry
    admitter.status, admitter.blocked_kind = "blocked", "candidate_or_rule"
    gh = FakeGitHub({"uv.lock": deployed, "tests/golden/spec_match_fingerprints.json": "{}\n",
                     "tests/golden/pins/tp_split_decisions.json": "{}\n"})
    gh.install(httpx_mock)
    before = json.loads(json.dumps(store.reviews))
    _dry(store)
    (row,) = store.reviews
    assert row["admission"]["report"]["status"] == "blocked"
    assert {k: v for k, v in row.items() if k != "admission"} == \
        {k: v for k, v in before[0].items() if k != "admission"}
    assert account._latest_review_is_pass(store.reviews, store.record)


def test_dry_run_without_github_is_unavailable(dry, monkeypatch):
    _httpx_mock, store, _admitter, _deployed = dry
    monkeypatch.delenv("HA_GITHUB_TOKEN")
    _dry(store)
    report = store.reviews[0]["admission"]["report"]
    assert report["status"] == "unavailable" and "GitHub is not configured" in report["errors"][0]


def test_dry_run_that_crashes_leaves_a_failed_envelope(dry, monkeypatch):
    _httpx_mock, store, _admitter, _deployed = dry
    monkeypatch.setattr(spec_pr.DryRunAdmission, "base",
                        lambda self: (_ for _ in ()).throw(RuntimeError("disk full")))
    events = _dry(store)
    assert store.reviews[0]["admission"]["job"] == "failed"
    assert "disk full" in store.reviews[0]["admission"]["error"]
    assert events[-1]["success"] is False


def test_an_admitter_without_a_report_is_unavailable(dry, monkeypatch):
    httpx_mock, store, _admitter, deployed = dry
    gh = FakeGitHub({"uv.lock": deployed})
    gh.install(httpx_mock)
    monkeypatch.setattr(adm.AdmissionTree, "runner", staticmethod(
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "ModuleNotFoundError: tools")))
    _dry(store)
    report = store.reviews[0]["admission"]["report"]
    assert report["status"] == "unavailable" and "ModuleNotFoundError" in report["errors"][0]


# ── what a report lets Open PR do ─────────────────────────────────────────

@pytest.mark.parametrize("status,kind,override,opens,draft,code", [
    ("ready", None, False, True, False, "ready"),
    ("needs_review", None, False, True, True, "needs_review"),
    ("blocked", "candidate_or_rule", False, False, False, "blocked"),
    ("blocked", "candidate_or_rule", True, True, True, "draft_anyway"),
    ("blocked", "base", False, False, False, "blocked_base"),
    ("blocked", "base", True, False, False, "blocked_base"),
    ("already_admitted", None, False, False, False, "already_admitted"),
    ("unavailable", None, True, False, False, "unavailable"),
])
def test_gate(status, kind, override, opens, draft, code):
    gate = adm.Gate.of({"status": status, "blocked_kind": kind, "gates": {}},
                       admin_override=override)
    assert (gate.opens, gate.draft, gate.code) == (opens, draft, code)


def test_pr_section_lists_written_pending_base_and_info():
    report = {
        "status": "needs_review", "base": {"sha": "a" * 40}, "env": {"python": "3.13"},
        "candidate": {"bundle_sha256": "b" * 64}, "report_sha256": "c" * 64,
        "items": [
            {"id": "P4", "title": "checkpoint", "key_states": {
                "EXACT/x": {"state": "written", "waits_on": []}}},
            {"id": "P2", "title": "SOL", "expected_failing_tests": ["tests/t.py::sol"],
             "key_states": {"x.json|single": {"state": "withheld", "waits_on": ["G4"]}}},
        ],
        "checks": [], "base_notes": [{"id": "P10", "detail": ["Kimi drift"]}],
        "decisions": [{"id": "D1", "state": "pending", "detail": ["visual"],
                       "suggestion": {"nodes": {"merger": "split"}}},
                      {"id": "D5", "state": "info", "detail": ["not listed"]}],
    }
    text = adm.pr_section(report)
    head = json.loads(text.split("<!-- admission ", 1)[1].split(" -->", 1)[0])
    assert head == {"base_sha": "a" * 40, "bundle_sha256": "b" * 64, "env": {"python": "3.13"},
                    "status": "needs_review", "report_sha256": "c" * 64}
    assert "P4 checkpoint: `EXACT/x`" in text
    assert "D1 (pending): visual" in text and '"merger": "split"' in text
    assert "`tests/t.py::sol`" in text and "wait on G4" in text
    assert "P10: Kimi drift" in text and "D5: not listed" in text


def test_an_archive_member_outside_the_tree_is_refused(tmp_path):
    import io
    import tarfile
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as tar:
        info = tarfile.TarInfo("repo/../../escape.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(tarfile.FilterError):
        adm.extract_archive(raw.getvalue(), tmp_path / "t")
    assert not (tmp_path / "escape.txt").exists()


def test_overlay_refuses_a_path_outside_the_tree():
    with pytest.raises(ValueError):
        adm._safe_relative("../etc/passwd")
    with pytest.raises(ValueError):
        adm._safe_relative("/abs")


def test_routes_start_jobs_and_poll_them(client, user_token, monkeypatch):
    from backend.service.routes import account
    record = dict(RECORD, user_id="user-uuid-1", model_id=MODEL)
    monkeypatch.setattr(account._gens, "get", lambda rid: record)
    monkeypatch.setattr(account, "_pass_review", lambda *a: {"id": "rev-1"})
    monkeypatch.setattr(account, "superseded_by", lambda rec: None)
    monkeypatch.setattr(account._github, "enabled", lambda: True)
    monkeypatch.setattr(account._github, "check_spec_bundle", lambda *a: None)
    monkeypatch.setattr(account, "_contributor_for", lambda uid: None)
    calls = []
    monkeypatch.setattr(account, "launch_spec_pr", lambda rec, review, **kw: calls.append(kw)
                        or {"status": "started", "run_key": "pr|user-uuid-1|rec"})
    monkeypatch.setattr(account._auth, "is_admin", lambda uid: False)
    r = client.post(f"/api/account/generations/{record['id']}/create-pr",
                    json={"draft_anyway": True}, headers=_auth(user_token))
    assert r.json()["status"] == "started"
    # draft_anyway is an admin's; for an owner it is ignored
    assert calls[-1]["admin_override"] is False and calls[-1]["update"] is False
    monkeypatch.setattr(account._auth, "is_admin", lambda uid: True)
    client.post(f"/api/account/generations/{record['id']}/create-pr",
                json={"draft_anyway": True}, headers=_auth(user_token))
    assert calls[-1]["admin_override"] is True
    seen = {}
    monkeypatch.setattr(account.spec_worker, "job_status",
                        lambda key, cursor: seen.setdefault("key", key) or {"state": "idle"})
    client.get(f"/api/account/generations/{record['id']}/pr-status", headers=_auth(user_token))
    assert seen["key"] == f"pr|user-uuid-1|{record['id']}"


def test_an_admitter_that_runs_too_long_is_unavailable(dry, monkeypatch):
    httpx_mock, store, _admitter, deployed = dry
    FakeGitHub({"uv.lock": deployed}).install(httpx_mock)

    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(adm.AdmissionTree, "runner", staticmethod(slow))
    _dry(store)
    report = store.reviews[0]["admission"]["report"]
    assert report["status"] == "unavailable" and "longer than" in report["errors"][0]


def test_an_envelope_the_review_row_refuses_lands_in_the_jobs_log_row(httpx_mock):
    """The real path, 0038 missing: the PATCH of the review row fails with
    PGRST204, the envelope is written onto the job's spec_generation_logs
    row, and it reads back whole."""
    from backend.service import spec_generation_logs as logs
    logs_url = re.compile(re.escape(f"{TEST_SUPABASE_URL}/rest/v1/spec_generation_logs") + r"(\?.*)?")
    httpx_mock.add_response(method="PATCH", url=_REVIEWS, status_code=400,
                            text='{"code":"PGRST204","message":"no admission column"}')
    httpx_mock.add_response(method="PATCH", url=logs_url, status_code=204)
    envelope = adm.envelope("done", mode=adm.PR, report={
        "status": "ready", "report_sha256": "a" * 64, "pr": {"head_sha": "b" * 40}})
    record = {"id": "rec-1", "user_id": "u1", "model_id": MODEL}
    logs.bind_run_row("log-1")  # the PR job thread's own durability row
    try:
        assert spec_pr.persist("rev-1", envelope, record=record,
                               run_key="pr|u1|rec-1") is False
    finally:
        logs.bind_run_row(None)
    patched = [r for r in httpx_mock.get_requests()
               if r.method == "PATCH" and "spec_generation_logs" in str(r.url)]
    (req,) = patched
    assert req.url.params["id"] == "eq.log-1"
    assert spec_pr.report_from_log(json.loads(req.content)["log"]) == envelope
    assert logs._MAX_LOG_CHARS > 200_000  # a report (tens of KB) is never truncated
