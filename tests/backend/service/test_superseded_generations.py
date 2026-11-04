"""A private generation whose model has a library spec is superseded.

`service_analyze` serves the library spec whenever one exists, so such a
record is never used for computation. The account list archives it (default
view without it, `?superseded=true` with only it), and Review, Agent Fix,
Refine and Open PR refuse it; Delete keeps its existing rules. Supersession
asks the same lookup analyze's `SpecLoader.match` does.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import backend.service.routes.account as _account
import backend.service.routes.spec as _spec
import backend.spec_worker as _sw
from backend.arch.loader import SpecLoader, get_default_loader
from backend.arch.snapshots import load_config
from backend.service import agent_credentials as _creds
from backend.service import spec_generation_logs as _genlogs
from backend.service import spec_generation_reviews as _reviews
from backend.service import spec_generations as _gens

from .conftest import mock_admins_yes

LIBRARY_ID = "acme/InLibrary"
LIBRARY_SPEC = "acme__InLibrary"
PRIVATE_ID = "acme/NotInLibrary"


def _auth(tok):
    return {"Authorization": f"Bearer {tok}"}


def _rec(rid, model_id, status="generated", **kw):
    return {"id": rid, "model_id": model_id, "user_id": "user-uuid-1",
            "status": status, "created_at": "2026-09-20T00:00:00+00:00",
            "spec_yaml": f"match: {{model_id: [{model_id}]}}\n",
            "new_component_yamls": [], **kw}


class _Library:
    """The shared loader's identity lookup, with one library model."""

    def spec_for_model_id(self, model_id):
        return SimpleNamespace(name=LIBRARY_SPEC) if model_id == LIBRARY_ID else None


@pytest.fixture(autouse=True)
def library(monkeypatch):
    monkeypatch.setattr(_spec, "get_default_loader", _Library)


@pytest.fixture(autouse=True)
def _fresh_quota():
    _spec._quota_hits.clear()
    yield
    _spec._quota_hits.clear()


@pytest.fixture
def jobs(monkeypatch):
    """Capture started jobs; the agent never runs."""
    started = []
    monkeypatch.setattr(_sw, "acquire", lambda k: "ok")
    monkeypatch.setattr(_sw, "release", lambda k: None)
    monkeypatch.setattr(_sw, "start_job", lambda k, fn, **kw: started.append(k))
    monkeypatch.setattr(_creds, "resolve_env",
                        lambda uid, **kw: {"agent_cli": "claude", "env": {}})
    return started


# ---------------------------------------------------------------------------
# The list
# ---------------------------------------------------------------------------
@pytest.fixture
def listing(monkeypatch):
    monkeypatch.setattr(_account, "_release_dates_for", lambda ids: {})
    monkeypatch.setattr(_reviews, "latest_verdicts_for_records", lambda ids: {})
    monkeypatch.setattr(_genlogs, "list_for_user", lambda uid, limit=50: [
        {"id": "l-orphan", "model_id": PRIVATE_ID, "success": False,
         "created_at": "2026-09-19T00:00:00+00:00"},
        {"id": "l-library", "model_id": LIBRARY_ID, "success": False,
         "created_at": "2026-09-18T00:00:00+00:00"}])
    monkeypatch.setattr(_gens, "list_for_user", lambda uid: [
        _rec("g-private", PRIVATE_ID),
        _rec("g-superseded", LIBRARY_ID, status="contributed",
             pr_url="https://github.com/o/r/pull/1"),
        _rec("g-merged", LIBRARY_ID, status="merged",
             pr_url="https://github.com/o/r/pull/2"),
    ])
    synced = []
    monkeypatch.setattr(_gens, "sync_pr_status", lambda rec: synced.append(rec["id"]))
    return synced


def test_default_list_excludes_superseded(client, user_token, listing):
    r = client.get("/api/account/generations", headers=_auth(user_token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert {it["id"]: it["superseded_by"] for it in body["items"]} == {
        "g-private": None, "g-merged": None}
    assert [l["id"] for l in body["orphan_logs"]] == ["l-orphan"]


def test_list_names_the_library_spec_of_a_merged_record(client, user_token, listing):
    """A merged record whose model the library serves is in the library, not
    superseded: the list names that spec so the page can say so."""
    r = client.get("/api/account/generations", headers=_auth(user_token))
    assert r.status_code == 200, r.text
    assert {it["id"]: it["library_spec"] for it in r.json()["items"]} == {
        "g-private": None, "g-merged": LIBRARY_SPEC}
    r = client.get("/api/account/generations?superseded=true", headers=_auth(user_token))
    assert [(it["id"], it["library_spec"]) for it in r.json()["items"]] == [
        ("g-superseded", LIBRARY_SPEC)]


def test_admin_list_and_detail_carry_library_standing(client, admin_token, monkeypatch, httpx_mock):
    import backend.service.routes.admin as _admin
    mock_admins_yes(httpx_mock, "admin-uuid-1")
    rows = {"get_spec_generations": [
                _rec("g-merged", LIBRARY_ID, status="merged"),
                _rec("g-superseded", LIBRARY_ID),
                _rec("g-private", PRIVATE_ID)],
            "get_spec_generations_count": 3,
            "get_spec_generation_detail": [_rec("g-merged", LIBRARY_ID, status="merged")]}
    monkeypatch.setattr(_admin, "_call_admin_rpc", lambda fn, args, user_jwt: rows[fn])
    monkeypatch.setattr(_gens, "sync_pr_status", lambda rec: None)
    r = client.get("/api/admin/spec-generations", headers=_auth(admin_token))
    assert r.status_code == 200, r.text
    assert [(it["id"], it["library_spec"], it["superseded_by"]) for it in r.json()["items"]] == [
        ("g-merged", LIBRARY_SPEC, None),
        ("g-superseded", LIBRARY_SPEC, LIBRARY_SPEC),
        ("g-private", None, None)]
    r = client.get("/api/admin/spec-generations/g-merged", headers=_auth(admin_token))
    assert r.status_code == 200, r.text
    assert (r.json()["library_spec"], r.json()["superseded_by"]) == (LIBRARY_SPEC, None)


def test_superseded_filter_lists_only_archived(client, user_token, listing):
    r = client.get("/api/account/generations?superseded=true",
                   headers=_auth(user_token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert [(it["id"], it["superseded_by"]) for it in body["items"]] == [
        ("g-superseded", LIBRARY_SPEC)]
    # A failed run's log of a library model is archived with its records, so
    # the model is not also listed in the default view under "run failed".
    assert [(l["id"], l["superseded_by"]) for l in body["orphan_logs"]] == [
        ("l-library", LIBRARY_SPEC)]
    # An open PR is only listed, never closed: the existing poll still reads it.
    assert listing == ["g-superseded"]


def test_detail_names_the_superseding_spec(client, user_token, monkeypatch):
    monkeypatch.setattr(_gens, "get_for_user",
                        lambda rid, uid: _rec("g-superseded", LIBRARY_ID))
    r = client.get("/api/account/generations/g-superseded", headers=_auth(user_token))
    assert r.status_code == 200
    assert r.json()["superseded_by"] == LIBRARY_SPEC
    assert r.json()["library_spec"] == LIBRARY_SPEC


# ---------------------------------------------------------------------------
# Actions on a superseded record
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("action", ["review", "fix", "create-pr"])
def test_actions_refused_on_superseded(client, user_token, monkeypatch, jobs, action):
    monkeypatch.setattr(_gens, "get", lambda rid: _rec("g-superseded", LIBRARY_ID))
    monkeypatch.setattr(_reviews, "list_for_record_owner",
                        lambda rid, uid: [{"verdict": "pass"}])
    r = client.post(f"/api/account/generations/g-superseded/{action}",
                    headers=_auth(user_token))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "superseded"
    assert body["superseded_by"] == LIBRARY_SPEC
    assert LIBRARY_SPEC in body["message"]
    assert jobs == []


def test_admin_review_refused_on_superseded(client, admin_token, monkeypatch, jobs, httpx_mock):
    mock_admins_yes(httpx_mock, "admin-uuid-1")
    monkeypatch.setattr(_gens, "get", lambda rid: _rec("g-superseded", LIBRARY_ID))
    r = client.post("/api/admin/spec-generations/g-superseded/review",
                    headers=_auth(admin_token))
    assert r.status_code == 200
    assert r.json()["status"] == "superseded"
    assert jobs == []


def test_refine_refused_for_a_library_model(client, user_token, monkeypatch, jobs):
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    monkeypatch.setattr(_sw, "get_agent_binary", lambda: "claude")
    monkeypatch.setattr("shutil.which", lambda _b: "/usr/bin/claude")
    monkeypatch.setattr(_spec._auth, "is_admin", lambda uid: False)
    monkeypatch.setattr("backend.service.spec_generations.serving_generation",
                        lambda uid, mid: pytest.fail("seeded from a superseded record"))
    r = client.post("/api/spec/refine", headers=_auth(user_token), json={
        "model_id": LIBRARY_ID, "spec_storage_bytes": 10,
        "hf_storage_bytes": 12, "storage_diff": 2})
    assert r.status_code == 200
    assert r.json()["status"] == "superseded"
    assert jobs == []


def test_analyze_library_answer_does_not_offer_refine(client, user_token, monkeypatch):
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    monkeypatch.setattr("backend.analyzer.analyze_model",
                        lambda **kw: {"success": True})
    monkeypatch.setattr("backend.api.routes.analyze.build_response",
                        lambda req, result, ms, overlay_dir=None: {"success": True})
    r = client.post("/api/spec/analyze", json={"model_id": LIBRARY_ID},
                    headers=_auth(user_token))
    assert r.json()["refine_offered"] is False


def test_delete_still_allowed(client, user_token, monkeypatch):
    monkeypatch.setattr(_gens, "list_for_user",
                        lambda uid: [_rec("g-superseded", LIBRARY_ID)])
    monkeypatch.setattr(_genlogs, "list_for_user", lambda uid, limit=50: [])
    seen = []
    monkeypatch.setattr(_gens, "delete_for_user",
                        lambda rid, uid: seen.append(rid) or True)
    r = client.delete("/api/account/generations/g-superseded", headers=_auth(user_token))
    assert r.status_code == 200 and r.json()["deleted"] is True
    assert seen == ["g-superseded"]


# ---------------------------------------------------------------------------
# A record with no library spec is unchanged
# ---------------------------------------------------------------------------
def test_review_starts_when_not_superseded(client, user_token, monkeypatch, jobs):
    monkeypatch.setattr(_gens, "get", lambda rid: _rec("g-private", PRIVATE_ID))
    r = client.post("/api/account/generations/g-private/review",
                    headers=_auth(user_token))
    assert r.status_code == 200 and r.json()["status"] == "started"
    assert jobs == ["review|user-uuid-1|g-private"]


def test_fix_without_review_keeps_its_answer(client, user_token, monkeypatch, jobs):
    monkeypatch.setattr(_gens, "get", lambda rid: _rec("g-private", PRIVATE_ID))
    monkeypatch.setattr(_reviews, "list_for_record_owner", lambda rid, uid: [])
    monkeypatch.setattr(_account, "_opens_on_refusals", lambda rec: False)
    r = client.post("/api/account/generations/g-private/fix", headers=_auth(user_token))
    assert r.json()["status"] == "no_review"


# ---------------------------------------------------------------------------
# Supersession and analyze precedence share one lookup
# ---------------------------------------------------------------------------
def test_match_answers_through_the_identity_lookup(monkeypatch):
    """`match` (what analyze runs) picks its spec only through
    `spec_for_model_id` (what supersession runs): stub the lookup and match
    follows it."""
    loader = get_default_loader()
    monkeypatch.setattr(SpecLoader, "spec_for_model_id", lambda self, mid: None)
    assert loader.match(load_config("moonshotai/Kimi-K3"),
                        model_id="moonshotai/Kimi-K3") is None


@pytest.mark.parametrize("model_id", [
    "moonshotai/Kimi-K3", "deepseek-ai/DeepSeek-V4-Flash", PRIVATE_ID])
def test_supersession_agrees_with_analyze_precedence(monkeypatch, model_id):
    monkeypatch.setattr(_spec, "get_default_loader", get_default_loader)
    config = load_config(model_id) or {}
    served = get_default_loader().match(config, model_id=model_id)
    library_spec = _spec.library_spec_for(model_id)
    assert (library_spec is not None) == (served is not None)
    if served is not None:
        assert library_spec == served.spec.name


def test_an_unloadable_library_spec_still_supersedes(monkeypatch):
    """Analyze answers a library spec that fails to load with a spec error and
    never serves the private record, so the record is superseded by it."""
    from backend.arch.loader import SpecLoadError

    class _Broken:
        def spec_for_model_id(self, model_id):
            raise SpecLoadError("broken", path="backend/arch/models/acme__Broken.yaml")
    monkeypatch.setattr(_spec, "get_default_loader", _Broken)
    assert _spec.superseded_by({"model_id": "acme/Broken", "status": "generated"}) == "acme__Broken"


def test_refine_of_a_library_model_is_refused_before_credentials_and_quota(
        client, user_token, monkeypatch, jobs):
    """A user with no stored key, or no quota left, still hears the real
    reason: the model is in the library, so refining it can never be served."""
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    monkeypatch.setattr(_spec._auth, "is_admin", lambda uid: False)
    monkeypatch.setattr(_spec, "_preflight",
                        lambda *a, **k: pytest.fail("credentials/quota checked first"))
    r = client.post("/api/spec/refine", headers=_auth(user_token), json={
        "model_id": LIBRARY_ID, "spec_storage_bytes": 10,
        "hf_storage_bytes": 12, "storage_diff": 2})
    assert r.status_code == 200
    assert r.json()["status"] == "superseded"
    assert jobs == []
