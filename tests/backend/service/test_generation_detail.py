"""User-facing spec-generation detail endpoint + retirement of the admin
content-mutation endpoints.

`GET /api/account/generations/{id}` lets a user read the full payload (spec_yaml
/ components) of their OWN generation — the list route omits it. The admin
create-pr / update-pr / fix endpoints were removed: fixing a spec and opening or
updating its PR are the contributor's own owner-scoped actions, never the
admin's (an admin must not rewrite another user's submission).
"""
from __future__ import annotations

import httpx
import pytest

from backend.service import spec_generations as _gens


def _auth(tok):
    return {"Authorization": f"Bearer {tok}"}


def test_get_my_generation_anonymous_401(client):
    r = client.get("/api/account/generations/rec-1")
    assert r.status_code == 401


def test_get_my_generation_owner_200(client, user_token, monkeypatch):
    rec = {
        "id": "rec-1", "user_id": "user-uuid-1", "model_id": "acme/Model",
        "spec_yaml": "model: acme/Model\n", "snapshot_json": {},
        "new_component_yamls": [{"path": "components/x.yaml", "body": "kind: x\n"}],
        "agent_cli": "claude", "status": "generated", "pr_url": None,
    }
    seen = {}
    monkeypatch.setattr(_gens, "get_for_user",
                        lambda rid, uid: (seen.update(rid=rid, uid=uid) or rec))
    r = client.get("/api/account/generations/rec-1", headers=_auth(user_token))
    body = r.json()
    assert r.status_code == 200
    assert body["spec_yaml"] == "model: acme/Model\n"
    assert body["new_component_yamls"][0]["path"] == "components/x.yaml"
    # Scoped by the caller's own id — never a caller-supplied one.
    assert seen == {"rid": "rec-1", "uid": "user-uuid-1"}


def test_get_my_generation_not_owner_or_missing_404(client, user_token, monkeypatch):
    # get_for_user returns None for both "no such record" and "not yours" — the
    # route must not distinguish (no existence leak to a non-owner).
    monkeypatch.setattr(_gens, "get_for_user", lambda rid, uid: None)
    r = client.get("/api/account/generations/rec-1", headers=_auth(user_token))
    assert r.status_code == 404


def test_get_my_generation_store_timeout_503(client, user_token, monkeypatch):
    """A Supabase blip is not a missing row — the drawer was showing
    'No such generation' for archived records that were still in the table."""
    def boom(rid, uid):
        raise _gens.SpecGenerationsUnavailable("The read operation timed out")
    monkeypatch.setattr(_gens, "get_for_user", boom)
    r = client.get("/api/account/generations/rec-1", headers=_auth(user_token))
    assert r.status_code == 503
    assert "No such generation" not in r.json()["detail"]


def test_get_for_user_timeout_is_unavailable(httpx_mock):
    httpx_mock.add_exception(httpx.ReadTimeout("The read operation timed out"))
    with pytest.raises(_gens.SpecGenerationsUnavailable):
        _gens.get_for_user("rec-1", "user-uuid-1")


def test_admin_content_mutation_endpoints_removed(client, admin_token):
    """Admin create-pr / update-pr / fix are gone (404). Admin keeps only
    read / review / reject over a contributor's submission."""
    for suffix in ("create-pr", "update-pr", "fix"):
        r = client.post(f"/api/admin/spec-generations/rec-1/{suffix}",
                        headers=_auth(admin_token))
        assert r.status_code == 404, f"{suffix} should be removed, got {r.status_code}"


def test_nominate_contribute_endpoint_removed(client, user_token):
    """The Inspector-only nominate/contribute route is gone. Opening a PR is
    POST .../create-pr after a PASS review."""
    r = client.post("/api/account/generations/rec-1/contribute",
                    headers=_auth(user_token))
    assert r.status_code == 404
