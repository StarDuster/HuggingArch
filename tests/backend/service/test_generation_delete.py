"""DELETE /api/account/generations/{id} — owner-scoped record delete.

Records with an open or merged PR stay: the scoped PostgREST DELETE only
matches `generated` / `rejected` / `closed`, or `contributed` with no
`pr_url`, on top of the owner filter. Reviews of the record cascade in the DB (0030)."""
from __future__ import annotations

import re

from backend.service import spec_generations as _gens
from .conftest import TEST_SUPABASE_URL

_GENS = re.compile(re.escape(f"{TEST_SUPABASE_URL}/rest/v1/spec_generations") + r"(\?.*)?")


def _auth(tok):
    return {"Authorization": f"Bearer {tok}"}


class TestDeleteForUser:
    def test_delete_sends_owner_and_status_scope(self, httpx_mock):
        httpx_mock.add_response(method="DELETE", url=_GENS, json=[{"id": "r1"}])
        assert _gens.delete_for_user("r1", "user-uuid-1") is True
        req = httpx_mock.get_requests(method="DELETE")[0]
        url = str(req.url)
        assert "id=eq.r1" in url
        assert "user_id=eq.user-uuid-1" in url
        assert req.url.params["or"] == (
            "(status.not.in.(contributed,merged),and(status.eq.contributed,pr_url.is.null))")

    def test_delete_false_when_nothing_matched(self, httpx_mock):
        """Wrong owner / protected status / missing record all look the same:
        an empty return=representation — never a reported success."""
        httpx_mock.add_response(method="DELETE", url=_GENS, json=[])
        assert _gens.delete_for_user("r1", "user-uuid-1") is False

    def test_delete_false_on_http_error(self, httpx_mock):
        httpx_mock.add_response(method="DELETE", url=_GENS, status_code=400, text="bad")
        assert _gens.delete_for_user("r1", "user-uuid-1") is False


class TestDeleteRoute:
    def test_anonymous_401(self, client):
        assert client.delete("/api/account/generations/rec-1").status_code == 401

    def test_owner_delete_ok(self, client, user_token, monkeypatch):
        seen = {}
        def fake_delete(rid, uid):
            seen["args"] = (rid, uid)
            return True
        monkeypatch.setattr(_gens, "delete_for_user", fake_delete)
        r = client.delete("/api/account/generations/rec-1", headers=_auth(user_token))
        assert r.status_code == 200 and r.json()["deleted"] is True
        assert seen["args"] == ("rec-1", "user-uuid-1")

    def test_not_found_or_protected_404(self, client, user_token, monkeypatch):
        monkeypatch.setattr(_gens, "delete_for_user", lambda rid, uid: False)
        r = client.delete("/api/account/generations/rec-1", headers=_auth(user_token))
        assert r.status_code == 404
        assert "PR" in r.json()["detail"]

    def test_delete_cascades_paired_run_log(self, client, user_token, monkeypatch):
        """A spec and its producing run log are bound: deleting the spec drops
        the paired log too, so no orphan is left behind."""
        from backend.service import spec_generation_logs as _genlogs
        monkeypatch.setattr(_gens, "list_for_user", lambda uid: [
            {"id": "rec-1", "model_id": "m", "created_at": "2026-01-01T00:00:00+00:00"},
        ])
        monkeypatch.setattr(_genlogs, "list_for_user", lambda uid, limit=1000: [
            {"id": "log-9", "model_id": "m", "created_at": "2025-12-31T23:59:00+00:00",
             "success": True, "record_id": "rec-1"},
        ])
        monkeypatch.setattr(_gens, "delete_for_user", lambda rid, uid: True)
        seen = {}
        def fake_log_delete(lid, uid):
            seen["args"] = (lid, uid)
            return True
        monkeypatch.setattr(_genlogs, "delete_for_user", fake_log_delete)
        r = client.delete("/api/account/generations/rec-1", headers=_auth(user_token))
        assert r.status_code == 200 and r.json()["deleted"] is True
        assert seen["args"] == ("log-9", "user-uuid-1")

    def test_delete_cascades_fix_and_review_logs(self, client, user_token, monkeypatch):
        """Fix/review transcripts hang off the spec the same way the produce
        log does — deleting the spec drops all of them."""
        from backend.service import spec_generation_logs as _genlogs
        monkeypatch.setattr(_gens, "list_for_user", lambda uid: [
            {"id": "rec-1", "model_id": "m", "created_at": "2026-01-01T00:00:00+00:00"},
        ])
        monkeypatch.setattr(_genlogs, "list_for_user", lambda uid, limit=1000: [
            {"id": "log-gen", "model_id": "m",
             "created_at": "2026-01-01T00:00:01+00:00", "success": True,
             "record_id": "rec-1"},
            {"id": "log-fix", "model_id": "m",
             "created_at": "2026-01-01T01:00:00+00:00", "success": True,
             "run_key": "fix|user-uuid-1|rec-1", "kind": "fix"},
            {"id": "log-rev", "model_id": "m",
             "created_at": "2026-01-01T02:00:00+00:00", "success": True,
             "run_key": "review|user-uuid-1|rec-1", "kind": "review"},
        ])
        monkeypatch.setattr(_gens, "delete_for_user", lambda rid, uid: True)
        deleted = []
        monkeypatch.setattr(_genlogs, "delete_for_user",
                            lambda lid, uid: deleted.append(lid) or True)
        r = client.delete("/api/account/generations/rec-1", headers=_auth(user_token))
        assert r.status_code == 200 and r.json()["deleted"] is True
        assert set(deleted) == {"log-gen", "log-fix", "log-rev"}

    def test_delete_leaves_another_runs_log(self, client, user_token, monkeypatch):
        """A failed run of the same model after the record wrote no record;
        its transcript is the user's own row and stays when the record goes."""
        from backend.service import spec_generation_logs as _genlogs
        monkeypatch.setattr(_gens, "list_for_user", lambda uid: [
            {"id": "rec-1", "model_id": "m", "created_at": "2026-01-01T00:00:00+00:00"},
        ])
        monkeypatch.setattr(_genlogs, "list_for_user", lambda uid, limit=1000: [
            {"id": "log-fail", "model_id": "m",
             "created_at": "2026-01-01T00:00:05+00:00", "success": False},
        ])
        monkeypatch.setattr(_gens, "delete_for_user", lambda rid, uid: True)
        deleted = []
        monkeypatch.setattr(_genlogs, "delete_for_user",
                            lambda lid, uid: deleted.append(lid) or True)
        r = client.delete("/api/account/generations/rec-1", headers=_auth(user_token))
        assert r.status_code == 200
        assert deleted == []

    def test_delete_without_paired_log_leaves_logs_untouched(self, client, user_token, monkeypatch):
        """No paired log (e.g. the run's log aged out) → the log table is never
        touched, and the spec delete still succeeds."""
        from backend.service import spec_generation_logs as _genlogs
        monkeypatch.setattr(_gens, "list_for_user", lambda uid: [
            {"id": "rec-1", "model_id": "m", "created_at": "2026-01-01T00:00:00+00:00"},
        ])
        monkeypatch.setattr(_genlogs, "list_for_user", lambda uid, limit=1000: [])
        monkeypatch.setattr(_gens, "delete_for_user", lambda rid, uid: True)
        calls = {"n": 0}
        def fake_log_delete(lid, uid):
            calls["n"] += 1
            return True
        monkeypatch.setattr(_genlogs, "delete_for_user", fake_log_delete)
        r = client.delete("/api/account/generations/rec-1", headers=_auth(user_token))
        assert r.status_code == 200
        assert calls["n"] == 0


class TestGenLogDelete:
    """DELETE /api/account/gen-logs/{id} — orphan run logs are managed inline
    on the generations page, so they get the same owner-scoped delete."""

    def test_anonymous_401(self, client):
        assert client.delete("/api/account/gen-logs/log-1").status_code == 401

    def test_owner_delete_ok(self, client, user_token, monkeypatch):
        from backend.service import spec_generation_logs as _genlogs
        seen = {}
        def fake_delete(lid, uid):
            seen["args"] = (lid, uid)
            return True
        monkeypatch.setattr(_genlogs, "delete_for_user", fake_delete)
        r = client.delete("/api/account/gen-logs/log-1", headers=_auth(user_token))
        assert r.status_code == 200 and r.json()["deleted"] is True
        assert seen["args"] == ("log-1", "user-uuid-1")

    def test_not_found_404(self, client, user_token, monkeypatch):
        from backend.service import spec_generation_logs as _genlogs
        monkeypatch.setattr(_genlogs, "delete_for_user", lambda lid, uid: False)
        r = client.delete("/api/account/gen-logs/log-1", headers=_auth(user_token))
        assert r.status_code == 404

    def test_service_scopes_by_owner(self, httpx_mock):
        from backend.service import spec_generation_logs as _genlogs
        url = re.compile(
            re.escape(f"{TEST_SUPABASE_URL}/rest/v1/spec_generation_logs") + r"(\?.*)?")
        httpx_mock.add_response(method="DELETE", url=url, json=[{"id": "log-1"}])
        assert _genlogs.delete_for_user("log-1", "user-uuid-1") is True
        req_url = str(httpx_mock.get_requests(method="DELETE")[0].url)
        assert "id=eq.log-1" in req_url and "user_id=eq.user-uuid-1" in req_url
