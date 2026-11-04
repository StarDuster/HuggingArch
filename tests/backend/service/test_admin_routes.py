"""/api/admin/* — invite, summary, events, delete.

Every endpoint is admin-gated by `require_admin`. Anonymous → 401,
signed-in non-admin → 403, admin → 200 (with mocked supabase).
"""
from __future__ import annotations

from .conftest import (
    TEST_SUPABASE_URL,
    mock_admins_no,
    mock_admins_yes,
)


class TestAdminGating:
    """Confirm every admin endpoint enforces auth + role consistently."""

    ENDPOINTS = (
        ("POST",   "/api/admin/users/invite",                {"email": "x@y.com"}),
        ("GET",    "/api/admin/users/abc/summary",           None),
        ("GET",    "/api/admin/users/abc/events",            None),
        ("POST",   "/api/admin/users/abc/delete",            None),
        # Cached-specs (Phase B)
        ("GET",    "/api/admin/cached-specs",                None),
        ("GET",    "/api/admin/cached-specs/foo/bar",        None),
        ("DELETE", "/api/admin/cached-specs/foo/bar",        None),
        # Model-stack refresh dispatch
        ("POST",   "/api/admin/update-model-stack",          None),
    )

    def test_anonymous_gets_401(self, client, subtests):
        for method, path, body in self.ENDPOINTS:
            with subtests.test(path=path):
                r = client.request(method, path, json=body)
                assert r.status_code == 401, f"{path} returned {r.status_code}"

    def test_non_admin_gets_403(self, client, user_token, httpx_mock, subtests):
        mock_admins_no(httpx_mock, "user-uuid-1")
        for method, path, body in self.ENDPOINTS:
            with subtests.test(path=path):
                r = client.request(
                    method, path, json=body,
                    headers={"Authorization": f"Bearer {user_token}"},
                )
                assert r.status_code == 403, f"{path} returned {r.status_code}"


# ---------------------------------------------------------------------------
# Invite
# ---------------------------------------------------------------------------

class TestInviteUser:
    def test_admin_can_invite(self, client, admin_token, httpx_mock):
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        captured = {}
        def _capture(req):
            import json
            captured["body"] = json.loads(req.content)
            from httpx import Response
            return Response(200, json={"id": "new-uuid"})
        httpx_mock.add_callback(
            _capture, method="POST",
            url=f"{TEST_SUPABASE_URL}/auth/v1/admin/invite",
        )

        r = client.post(
            "/api/admin/users/invite",
            json={"email": "new@example.com", "username": "newbie"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["invited"] is True
        assert body["email"] == "new@example.com"
        assert body["user_id"] == "new-uuid"
        # Verify the data field made it through (drives profile pre-fill)
        assert captured["body"]["email"] == "new@example.com"
        assert captured["body"]["data"] == {"username": "newbie"}

    def test_invalid_email_returns_400(self, client, admin_token, httpx_mock):
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        r = client.post(
            "/api/admin/users/invite",
            json={"email": "not-an-email"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 400
        assert "invalid email" in r.json()["detail"].lower()

    def test_supabase_error_propagates(self, client, admin_token, httpx_mock):
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        httpx_mock.add_response(
            method="POST",
            url=f"{TEST_SUPABASE_URL}/auth/v1/admin/invite",
            status_code=422,
            text='{"msg": "User already registered"}',
        )
        r = client.post(
            "/api/admin/users/invite",
            json={"email": "exists@example.com"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 422
        assert "already registered" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Per-user drill-down
# ---------------------------------------------------------------------------

class TestUserSummaryAndEvents:
    def test_summary_passes_through(self, client, admin_token, httpx_mock):
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        rpc_body = [{"total_calls": 42, "analyze_calls": 30, "inference_calls": 12,
                     "unique_models": 5, "first_event_at": None, "last_event_at": None,
                     "login_count": 3, "last_login_at": None}]
        httpx_mock.add_response(
            method="POST",
            url=f"{TEST_SUPABASE_URL}/rest/v1/rpc/get_user_summary",
            json=rpc_body,
        )
        r = client.get(
            "/api/admin/users/target-uuid/summary",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 200
        assert r.json() == rpc_body

    def test_events_passes_pagination(self, client, admin_token, httpx_mock):
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        captured = {}
        def _capture(req):
            import json
            captured["body"] = json.loads(req.content)
            from httpx import Response
            return Response(200, json=[])
        httpx_mock.add_callback(
            _capture, method="POST",
            url=f"{TEST_SUPABASE_URL}/rest/v1/rpc/get_user_analyze_logs",
        )
        client.get(
            "/api/admin/users/target-uuid/events?limit=25&offset=50",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert captured["body"] == {
            "p_user_id": "target-uuid", "p_limit": 25, "p_offset": 50,
        }


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

class TestDeleteUser:
    def test_admin_can_delete_other_user(self, client, admin_token, httpx_mock):
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        # Step 1: Postgres-side cascade RPC
        httpx_mock.add_response(
            method="POST",
            url=f"{TEST_SUPABASE_URL}/rest/v1/rpc/delete_user_data",
            json=True,
        )
        # Step 2: Auth Admin DELETE
        httpx_mock.add_response(
            method="DELETE",
            url=f"{TEST_SUPABASE_URL}/auth/v1/admin/users/target-uuid",
            status_code=200,
            json={},
        )

        r = client.post(
            "/api/admin/users/target-uuid/delete",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 200
        assert r.json() == {"deleted": True, "user_id": "target-uuid"}

    def test_admin_cannot_delete_self(self, client, admin_token, httpx_mock):
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        r = client.post(
            "/api/admin/users/admin-uuid-1/delete",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 400
        assert "yourself" in r.json()["detail"].lower()

    def test_partial_failure_surfaces_502(self, client, admin_token, httpx_mock):
        """Step 1 succeeds, step 2 fails → 502 so admin can retry."""
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        httpx_mock.add_response(
            method="POST",
            url=f"{TEST_SUPABASE_URL}/rest/v1/rpc/delete_user_data",
            json=True,
        )
        httpx_mock.add_response(
            method="DELETE",
            url=f"{TEST_SUPABASE_URL}/auth/v1/admin/users/target-uuid",
            status_code=500, text="auth admin down",
        )
        r = client.post(
            "/api/admin/users/target-uuid/delete",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 502


# ---------------------------------------------------------------------------
# Cached specs (Phase B)
# ---------------------------------------------------------------------------

class TestCachedSpecsList:
    def test_admin_lists_cached_specs(self, client, admin_token, httpx_mock):
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        rows = [{
            "model_id": "meta-llama/Llama-4",
            "agent_cli": "claude",
            "hit_count": 17,
            "spec_yaml_bytes": 2048,
            "new_component_count": 1,
            "generated_by_email": "alice@example.com",
            "created_at": "2026-05-05T00:00:00Z",
            "last_used_at": "2026-05-05T01:00:00Z",
        }]
        httpx_mock.add_response(
            method="POST",
            url=f"{TEST_SUPABASE_URL}/rest/v1/rpc/get_cached_specs",
            json=rows,
        )
        httpx_mock.add_response(
            method="POST",
            url=f"{TEST_SUPABASE_URL}/rest/v1/rpc/get_cached_specs_count",
            json=42,
        )
        r = client.get(
            "/api/admin/cached-specs?limit=50&offset=0&sort=hits",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["items"] == rows
        assert body["total"] == 42

    def test_invalid_sort_400(self, client, admin_token, httpx_mock):
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        r = client.get(
            "/api/admin/cached-specs?sort=bogus",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 400


class TestCachedSpecDetail:
    def test_returns_full_payload(self, client, admin_token, httpx_mock):
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        row = {
            "model_id": "meta-llama/Llama-4",
            "agent_cli": "claude",
            "hit_count": 17,
            "spec_yaml": "match:\n  model_type: [llama4]\n",
            "snapshot_json": {"total_params": 70_000_000_000},
            "new_component_yamls": [{"path": "x.yaml", "body": "name: x"}],
            "generated_by_email": "alice@example.com",
            "created_at": "2026-05-05T00:00:00Z",
            "last_used_at": "2026-05-05T01:00:00Z",
        }
        httpx_mock.add_response(
            method="POST",
            url=f"{TEST_SUPABASE_URL}/rest/v1/rpc/get_cached_spec_detail",
            json=[row],
        )
        r = client.get(
            "/api/admin/cached-specs/meta-llama/Llama-4",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 200
        assert r.json() == row

    def test_404_when_missing(self, client, admin_token, httpx_mock):
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        httpx_mock.add_response(
            method="POST",
            url=f"{TEST_SUPABASE_URL}/rest/v1/rpc/get_cached_spec_detail",
            json=[],
        )
        r = client.get(
            "/api/admin/cached-specs/never/cached",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 404


class TestCachedSpecDelete:
    def test_admin_can_delete(self, client, admin_token, httpx_mock):
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        httpx_mock.add_response(
            method="POST",
            url=f"{TEST_SUPABASE_URL}/rest/v1/rpc/delete_cached_spec",
            json=True,
        )
        r = client.delete(
            "/api/admin/cached-specs/foo/bar",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 200
        assert r.json() == {"deleted": True, "model_id": "foo/bar"}

    def test_no_row_returns_deleted_false(self, client, admin_token, httpx_mock):
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        httpx_mock.add_response(
            method="POST",
            url=f"{TEST_SUPABASE_URL}/rest/v1/rpc/delete_cached_spec",
            json=False,
        )
        r = client.delete(
            "/api/admin/cached-specs/never/existed",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 200
        assert r.json() == {"deleted": False, "model_id": "never/existed"}


# ---------------------------------------------------------------------------
# Model-stack refresh dispatch (POST /api/admin/update-model-stack)
# ---------------------------------------------------------------------------

_GH_REPO = "https://api.github.com/repos/shenh10/HuggingArch"
_GH_DISPATCH = f"{_GH_REPO}/actions/workflows/update-transformers.yml/dispatches"


def _clear_github_env(monkeypatch):
    """Isolate from any HA_GITHUB_* the host env may provide (App creds win
    over the PAT in _token(), so clear both modes)."""
    for var in ("HA_GITHUB_TOKEN", "HA_GITHUB_APP_ID",
                "HA_GITHUB_APP_PRIVATE_KEY", "HA_GITHUB_APP_INSTALLATION_ID",
                "HA_GITHUB_REPO"):
        monkeypatch.delenv(var, raising=False)


class TestUpdateModelStack:
    def test_unconfigured_returns_503(self, client, admin_token, httpx_mock, monkeypatch):
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        _clear_github_env(monkeypatch)
        r = client.post(
            "/api/admin/update-model-stack",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 503

    def test_dispatches_workflow_on_default_branch(
            self, client, admin_token, httpx_mock, monkeypatch):
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        _clear_github_env(monkeypatch)
        monkeypatch.setenv("HA_GITHUB_TOKEN", "tok")
        httpx_mock.add_response(
            method="GET", url=_GH_REPO, json={"default_branch": "main"})
        captured = {}

        def _capture(req):
            import json
            captured["body"] = json.loads(req.content)
            from httpx import Response
            return Response(204)

        httpx_mock.add_callback(_capture, method="POST", url=_GH_DISPATCH)
        r = client.post(
            "/api/admin/update-model-stack",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["triggered"] is True
        assert "update-transformers.yml" in body["workflow_url"]
        assert captured["body"] == {"ref": "main"}

    def test_github_403_surfaces_502_with_detail(
            self, client, admin_token, httpx_mock, monkeypatch):
        """The expected misconfiguration: the App/PAT lacks Actions: write.
        The dashboard must see GitHub's message, not a bare 500."""
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        _clear_github_env(monkeypatch)
        monkeypatch.setenv("HA_GITHUB_TOKEN", "tok")
        httpx_mock.add_response(
            method="GET", url=_GH_REPO, json={"default_branch": "main"})
        httpx_mock.add_response(
            method="POST", url=_GH_DISPATCH, status_code=403,
            text='{"message":"Resource not accessible by integration"}')
        r = client.post(
            "/api/admin/update-model-stack",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 502
        assert "403" in r.json()["detail"]
        assert "Resource not accessible" in r.json()["detail"]
