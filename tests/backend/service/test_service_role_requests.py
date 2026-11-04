"""Every request made as the Supabase service role carries the same identity,
and each caller's request is exactly what it sent before the headers were
spelled once in `backend.supabase_rest`.

Each case drives one caller against a mocked transport and pins the method,
URL, the four identity/intent headers (present with a value, or absent) and
the JSON body. The expected values are literal, not derived from
`service_headers`, so a change to that function shows up here as a changed
request.
"""
from __future__ import annotations

import json

import pytest

from .conftest import TEST_SERVICE_KEY, TEST_SUPABASE_URL

REST = f"{TEST_SUPABASE_URL}/rest/v1"
AUTH = f"Bearer {TEST_SERVICE_KEY}"
JSON = "application/json"


def _identity(req) -> dict:
    """The headers this test pins; an absent header is None."""
    return {h: req.headers.get(h)
            for h in ("apikey", "authorization", "content-type", "prefer")}


def _only_request(httpx_mock):
    reqs = httpx_mock.get_requests()
    assert len(reqs) == 1, [str(r.url) for r in reqs]
    return reqs[0]


def _body(req):
    return json.loads(req.content) if req.content else None


def _headers(*, content_type=JSON, prefer=None) -> dict:
    return {"apikey": TEST_SERVICE_KEY, "authorization": AUTH,
            "content-type": content_type, "prefer": prefer}


class TestRpc:
    @pytest.mark.parametrize("module, fn", [
        ("backend.service.sweep_cache", "sweep_cache_get"),
        ("backend.service.shared_spec_cache", "get_cached_spec"),
    ])
    def test_cache_rpc(self, httpx_mock, module, fn):
        import importlib
        mod = importlib.import_module(module)
        httpx_mock.add_response(method="POST", url=f"{REST}/rpc/{fn}", json=[{"a": 1}])
        assert mod._rpc(fn, {"p_key": "k"}) == [{"a": 1}]
        req = _only_request(httpx_mock)
        assert req.method == "POST"
        assert _identity(req) == _headers()
        assert _body(req) == {"p_key": "k"}

    @pytest.mark.parametrize("module", ["backend.service.sweep_cache",
                                        "backend.service.shared_spec_cache"])
    def test_cache_rpc_void_reply_is_empty_list(self, httpx_mock, module):
        import importlib
        mod = importlib.import_module(module)
        httpx_mock.add_response(method="POST", url=f"{REST}/rpc/f", status_code=204)
        assert mod._rpc("f", {}) == []

    @pytest.mark.parametrize("module", ["backend.service.sweep_cache",
                                        "backend.service.shared_spec_cache"])
    def test_cache_rpc_http_error_is_none(self, httpx_mock, module):
        import importlib
        mod = importlib.import_module(module)
        httpx_mock.add_response(method="POST", url=f"{REST}/rpc/f", status_code=500)
        assert mod._rpc("f", {}) is None

    def test_spec_lock_rpc(self, httpx_mock):
        from backend.spec_worker import runtime
        httpx_mock.add_response(method="POST", url=f"{REST}/rpc/try_acquire_spec_lock",
                                json=[{"acquired": True}])
        assert runtime._supabase_rpc("try_acquire_spec_lock", {"p_model_id": "m"}) \
            == [{"acquired": True}]
        req = _only_request(httpx_mock)
        assert _identity(req) == _headers()
        assert _body(req) == {"p_model_id": "m"}

    def test_spec_lock_rpc_void_reply_raises(self, httpx_mock):
        """The lock RPC always decodes the body: an empty one raises, as it
        did before, where the caches read it as []."""
        from backend.spec_worker import runtime
        httpx_mock.add_response(method="POST", url=f"{REST}/rpc/f", status_code=204)
        with pytest.raises(ValueError):
            runtime._supabase_rpc("f", {})

    def test_spec_lock_rpc_http_error_is_none(self, httpx_mock):
        from backend.spec_worker import runtime
        httpx_mock.add_response(method="POST", url=f"{REST}/rpc/f", status_code=503)
        assert runtime._supabase_rpc("f", {}) is None


class TestTableCallers:
    def test_spec_generations_record(self, httpx_mock):
        from backend.service import spec_generations
        httpx_mock.add_response(method="POST", url=f"{REST}/rpc/record_generation",
                                json={"outcome": "recorded", "id": "r1"})
        assert spec_generations.record(
            model_id="o/m", user_id="u", spec_yaml="y",
            run_started_at="2026-10-01T00:00:00+00:00",
        ) == {"outcome": "recorded", "id": "r1"}
        req = _only_request(httpx_mock)
        assert _identity(req) == _headers()
        assert _body(req) == {"p_user_id": "u", "p_model_id": "o/m", "p_spec_yaml": "y",
                              "p_snapshot_json": {}, "p_new_component_yamls": [],
                              "p_agent_cli": None, "p_agent_model": None,
                              "p_run_started_at": "2026-10-01T00:00:00+00:00"}

    def test_spec_generations_list(self, httpx_mock):
        from backend.service import spec_generations
        httpx_mock.add_response(method="GET", json=[])
        assert spec_generations.list_for_user("u") == []
        req = _only_request(httpx_mock)
        assert req.url.path == "/rest/v1/spec_generations"
        assert _identity(req) == _headers()

    def test_run_log_record(self, httpx_mock):
        from backend.service import spec_generation_logs as logs
        httpx_mock.add_response(method="POST", url=f"{REST}/spec_generation_logs",
                                status_code=201)
        assert logs.record(user_id="u", model_id="o/m", log="L", run_key="k",
                           success=True) is True
        reqs = httpx_mock.get_requests()
        assert [r.method for r in reqs] == ["POST"]
        req = reqs[0]
        assert _identity(req) == _headers(prefer="return=minimal")
        assert _body(req) == {"user_id": "u", "model_id": "o/m", "run_key": "k",
                              "success": True, "log": "L", "state": "done"}

    def test_run_log_delete(self, httpx_mock):
        from backend.service import spec_generation_logs as logs
        httpx_mock.add_response(method="DELETE", json=[{"id": "1"}])
        assert logs.delete_for_user("1", "u") is True
        req = _only_request(httpx_mock)
        assert req.url.path == "/rest/v1/spec_generation_logs"
        assert _identity(req) == _headers(content_type=None, prefer="return=representation")

    @pytest.mark.parametrize("call", ["get", "list_for_user"])
    def test_run_log_reads(self, httpx_mock, call):
        from backend.service import spec_generation_logs as logs
        httpx_mock.add_response(method="GET", json=[])
        if call == "get":
            logs.get("1", "u")
        else:
            logs.list_for_user("u")
        req = _only_request(httpx_mock)
        assert req.url.path == "/rest/v1/spec_generation_logs"
        assert _identity(req) == _headers(content_type=None)

    def test_module_variants_list(self, httpx_mock):
        from backend.service import module_variants
        httpx_mock.add_response(method="GET", json=[])
        assert module_variants.list_for_user("u") == []
        req = _only_request(httpx_mock)
        assert req.url.path == "/rest/v1/user_module_variants"
        assert _identity(req) == _headers()

    def test_agent_credentials_list_and_delete(self, httpx_mock):
        from backend.service import agent_credentials as creds
        httpx_mock.add_response(method="GET", json=[])
        httpx_mock.add_response(method="DELETE", status_code=204)
        creds.list_for_user("u")
        creds.delete("u", "claude")
        get, delete = httpx_mock.get_requests()
        assert get.url.path == delete.url.path == "/rest/v1/user_agent_credentials"
        assert _identity(get) == _headers()
        assert _identity(delete) == _headers(prefer="return=minimal")

    def test_agent_credentials_set_default(self, httpx_mock):
        from backend.service import agent_credentials as creds
        httpx_mock.add_response(method="PATCH", status_code=204)
        httpx_mock.add_response(method="PATCH", json=[{"agent_cli": "claude"}])
        assert creds.set_default(user_id="u", agent_cli="claude") is True
        clear, target = httpx_mock.get_requests()
        assert _identity(clear) == _headers(prefer="return=minimal")
        assert _identity(target) == _headers(prefer="return=representation")
        assert _body(target)["is_default"] is True

    def test_personal_tokens(self, httpx_mock):
        from backend.service import personal_tokens as pats
        httpx_mock.add_response(method="GET", json=[])
        httpx_mock.add_response(method="PATCH", json=[{"id": "t"}])
        pats.list_for_user("u")
        pats.revoke(token_id="t", user_id="u")
        get, patch = httpx_mock.get_requests()
        assert get.url.path == patch.url.path == "/rest/v1/personal_access_tokens"
        assert _identity(get) == _headers()
        assert _identity(patch) == _headers(prefer="return=representation")

    def test_profiles(self, httpx_mock):
        from backend.service import profiles
        httpx_mock.add_response(method="GET", json=[{"github_handle": "octo"}])
        assert profiles.get_github_handle("u") == "octo"
        req = _only_request(httpx_mock)
        assert req.url.path == "/rest/v1/profiles"
        assert _identity(req) == _headers()

    def test_auth_service_role_client(self):
        from backend.service import auth
        with auth._service_role_client() as client:
            assert str(client.base_url) == f"{REST}/"
            assert {h: client.headers.get(h) for h in ("apikey", "authorization",
                                                        "content-type", "prefer")} \
                == _headers()
            assert client.timeout.read == 5.0


class TestAdminAuthCalls:
    def test_invite(self, client, admin_token, httpx_mock):
        from .conftest import mock_admins_yes
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        httpx_mock.add_response(method="POST", url=f"{TEST_SUPABASE_URL}/auth/v1/admin/invite",
                                json={"id": "n"})
        r = client.post("/api/admin/users/invite", json={"email": "a@b.co"},
                        headers={"Authorization": f"Bearer {admin_token}"})
        assert r.status_code == 200
        req = next(q for q in httpx_mock.get_requests() if q.url.path.endswith("/invite"))
        assert _identity(req) == _headers()
        assert _body(req) == {"email": "a@b.co"}

    def test_delete(self, client, admin_token, httpx_mock):
        from .conftest import mock_admins_yes
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        httpx_mock.add_response(method="POST", url=f"{REST}/rpc/delete_user_data", json=True)
        httpx_mock.add_response(method="DELETE",
                                url=f"{TEST_SUPABASE_URL}/auth/v1/admin/users/t", json={})
        r = client.post("/api/admin/users/t/delete",
                        headers={"Authorization": f"Bearer {admin_token}"})
        assert r.status_code == 200
        req = next(q for q in httpx_mock.get_requests() if q.method == "DELETE")
        assert _identity(req) == _headers(content_type=None)
