"""/api/auth/* — cookie-based browser session endpoints.

Covers: login sets the two httpOnly cookies (and never leaks a token in
the body), refresh rotates them, logout clears them, the implicit-flow
exchange verifies before trusting, /api/auth/me doubles as the session
probe, cookie auth satisfies require_user and require_user_or_pat, the
bearer path (PATs, scripts) is untouched, and the CSRF guard rejects
cross-origin cookie-authed mutations plus any session-endpoint call
missing the SPA's custom header.
"""
from __future__ import annotations

import re

from .conftest import (
    TEST_SUPABASE_URL,
    make_expired_jwt,
    make_user_jwt,
    mock_admins_no,
    mock_admins_yes,
)


_GOTRUE_USER = {"id": "user-uuid-1", "email": "alice@example.com", "user_metadata": {}}

# The custom header the SPA sends on /api/auth/* mutations; the CSRF
# middleware rejects any session-endpoint mutation without it.
SESSION_HEADERS = {"X-HA-Auth": "1"}


def _set_cookies(r):
    return r.headers.get_list("set-cookie")


def _cookie_named(r, name):
    return [c for c in _set_cookies(r) if c.startswith(f"{name}=")]


def _mock_gotrue_token(httpx_mock, grant_type, *, status=200, body=None, reusable=False):
    httpx_mock.add_response(
        method="POST",
        url=f"{TEST_SUPABASE_URL}/auth/v1/token?grant_type={grant_type}",
        status_code=status,
        json=body if body is not None else {
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
            "expires_in": 3600,
            "user": _GOTRUE_USER,
        },
        is_reusable=reusable,
    )


def _mock_me_upstream(httpx_mock, user_id="user-uuid-1"):
    """The three calls /api/auth/me makes upstream for a signed-in user."""
    httpx_mock.add_response(
        method="GET", url=f"{TEST_SUPABASE_URL}/auth/v1/user",
        json={**_GOTRUE_USER, "id": user_id},
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{TEST_SUPABASE_URL}/rest/v1/rpc/create_user_profile_if_not_exists",
        json=None,
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{TEST_SUPABASE_URL}/rest/v1/profiles?id=eq.{user_id}&select=*",
        json=[{"username": "alice"}],
    )


class TestLogin:
    def test_sets_both_cookies_without_leaking_tokens(self, client, httpx_mock):
        _mock_gotrue_token(httpx_mock, "password")
        r = client.post(
            "/api/auth/login",
            json={"email": "a@b.c", "password": "pw"},
            headers=SESSION_HEADERS,
        )
        assert r.status_code == 200
        assert r.json() == {"user": _GOTRUE_USER}
        assert "access_token" not in r.text and "fresh-refresh" not in r.text

        (at,) = _cookie_named(r, "ha_at")
        assert "fresh-access" in at
        assert "HttpOnly" in at and "SameSite=lax" in at and "Path=/" in at
        (rt,) = _cookie_named(r, "ha_rt")
        assert "fresh-refresh" in rt
        assert "HttpOnly" in rt and "SameSite=strict" in rt and "Path=/api/auth" in rt

    def test_bad_password_propagates_gotrue_message(self, client, httpx_mock):
        _mock_gotrue_token(
            httpx_mock, "password",
            status=400, body={"error_description": "Invalid login credentials"},
        )
        r = client.post(
            "/api/auth/login",
            json={"email": "a@b.c", "password": "no"},
            headers=SESSION_HEADERS,
        )
        assert r.status_code == 401
        assert r.json()["detail"] == "Invalid login credentials"
        assert _set_cookies(r) == []


class TestRefresh:
    def test_rotates_cookies(self, client, httpx_mock):
        _mock_gotrue_token(httpx_mock, "refresh_token", body={
            "access_token": "rotated-access",
            "refresh_token": "rotated-refresh",
            "expires_in": 3600,
            "user": _GOTRUE_USER,
        })
        client.cookies.set("ha_rt", "old-refresh")
        r = client.post("/api/auth/refresh", headers=SESSION_HEADERS)
        assert r.status_code == 200
        assert "rotated-access" in _cookie_named(r, "ha_at")[0]
        assert "rotated-refresh" in _cookie_named(r, "ha_rt")[0]

    def test_no_refresh_cookie_is_401(self, client):
        r = client.post("/api/auth/refresh", headers=SESSION_HEADERS)
        assert r.status_code == 401

    def test_failed_refresh_clears_cookies(self, client, httpx_mock):
        _mock_gotrue_token(
            httpx_mock, "refresh_token",
            status=400, body={"msg": "Invalid Refresh Token"},
        )
        client.cookies.set("ha_rt", "stale")
        r = client.post("/api/auth/refresh", headers=SESSION_HEADERS)
        assert r.status_code == 401
        # delete_cookie emits an expired empty value for both names.
        assert any(c.startswith("ha_at=") and "expires=" in c.lower() for c in _set_cookies(r))
        assert any(c.startswith("ha_rt=") and "expires=" in c.lower() for c in _set_cookies(r))

    def test_transient_gotrue_failure_keeps_session(self, client, httpx_mock):
        """A provider 5xx must NOT destroy a locally-valid session: the
        cookies stay and the next page load retries."""
        _mock_gotrue_token(httpx_mock, "refresh_token", status=500, body={})
        client.cookies.set("ha_rt", "still-valid")
        r = client.post("/api/auth/refresh", headers=SESSION_HEADERS)
        assert r.status_code == 502
        assert _set_cookies(r) == []

    def test_gotrue_rate_limit_keeps_session(self, client, httpx_mock):
        """GoTrue's own 429 is 'retry later', not 'credential invalid' —
        same keep-the-session treatment as a 5xx."""
        _mock_gotrue_token(httpx_mock, "refresh_token", status=429, body={})
        client.cookies.set("ha_rt", "still-valid")
        r = client.post("/api/auth/refresh", headers=SESSION_HEADERS)
        assert r.status_code == 502
        assert _set_cookies(r) == []


class TestLogout:
    def test_clears_cookies_and_calls_gotrue(self, client, httpx_mock):
        httpx_mock.add_response(
            method="POST", url=f"{TEST_SUPABASE_URL}/auth/v1/logout", status_code=204,
        )
        client.cookies.set("ha_at", make_user_jwt())
        r = client.post("/api/auth/logout", headers=SESSION_HEADERS)
        assert r.status_code == 200
        assert any(c.startswith("ha_at=") for c in _set_cookies(r))
        assert any(c.startswith("ha_rt=") for c in _set_cookies(r))


class TestExchange:
    def test_valid_token_sets_cookies(self, client):
        r = client.post("/api/auth/exchange", json={
            "access_token": make_user_jwt(),
            "refresh_token": "rt-1",
            "expires_in": 3600,
        }, headers=SESSION_HEADERS)
        assert r.status_code == 200
        assert r.json()["user"]["id"] == "user-uuid-1"
        assert _cookie_named(r, "ha_at") and _cookie_named(r, "ha_rt")

    def test_garbage_token_rejected(self, client):
        r = client.post(
            "/api/auth/exchange", json={"access_token": "not-a-jwt"},
            headers=SESSION_HEADERS,
        )
        assert r.status_code == 401
        assert _set_cookies(r) == []

    def test_expired_token_rejected(self, client):
        r = client.post("/api/auth/exchange", json={
            "access_token": make_expired_jwt(),
        }, headers=SESSION_HEADERS)
        assert r.status_code == 401


class TestMe:
    def test_anonymous_gets_null_user(self, client):
        r = client.get("/api/auth/me")
        assert r.status_code == 200
        body = r.json()
        assert body["authEnabled"] is True
        assert body["user"] is None and body["isAdmin"] is False

    def test_expired_cookie_is_401_so_client_refreshes(self, client):
        client.cookies.set("ha_at", make_expired_jwt())
        r = client.get("/api/auth/me")
        assert r.status_code == 401

    def test_refresh_cookie_alone_is_401_so_client_refreshes(self, client):
        # The access cookie's Max-Age tracks the JWT, so an hour later the
        # browser sends ha_rt alone — that is "refresh me", not "anonymous".
        client.cookies.set("ha_rt", "rt-1")
        r = client.get("/api/auth/me")
        assert r.status_code == 401

    def test_signed_in_cookie_returns_user_profile_admin(self, client, httpx_mock):
        _mock_me_upstream(httpx_mock)
        mock_admins_yes(httpx_mock, "user-uuid-1")
        client.cookies.set("ha_at", make_user_jwt())
        r = client.get("/api/auth/me")
        assert r.status_code == 200
        body = r.json()
        assert body["user"]["id"] == "user-uuid-1"
        assert body["profile"] == {"username": "alice"}
        assert body["isAdmin"] is True

    def test_bearer_header_still_works(self, client, httpx_mock):
        _mock_me_upstream(httpx_mock)
        mock_admins_no(httpx_mock, "user-uuid-1")
        r = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {make_user_jwt()}"},
        )
        assert r.status_code == 200
        assert r.json()["user"]["id"] == "user-uuid-1"
        assert r.json()["isAdmin"] is False

    def test_no_service_role_key_degrades_to_non_admin(
        self, client, httpx_mock, monkeypatch,
    ):
        """Deployments without SUPABASE_SERVICE_ROLE_KEY can't check admin
        status — the probe must still succeed, with isAdmin False."""
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY")
        _mock_me_upstream(httpx_mock)
        client.cookies.set("ha_at", make_user_jwt())
        r = client.get("/api/auth/me")
        assert r.status_code == 200
        assert r.json()["user"]["id"] == "user-uuid-1"
        assert r.json()["isAdmin"] is False

    def test_supabase_outage_degrades_to_non_admin(self, client, httpx_mock):
        """The admins lookup hitting a network error must not 500 the whole
        session probe."""
        import httpx as _httpx
        _mock_me_upstream(httpx_mock)
        httpx_mock.add_exception(
            _httpx.ConnectError("supabase down"),
            method="GET",
            url=f"{TEST_SUPABASE_URL}/rest/v1/admins?user_id=eq.user-uuid-1&select=user_id",
        )
        client.cookies.set("ha_at", make_user_jwt())
        r = client.get("/api/auth/me")
        assert r.status_code == 200
        assert r.json()["user"]["id"] == "user-uuid-1"
        assert r.json()["isAdmin"] is False


class TestCookieSatisfiesRequireUser:
    """A require_user route must accept the session cookie with no
    Authorization header — this is what lets the SPA drop bearer plumbing."""

    def test_cookie_auth_on_account_route(self, client, httpx_mock):
        httpx_mock.add_response(
            method="GET",
            url=re.compile(re.escape(f"{TEST_SUPABASE_URL}/rest/v1/user_agent_credentials") + r".*"),
            json=[],
        )
        httpx_mock.add_response(
            method="GET",
            url=re.compile(re.escape(f"{TEST_SUPABASE_URL}/rest/v1/agent_task_defaults") + r".*"),
            json=[],
        )
        httpx_mock.add_response(
            method="GET",
            url=re.compile(re.escape(f"{TEST_SUPABASE_URL}/rest/v1/profiles") + r".*"),
            json=[],
        )
        client.cookies.set("ha_at", make_user_jwt())
        r = client.get("/api/account/agent-credentials")
        assert r.status_code == 200

    def test_no_credentials_is_401(self, client):
        r = client.get("/api/account/agent-credentials")
        assert r.status_code == 401


class TestBenchUploadWithCookie:
    """The HA API's proxy accepts a PAT or a session on /api/bench/upload:
    PATs arrive as bearer, but the browser authenticates with the session
    cookie; the Kernel Bench service is the in-process one."""

    def test_cookie_auth_upload_accepted(self, client, bench_lake, bench_proxied, monkeypatch):
        monkeypatch.setattr("backend.service.auth.is_admin", lambda uid: False)
        import tests.corpus_row as _corpus
        client.cookies.set("ha_at", make_user_jwt(sub="11111111-1111-4111-8111-111111111111"))
        r = client.post(
            "/api/bench/upload",
            json={"csv_text": _corpus.to_csv([_corpus.row(
                contributor="@me", date="2026-07-26", node_spec_id="self_attn",
            )])},
        )
        assert r.status_code == 200, r.text

    def test_anonymous_upload_says_sign_in(self, client, bench_proxied):
        import tests.corpus_row as _corpus
        r = client.post(
            "/api/bench/upload",
            json={"csv_text": _corpus.to_csv([_corpus.row(
                contributor="@me", date="2026-07-26", node_spec_id="self_attn",
            )])},
        )
        assert r.status_code == 401
        assert r.json()["detail"] == "Sign-in required"


class TestUpdatePassword:
    def test_cookie_auth_proxies_to_gotrue(self, client, httpx_mock):
        httpx_mock.add_response(
            method="PUT", url=f"{TEST_SUPABASE_URL}/auth/v1/user", json=_GOTRUE_USER,
        )
        client.cookies.set("ha_at", make_user_jwt())
        r = client.put(
            "/api/auth/password", json={"password": "new-pw"},
            headers=SESSION_HEADERS,
        )
        assert r.status_code == 200

    def test_anonymous_is_401(self, client):
        r = client.put(
            "/api/auth/password", json={"password": "new-pw"},
            headers=SESSION_HEADERS,
        )
        assert r.status_code == 401


class TestAdminRpc:
    def test_allowlisted_rpc_forwards_jwt(self, client, httpx_mock):
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        captured = {}

        def _capture(request):
            import json as _json
            captured["authorization"] = request.headers.get("authorization")
            captured["body"] = _json.loads(request.content)
            from httpx import Response
            return Response(200, json={"ok": True})

        httpx_mock.add_callback(
            _capture, method="POST",
            url=f"{TEST_SUPABASE_URL}/rest/v1/rpc/add_admin",
        )
        client.cookies.set("ha_at", make_user_jwt(sub="admin-uuid-1"))
        r = client.post("/api/auth/admin-rpc", json={
            "name": "add_admin", "args": {"p_user_id": "user-uuid-1"},
        }, headers=SESSION_HEADERS)
        assert r.status_code == 200
        assert captured["authorization"].startswith("Bearer ")
        assert captured["body"] == {"p_user_id": "user-uuid-1"}

    def test_non_allowlisted_name_rejected(self, client, httpx_mock):
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        client.cookies.set("ha_at", make_user_jwt(sub="admin-uuid-1"))
        r = client.post("/api/auth/admin-rpc", json={
            "name": "drop_all_tables", "args": {},
        }, headers=SESSION_HEADERS)
        assert r.status_code == 403

    def test_non_admin_rejected(self, client, httpx_mock):
        mock_admins_no(httpx_mock, "user-uuid-1")
        client.cookies.set("ha_at", make_user_jwt())
        r = client.post("/api/auth/admin-rpc", json={
            "name": "add_admin", "args": {"p_user_id": "x"},
        }, headers=SESSION_HEADERS)
        assert r.status_code == 403


class TestCsrfGuard:
    """Cookie-authed mutations from a foreign Origin are refused, and every
    /api/auth/* mutation must carry the SPA's custom header — those endpoints
    SET cookies, so cookie presence cannot gate them (login CSRF). Bearer
    requests are not ambient and stay untouched."""

    def test_foreign_origin_rejected(self, client):
        client.cookies.set("ha_at", make_user_jwt())
        client.cookies.set("ha_rt", "rt")
        r = client.post(
            "/api/auth/refresh",
            headers={"Origin": "https://evil.example.com", **SESSION_HEADERS},
        )
        assert r.status_code == 403

    def test_missing_custom_header_rejected(self, client):
        # A cross-site form carries no cookie and no custom header, yet its
        # Set-Cookie would land — refused before reaching the endpoint.
        r = client.post(
            "/api/auth/login",
            json={"email": "a@b.c", "password": "pw"},
        )
        assert r.status_code == 403

    def test_same_origin_allowed(self, client, httpx_mock):
        _mock_gotrue_token(httpx_mock, "refresh_token")
        client.cookies.set("ha_at", make_user_jwt())
        client.cookies.set("ha_rt", "rt")
        r = client.post(
            "/api/auth/refresh",
            headers={"Origin": "http://testserver", **SESSION_HEADERS},
        )
        assert r.status_code == 200

    def test_bearer_skips_guard(self, client, httpx_mock):
        httpx_mock.add_response(
            method="POST",
            url=f"{TEST_SUPABASE_URL}/rest/v1/analyze_logs",
            status_code=201,
            json={},
        )
        client.cookies.set("ha_at", make_user_jwt())
        r = client.post(
            "/analytics/event",
            json={"endpoint": "x", "model_id": "y", "duration_ms": 1, "metadata": {}},
            headers={
                "Authorization": f"Bearer {make_user_jwt()}",
                "Origin": "https://evil.example.com",
            },
        )
        assert r.status_code != 403


class TestNoStore:
    def test_session_responses_are_not_cacheable(self, client):
        r = client.get("/api/auth/me")
        assert r.headers["Cache-Control"] == "no-store"


class TestHostPrefix:
    """Under https the access cookie is `__Host-ha_at` (Secure, host-only);
    readers accept either name."""

    def test_https_sets_prefixed_secure_cookie(self, client, httpx_mock):
        _mock_gotrue_token(httpx_mock, "password")
        r = client.post(
            "/api/auth/login",
            json={"email": "a@b.c", "password": "pw"},
            headers={**SESSION_HEADERS, "x-forwarded-proto": "https"},
        )
        assert r.status_code == 200
        (at,) = _cookie_named(r, "__Host-ha_at")
        assert "Secure" in at and "Path=/" in at
        assert _cookie_named(r, "ha_at") == []

    def test_https_ignores_the_bare_cookie_name(self, client):
        """Cookie tossing guard: on a secure request only `__Host-ha_at`
        counts — a sibling subdomain can plant `Domain=parent` ha_at but
        it must not authenticate anyone."""
        client.cookies.set("ha_at", make_user_jwt())
        r = client.get(
            "/api/account/agent-credentials",
            headers={"x-forwarded-proto": "https"},
        )
        assert r.status_code == 401

    def test_proxy_header_not_trusted_off_fly(self, client, httpx_mock, monkeypatch):
        """Without Fly (or an opt-in), x-forwarded-proto is client-controlled
        and must not upgrade the request to secure cookie handling."""
        monkeypatch.delenv("FLY_APP_NAME")
        _mock_gotrue_token(httpx_mock, "password")
        r = client.post(
            "/api/auth/login",
            json={"email": "a@b.c", "password": "pw"},
            headers={**SESSION_HEADERS, "x-forwarded-proto": "https"},
        )
        assert r.status_code == 200
        # Plain name, no Secure flag — as any other http request.
        assert _cookie_named(r, "__Host-ha_at") == []
        (at,) = _cookie_named(r, "ha_at")
        assert "Secure" not in at

    def test_prefixed_cookie_authenticates(self, client, httpx_mock):
        httpx_mock.add_response(
            method="GET",
            url=re.compile(re.escape(f"{TEST_SUPABASE_URL}/rest/v1/user_agent_credentials") + r".*"),
            json=[],
        )
        httpx_mock.add_response(
            method="GET",
            url=re.compile(re.escape(f"{TEST_SUPABASE_URL}/rest/v1/agent_task_defaults") + r".*"),
            json=[],
        )
        httpx_mock.add_response(
            method="GET",
            url=re.compile(re.escape(f"{TEST_SUPABASE_URL}/rest/v1/profiles") + r".*"),
            json=[],
        )
        client.cookies.set("__Host-ha_at", make_user_jwt())
        r = client.get("/api/account/agent-credentials")
        assert r.status_code == 200

    def test_logout_clears_both_access_names(self, client, httpx_mock):
        httpx_mock.add_response(
            method="POST", url=f"{TEST_SUPABASE_URL}/auth/v1/logout", status_code=204,
        )
        client.cookies.set("__Host-ha_at", make_user_jwt())
        r = client.post(
            "/api/auth/logout",
            headers={**SESSION_HEADERS, "x-forwarded-proto": "https"},
        )
        assert r.status_code == 200
        assert any(c.startswith("__Host-ha_at=") for c in _set_cookies(r))
        assert any(c.startswith("ha_at=") for c in _set_cookies(r))


class TestLogoutRevocation:
    def test_refresh_token_drives_server_side_revocation(self, client, httpx_mock):
        """Access cookie gone, refresh cookie present: redeem it once and
        revoke the session at GoTrue, instead of only forgetting locally."""
        _mock_gotrue_token(httpx_mock, "refresh_token")
        captured = {}

        def _capture_logout(request):
            captured["authorization"] = request.headers.get("authorization")
            from httpx import Response
            return Response(204)

        httpx_mock.add_callback(
            _capture_logout, method="POST",
            url=f"{TEST_SUPABASE_URL}/auth/v1/logout",
        )
        client.cookies.set("ha_rt", "rt-1")
        r = client.post("/api/auth/logout", headers=SESSION_HEADERS)
        assert r.status_code == 200
        assert captured["authorization"] == "Bearer fresh-access"
        assert any(c.startswith("ha_rt=") for c in _set_cookies(r))


class TestLoginRateLimit:
    def test_login_is_rate_limited(self, client, httpx_mock, monkeypatch):
        from backend.service import rate_limit
        monkeypatch.setitem(rate_limit._LIMITS, "/api/auth/login", 2)
        monkeypatch.setattr(rate_limit, "_buckets", {})
        _mock_gotrue_token(httpx_mock, "password", reusable=True)
        codes = [
            client.post(
                "/api/auth/login",
                json={"email": "a@b.c", "password": "pw"},
                headers=SESSION_HEADERS,
            ).status_code
            for _ in range(3)
        ]
        assert codes == [200, 200, 429]
