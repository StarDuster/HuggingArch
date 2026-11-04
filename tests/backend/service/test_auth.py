"""backend.service.auth — JWT verification + admin lookup.

Covers both signing-algorithm paths (legacy HS256 + modern ES256/JWKS),
the dependency chain (`get_current_user` → `require_user` → `require_admin`),
and the in-process admin cache.
"""
from __future__ import annotations

import pytest

from .conftest import (
    TEST_SUPABASE_URL,
    make_user_jwt,
    make_expired_jwt,
    mock_admins_no,
    mock_admins_yes,
    sign_es256,
)


# ---------------------------------------------------------------------------
# verify_supabase_jwt — happy paths and failure modes
# ---------------------------------------------------------------------------

class TestVerifyHS256:
    def test_accepts_valid_token(self, user_token):
        from backend.service.auth import verify_supabase_jwt
        claims = verify_supabase_jwt(user_token)
        assert claims["sub"] == "user-uuid-1"
        assert claims["aud"] == "authenticated"

    def test_rejects_expired_token(self):
        from fastapi import HTTPException
        from backend.service.auth import verify_supabase_jwt
        token = make_expired_jwt()
        with pytest.raises(HTTPException) as exc:
            verify_supabase_jwt(token)
        assert exc.value.status_code == 401
        assert "expired" in exc.value.detail.lower()

    def test_rejects_wrong_audience(self):
        from fastapi import HTTPException
        from backend.service.auth import verify_supabase_jwt
        token = make_user_jwt(audience="anon")  # supabase service-role uses different aud
        with pytest.raises(HTTPException) as exc:
            verify_supabase_jwt(token)
        assert exc.value.status_code == 401
        assert "invalid" in exc.value.detail.lower()

    def test_rejects_tampered_signature(self, user_token):
        from fastapi import HTTPException
        from backend.service.auth import verify_supabase_jwt
        # Replace the entire signature with a deterministic-but-wrong
        # base64url string of the right length. (Flipping one char is
        # flaky — base64 decodes 6-bit chunks and many single-char
        # flips fall in padding bits where the byte signature is
        # unchanged.)
        head, payload, sig = user_token.rsplit(".", 2)
        bogus = "x" * len(sig)
        tampered = f"{head}.{payload}.{bogus}"
        with pytest.raises(HTTPException) as exc:
            verify_supabase_jwt(tampered)
        assert exc.value.status_code == 401

    def test_rejects_malformed(self):
        from fastapi import HTTPException
        from backend.service.auth import verify_supabase_jwt
        with pytest.raises(HTTPException) as exc:
            verify_supabase_jwt("not.a.jwt")
        assert exc.value.status_code == 401

    def test_returns_503_when_jwt_secret_unset(self, user_token, monkeypatch):
        from fastapi import HTTPException
        from backend.service.auth import verify_supabase_jwt
        monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
        with pytest.raises(HTTPException) as exc:
            verify_supabase_jwt(user_token)
        assert exc.value.status_code == 503


class TestVerifyES256:
    """When Supabase signs sessions with the new asymmetric keys, the
    token header is {alg: ES256, kid: ...}. Verifier must fetch the
    JWKS, find the matching key, and validate."""

    def test_accepts_valid_es256_token(self, ec_keypair, monkeypatch):
        """PyJWKClient uses urllib internally, so we replace _get_jwks_client
        with a stub that yields a SigningKey wrapping our test public key."""
        import time
        from cryptography.hazmat.primitives import serialization
        from backend.service import auth as _auth
        priv_pem, _jwk = ec_keypair

        # Recover the public key from the private PEM (cheaper than
        # rebuilding from JWK in test code).
        priv = serialization.load_pem_private_key(priv_pem, password=None)
        pub = priv.public_key()

        class _StubKey:
            def __init__(self, key):
                self.key = key

        class _StubJWKS:
            def get_signing_key_from_jwt(self, _token):
                return _StubKey(pub)

        monkeypatch.setattr(_auth, "_get_jwks_client", _StubJWKS)

        now = int(time.time())
        token = sign_es256({
            "sub": "abc", "aud": "authenticated",
            "iat": now, "exp": now + 600,
        }, priv_pem)
        claims = _auth.verify_supabase_jwt(token)
        assert claims["sub"] == "abc"

    def test_jwks_fetch_failure_returns_401(self, ec_keypair, monkeypatch):
        """When JWKS endpoint is unreachable, surface a clean 401 instead
        of leaking urllib internals."""
        from fastapi import HTTPException
        from backend.service import auth as _auth
        priv_pem, _ = ec_keypair

        def _explode():
            raise RuntimeError("simulated network failure")
        monkeypatch.setattr(_auth, "_get_jwks_client", _explode)

        import time
        token = sign_es256({"sub": "x", "aud": "authenticated",
                            "iat": int(time.time()), "exp": int(time.time()) + 60},
                           priv_pem)
        with pytest.raises(HTTPException) as exc:
            _auth.verify_supabase_jwt(token)
        assert exc.value.status_code == 401
        assert "jwks" in exc.value.detail.lower()

    def test_rejects_unknown_alg(self):
        from fastapi import HTTPException
        from backend.service.auth import verify_supabase_jwt
        # PyJWT lets us forge tokens with unsupported alg via raw encoding.
        # Easiest: an HS512 token (HS algos only HS256 in our allowlist).
        import jwt
        bad = jwt.encode({"sub": "x"}, "k", algorithm="HS512")
        with pytest.raises(HTTPException) as exc:
            verify_supabase_jwt(bad)
        assert exc.value.status_code == 401
        assert "unsupported alg" in exc.value.detail.lower()


# ---------------------------------------------------------------------------
# auth_enabled() — env-driven feature flag
# ---------------------------------------------------------------------------

class TestSupabaseMode:
    """Three questions, three answers, all read off `backend.supabase_env`:
    can the server call Supabase (URL + service key), can it verify a user's
    JWT (URL), can the browser sign in (URL + anon key). They differ on real
    deployments — a URL alone is a verifying single-tenant box, URL + anon
    key is a static site — so no single "auth on" flag stands in for them."""

    def test_three_questions_three_answers(self, monkeypatch):
        from backend import supabase_env as env
        assert env.configured() and env.jwt_verification_available() and env.browser_auth_available()
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        assert not env.configured()
        assert env.jwt_verification_available() and env.browser_auth_available()
        monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
        assert env.jwt_verification_available() and not env.browser_auth_available()
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        assert not env.jwt_verification_available()

    def test_the_service_layer_has_no_second_definition(self):
        import backend.service.auth as auth
        assert not hasattr(auth, "auth_enabled")

    def test_anonymous_specgen_follows_jwt_verification(self, monkeypatch):
        """B15: the core gate reads the same declaration, without a net.
        With a URL set (multi-user) anonymous generation is refused."""
        from backend.api.routes import spec as core_spec
        assert "_anon_specgen_gate" not in vars(core_spec)
        from backend import supabase_env as env
        assert env.jwt_verification_available()
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        assert not env.jwt_verification_available()


# ---------------------------------------------------------------------------
# is_admin() — service-role lookup against /rest/v1/admins, with caching
# ---------------------------------------------------------------------------

class TestIsAdminLookup:
    def test_returns_true_for_existing_admin(self, httpx_mock):
        from backend.service.auth import is_admin
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        assert is_admin("admin-uuid-1") is True

    def test_returns_false_for_non_admin(self, httpx_mock):
        from backend.service.auth import is_admin
        mock_admins_no(httpx_mock, "regular-uuid")
        assert is_admin("regular-uuid") is False

    def test_caches_per_user(self, httpx_mock):
        """Second call must NOT hit the network."""
        from backend.service.auth import is_admin
        # Register a single (non-reusable) response
        httpx_mock.add_response(
            method="GET",
            url=f"{TEST_SUPABASE_URL}/rest/v1/admins?user_id=eq.cached-uuid&select=user_id",
            json=[{"user_id": "cached-uuid"}],
            is_reusable=False,
        )
        assert is_admin("cached-uuid") is True
        # If the cache wasn't honored, this second call would explode
        # because httpx_mock has no second response queued.
        assert is_admin("cached-uuid") is True

    def test_does_not_cache_negative_on_error(self, httpx_mock):
        """A 500 from supabase shouldn't poison the cache with False."""
        from backend.service.auth import is_admin
        httpx_mock.add_response(
            method="GET",
            url=f"{TEST_SUPABASE_URL}/rest/v1/admins?user_id=eq.flaky&select=user_id",
            status_code=500,
            text="boom",
            is_reusable=False,
        )
        assert is_admin("flaky") is False
        # Next call: planet a real 200 → must NOT be served the cached False
        mock_admins_yes(httpx_mock, "flaky")
        assert is_admin("flaky") is True


# ---------------------------------------------------------------------------
# require_user / require_admin via TestClient
# ---------------------------------------------------------------------------

class TestRequireUser:
    """Use POST /analytics/event as a probe — it depends on get_current_user."""

    def test_anonymous_accepted_by_get_current_user(self, client):
        # No bearer token — analytics still 200 (anonymous-safe ingestion)
        # because the route uses get_current_user (None on miss), not
        # require_user.
        r = client.post("/analytics/event", json={"endpoint": "analyze", "model_id": "x"})
        assert r.status_code == 200, r.text

    def test_invalid_bearer_rejected(self, client):
        # Bearer present but unparseable → 401 from verify_supabase_jwt
        r = client.post(
            "/analytics/event",
            json={"endpoint": "analyze", "model_id": "x"},
            headers={"Authorization": "Bearer not.a.jwt"},
        )
        assert r.status_code == 401


class TestRequireAdmin:
    """Probe via /api/admin/health — it's gated by require_admin."""

    def test_anonymous_returns_401(self, client):
        r = client.get("/api/admin/health")
        assert r.status_code == 401

    def test_non_admin_user_returns_403(self, client, user_token, httpx_mock):
        mock_admins_no(httpx_mock, "user-uuid-1")
        r = client.get("/api/admin/health", headers={"Authorization": f"Bearer {user_token}"})
        assert r.status_code == 403

    def test_admin_user_returns_200(self, client, admin_token, httpx_mock):
        mock_admins_yes(httpx_mock, "admin-uuid-1")
        r = client.get("/api/admin/health", headers={"Authorization": f"Bearer {admin_token}"})
        assert r.status_code == 200
        body = r.json()
        assert body["admin"] is True
        assert body["user_id"] == "admin-uuid-1"
