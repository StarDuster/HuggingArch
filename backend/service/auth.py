"""Supabase JWT verification + FastAPI dependencies.

The browser session lives in an httpOnly cookie (`ha_at`) set by
`backend.service.routes.auth_session`; headless clients (hbench PATs,
scripts, tests) send `Authorization: Bearer <jwt>` instead. Both carry
the same Supabase session JWT, verified here using the project's HS256
secret (the same secret Supabase signs sessions with) or the JWKS for
asymmetric projects. This module exposes three dependencies for routes:

    get_current_user  — None for anonymous, dict of claims for signed-in
    require_user      — 401 if not signed in
    require_admin     — 403 if not in public.admins

Why verify in the backend at all? Most user data flows directly between
the browser and Supabase (RLS handles isolation). But anything the
backend writes on the user's behalf — analytics events tagged to a
user, future Phase-2 endpoints like `/playground/save` — needs to know
who's calling. The JWT secret stays server-only; the public anon_key
is what the frontend uses.
"""
from __future__ import annotations

import time
from typing import Optional

import httpx
import jwt
from fastapi import Depends, Header, HTTPException, Request


# Supabase 2025 projects sign session JWTs with EITHER:
#   • Legacy HS256 + the symmetric JWT Secret  (pre-2025 default)
#   • Modern ES256 / RS256 + asymmetric keys   (new project default)
# We support both. The token's own `alg` header tells us which path.
_ASYMMETRIC_ALGS = {"ES256", "RS256"}
from backend import supabase_env, supabase_rest



def _jwt_secret() -> Optional[str]:
    return supabase_env.jwt_secret()


def _supabase_url() -> Optional[str]:
    return supabase_env.url()


def _service_role_key() -> Optional[str]:
    return supabase_env.service_role_key()


def _anon_key() -> Optional[str]:
    """Public-facing anon key. Required as `apikey` header for any REST
    request that should run as a specific user (the user's JWT goes in
    `Authorization`). Service-role is the wrong choice for those flows
    because it has no `auth.uid()` context — the SQL gate
    `is_admin(auth.uid())` then evaluates against NULL and always
    rejects, which silently broke the admin Cached-Specs panel until
    we noticed.
    """
    return supabase_env.anon_key()



# JWKS client cache — pyjwt's PyJWKClient handles HTTP fetch + key cache
# (default TTL 1 hour), so we only construct one per process per URL.
_jwks_client: Optional["jwt.PyJWKClient"] = None


def _get_jwks_client() -> "jwt.PyJWKClient":
    global _jwks_client
    if _jwks_client is None:
        url = _supabase_url()
        if not url:
            raise HTTPException(status_code=503, detail="Auth not configured")
        # Supabase exposes the JWKS at this path (auth-v1 + .well-known).
        jwks_url = url.rstrip("/") + "/auth/v1/.well-known/jwks.json"
        _jwks_client = jwt.PyJWKClient(jwks_url, cache_keys=True)
    return _jwks_client


def verify_supabase_jwt(token: str) -> dict:
    """Decode + validate a Supabase session JWT. Dispatches by algorithm:
      • HS256                → use SUPABASE_JWT_SECRET (legacy path)
      • ES256 / RS256        → fetch public key from JWKS (modern path)
    Raises HTTPException(401) on any signature / expiry / claim mismatch.
    """
    try:
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Malformed token: {e}") from e

    alg = header.get("alg")
    common_kwargs = dict(audience="authenticated", options={"verify_aud": True})

    try:
        if alg == "HS256":
            secret = _jwt_secret()
            if not secret:
                raise HTTPException(
                    status_code=503,
                    detail="Token uses HS256 but SUPABASE_JWT_SECRET is unset; "
                           "switch the project to asymmetric keys or set the secret.",
                )
            return jwt.decode(token, secret, algorithms=["HS256"], **common_kwargs)
        if alg in _ASYMMETRIC_ALGS:
            try:
                signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
            except Exception as e:
                raise HTTPException(status_code=401, detail=f"JWKS fetch failed: {e}") from e
            return jwt.decode(
                token, signing_key.key, algorithms=list(_ASYMMETRIC_ALGS), **common_kwargs,
            )
        raise HTTPException(status_code=401, detail=f"Unsupported alg: {alg!r}")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired") from None
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}") from e


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def _request_token(
    authorization: Optional[str],
    cookie: Optional[str],
) -> Optional[str]:
    """The caller's session JWT: the Authorization bearer when present
    (hbench PATs, scripts, tests), else the httpOnly session cookie the
    browser sends."""
    bearer = _extract_bearer(authorization)
    if bearer:
        return bearer
    return cookie if isinstance(cookie, str) and cookie else None


#: The access cookie is `__Host-ha_at` under https and `ha_at` under plain
#: http (the `__Host-` prefix requires Secure, which dev doesn't have).
ACCESS_COOKIE = "ha_at"
ACCESS_COOKIE_HOST = "__Host-ha_at"


def session_cookie(request) -> Optional[str]:
    """The access-cookie value, scheme-appropriate: https requests honor
    ONLY `__Host-ha_at` — a sibling subdomain can plant the bare name
    (Domain=parent) but never the prefixed one, so accepting the bare name
    on a secure request would reopen cookie tossing. Plain-http dev accepts
    either. Mirrored in `backend.api._access` (core cannot import service).
    """
    if request is None:
        return None
    if supabase_env.secure_request(request):
        return request.cookies.get(ACCESS_COOKIE_HOST) or None
    return request.cookies.get(ACCESS_COOKIE) or request.cookies.get(ACCESS_COOKIE_HOST)


def _get_current_user(
    authorization: Optional[str],
    cookie: Optional[str],
) -> Optional[dict]:
    if not supabase_env.jwt_verification_available():
        return None
    token = _request_token(authorization, cookie)
    if not token:
        return None
    return verify_supabase_jwt(token)


def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(None),
) -> Optional[dict]:
    """Returns the JWT claims dict for a signed-in user, or None for an
    anonymous request. Auth being disabled (no env vars) also returns
    None — callers shouldn't crash in dev mode.
    """
    return _get_current_user(authorization, session_cookie(request))


def require_user(user: Optional[dict] = Depends(get_current_user)) -> dict:
    if user is None:
        raise HTTPException(status_code=401, detail="Sign-in required")
    return user


def uid(user: dict, *, detail: str = "Missing user identity") -> str:
    """The signed-in caller's user id (the token's `sub`); 401 when a
    verified token carries none."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail=detail)
    return user_id


def require_user_or_pat(
    request: Request = None,
    authorization: Optional[str] = Header(None),
) -> dict:
    """Accept EITHER a personal access token (PAT) as the bearer OR a normal
    session (bearer JWT or the httpOnly cookie). Used by headless-friendly
    routes (e.g. `hbench upload`) where a 1-hour JWT is impractical. A PAT is
    recognized by its prefix and verified by hash — PATs stay header-only;
    the cookie is only consulted on the ordinary JWT path. Returns the same
    `{"sub": user_id}`-shaped dict as `require_user`."""
    user = user_or_pat(authorization, session_cookie(request))
    if user is None:
        raise HTTPException(status_code=401, detail="Sign-in required")
    return user


def user_or_pat(authorization: Optional[str], cookie: Optional[str]) -> Optional[dict]:
    """The caller `require_user_or_pat` accepts — a PAT bearer, else a
    session (bearer JWT or `cookie`, already picked for the scheme) — or None
    when the request carries no credential. A credential that does not
    verify raises 401."""
    token = _extract_bearer(authorization)
    if token:
        from backend.service.personal_tokens import looks_like_pat, verify as verify_pat
        if looks_like_pat(token):
            uid = verify_pat(token)
            if uid:
                return {"sub": uid, "auth": "pat"}
            raise HTTPException(status_code=401, detail="Invalid or revoked access token")
    # Not a PAT → normal JWT path.
    return _get_current_user(authorization, cookie)


def _service_role_client() -> httpx.Client:
    """HTTP client preconfigured with service-role headers. Use for
    privileged calls that need to bypass RLS (e.g. checking admin
    status during route gating). NEVER expose service-role to
    untrusted code paths or the frontend.
    """
    url = _supabase_url()
    key = _service_role_key()
    if not (url and key):
        raise HTTPException(
            status_code=503, detail="Supabase service-role not configured"
        )
    return httpx.Client(
        base_url=url.rstrip("/") + "/rest/v1",
        headers=supabase_rest.service_headers(key),
        timeout=5.0,
    )


#: user_id -> (is admin, monotonic time the answer expires).
_ADMIN_CACHE: dict[str, tuple[bool, float]] = {}
#: How long one lookup stands. A revoked admin (or a deleted user whose JWT
#: has not expired) loses admin access within this, not at the next restart.
_ADMIN_TTL_SECONDS = 300.0


#: user_id -> how many times their answer was forgotten. A lookup records the
#: count it started under and caches its answer only if it is unchanged, so a
#: lookup in flight across `forget_admin` cannot write the old answer back.
_ADMIN_FORGOTTEN: dict[str, int] = {}


def forget_admin(user_id: str) -> None:
    """Drop the cached answer for `user_id` (their row changed or is gone)."""
    _ADMIN_FORGOTTEN[user_id] = _ADMIN_FORGOTTEN.get(user_id, 0) + 1
    _ADMIN_CACHE.pop(user_id, None)


def is_admin(user_id: str) -> bool:
    """Service-role lookup against public.admins, cached per process for
    `_ADMIN_TTL_SECONDS`."""
    cached = _ADMIN_CACHE.get(user_id)
    if cached is not None and cached[1] > time.monotonic():
        return cached[0]
    generation = _ADMIN_FORGOTTEN.get(user_id, 0)
    with _service_role_client() as client:
        resp = client.get(
            "/admins",
            params={"user_id": f"eq.{user_id}", "select": "user_id"},
        )
    if resp.status_code != 200:
        # Service-role request failed — don't cache the negative.
        return False
    found = bool(resp.json())
    if _ADMIN_FORGOTTEN.get(user_id, 0) == generation:
        _ADMIN_CACHE[user_id] = (found, time.monotonic() + _ADMIN_TTL_SECONDS)
    return found


def require_admin(user: dict = Depends(require_user)) -> dict:
    user_id = user.get("sub")
    if not user_id or not is_admin(user_id):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def require_admin_if_enabled(
    authorization: Optional[str] = None,
    cookie: Optional[str] = None,
) -> Optional[dict]:
    """Service callback for privileged routes that also exist in core mode.

    No Supabase URL means a local/single-user deployment, where these routes
    retain their original open behavior. Once auth is configured, callers
    must present a valid admin JWT (bearer header or session cookie — the
    scheme-appropriate cookie is picked by `backend.api._access`).
    """
    if not supabase_env.jwt_verification_available():
        return None
    user = _get_current_user(authorization, cookie)
    if user is None:
        raise HTTPException(status_code=401, detail="Sign-in required")
    return require_admin(user)


def current_token(
    request: Request,
    user: dict = Depends(require_user),
    authorization: Optional[str] = Header(None),
) -> str:
    """Returns the raw JWT for an authenticated request (cookie or bearer).

    `require_user` runs first, so reaching the fallback without a token
    means the credentials were stripped between dependencies — surface as
    401 rather than 500.
    """
    token = _request_token(authorization, session_cookie(request))
    if not token:
        raise HTTPException(status_code=401, detail="Sign-in required")
    return token


def admin_token(
    request: Request,
    user: dict = Depends(require_admin),
    authorization: Optional[str] = Header(None),
) -> str:
    """Returns the raw JWT for an admin-authenticated request.

    `require_admin` runs first (FastAPI-side gate), then we re-extract
    the token from the same request. Forwarding this JWT to Supabase
    REST as `Authorization: Bearer <jwt>` (alongside the anon key as
    `apikey:`) makes `auth.uid()` resolve to the admin's user-id inside
    SQL — so SECURITY DEFINER functions whose body checks
    `is_admin(auth.uid())` actually work.

    Why not service-role: service-role bypasses RLS but has no user
    identity (`auth.uid()` is NULL), so the SQL `is_admin()` defense-
    in-depth check inside the cached-specs admin RPCs always rejected
    every call. The functions are deliberately gated twice (FastAPI
    here + SQL inside) so a future route that forgets `require_admin`
    still can't reach the data; restoring the second gate requires
    propagating real user identity, which means the user's JWT.
    """
    token = _request_token(authorization, session_cookie(request))
    if not token:
        raise HTTPException(status_code=401, detail="Sign-in required")
    return token


def insert_analyze_log(
    *,
    endpoint: str,
    model_id: str,
    duration_ms: Optional[int],
    metadata: Optional[dict],
    user_id: Optional[str],
) -> None:
    """Insert one row into public.analyze_logs via the service-role
    REST endpoint. Anonymous events (user_id=None) are kept — they
    feed traffic graphs that include logged-out users.
    Silent-fail by design: analytics outages must NOT break /analyze.
    """
    if not (_supabase_url() and _service_role_key()):
        return
    payload = {
        "endpoint": endpoint,
        "model_id": model_id,
        "duration_ms": duration_ms,
        "metadata": metadata or {},
        "user_id": user_id,
    }
    try:
        with _service_role_client() as client:
            client.post("/analyze_logs", json=payload)
    except Exception:
        # Analytics MUST NOT break user-facing endpoints. Drop quietly.
        pass
