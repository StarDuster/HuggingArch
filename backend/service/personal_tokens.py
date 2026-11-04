"""Personal access tokens (PAT) — long-lived credentials for headless upload.

A signed-in user mints an opaque token in the web account page and presents it
as `Authorization: Bearer <token>` from a GPU box (`hbench upload`), where a
1-hour Supabase JWT is impractical. Only a SHA-256 hash is stored
(public.personal_access_tokens, migration 0021) via the service-role REST
endpoint; the plaintext is returned ONCE at creation and never recoverable.

Service-layer module: the OSS core never imports this. Verification is one-way
(hash + lookup), so unlike BYOK agent keys there is no encryption/decryption —
we never need the plaintext back, only to recognize it.
"""
from __future__ import annotations

import hashlib
import logging

from backend import supabase_env, supabase_rest
import secrets
from datetime import datetime, timezone
from typing import Optional

import httpx


logger = logging.getLogger(__name__)

# Distinguishable prefix so the auth layer can tell a PAT from a JWT by shape,
# and so a leaked token is greppable. The body is URL-safe random.
TOKEN_PREFIX = "hba_"


def _supabase() -> tuple[Optional[str], Optional[str]]:
    return (
        supabase_env.url(),
        supabase_env.service_role_key(),
    )


def enabled() -> bool:
    url, key = _supabase()
    return bool(url and key)


def _table_url(url: str) -> str:
    return f"{url}/rest/v1/personal_access_tokens"


def looks_like_pat(token: str) -> bool:
    return bool(token) and token.startswith(TOKEN_PREFIX)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue(*, user_id: str, name: str = "") -> Optional[dict]:
    """Mint a token for the user. Returns {id, token, token_prefix, name} with
    the PLAINTEXT token (shown once) on success, None if unconfigured. Raises on
    a hard Supabase failure (this is a deliberate user action)."""
    url, key = _supabase()
    if not (url and key):
        return None
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    prefix = token[:12]
    with httpx.Client(timeout=10.0) as client:
        r = client.post(
            _table_url(url),
            headers=supabase_rest.service_headers(key, prefer="return=representation"),
            json={"user_id": user_id, "name": name or "",
                  "token_hash": _hash(token), "token_prefix": prefix},
        )
    if r.status_code >= 400:
        raise RuntimeError(f"token issue failed: {r.status_code} {r.text[:200]}")
    row = (r.json() or [{}])[0]
    return {"id": row.get("id"), "token": token, "token_prefix": prefix, "name": name or ""}


def verify(token: str) -> Optional[str]:
    """Return the owning user_id for a live (unrevoked) token, else None. Best-
    effort stamps last_used_at. Silent-fail (a lookup outage denies, never
    raises into the request)."""
    if not looks_like_pat(token):
        return None
    url, key = _supabase()
    if not (url and key):
        return None
    try:
        with httpx.Client(timeout=8.0) as client:
            r = client.get(_table_url(url), params={
                "token_hash": f"eq.{_hash(token)}",
                "revoked_at": "is.null",
                "select": "id,user_id",
            }, headers=supabase_rest.service_headers(key))
            if r.status_code >= 400:
                logger.warning(f"token verify failed: {r.status_code} {r.text[:200]}")
                return None
            rows = r.json() or []
            if not rows:
                return None
            row = rows[0]
            # last_used_at is a nice-to-have; don't let it block auth.
            try:
                client.patch(
                    _table_url(url), params={"id": f"eq.{row['id']}"},
                    headers=supabase_rest.service_headers(key, prefer="return=minimal"),
                    json={"last_used_at": datetime.now(timezone.utc).isoformat()})
            except httpx.RequestError:
                pass
            return row.get("user_id")
    except httpx.RequestError as e:
        logger.warning(f"token verify errored: {e}")
        return None


def list_for_user(user_id: str) -> list[dict]:
    """A user's tokens (newest first) — metadata only, never the hash."""
    url, key = _supabase()
    if not (url and key):
        return []
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(_table_url(url), params={
                "user_id": f"eq.{user_id}",
                "select": "id,name,token_prefix,created_at,last_used_at,revoked_at",
                "order": "created_at.desc",
            }, headers=supabase_rest.service_headers(key))
        if r.status_code >= 400:
            logger.warning(f"token list failed: {r.status_code} {r.text[:200]}")
            return []
        return r.json() or []
    except httpx.RequestError as e:
        logger.warning(f"token list errored: {e}")
        return []


def revoke(*, token_id: str, user_id: str) -> bool:
    """Revoke one of the caller's tokens (owner-scoped). Returns True on a hit."""
    url, key = _supabase()
    if not (url and key):
        return False
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.patch(
                _table_url(url),
                params={"id": f"eq.{token_id}", "user_id": f"eq.{user_id}",
                        "revoked_at": "is.null"},
                headers=supabase_rest.service_headers(key, prefer="return=representation"),
                json={"revoked_at": datetime.now(timezone.utc).isoformat()})
        if r.status_code >= 400:
            return False
        return bool(r.json() or [])
    except httpx.RequestError:
        return False
