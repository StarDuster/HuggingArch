"""Service-role reads/writes of a few `public.profiles` fields that the
service layer needs outside the avatar flow — currently the optional
`github_handle` used to credit a contributor on their spec PR (Phase 5).

Kept separate from the route modules so both the account route (owner writes
their own handle) and the admin route (reads a contributor's handle when
opening the PR) can share one implementation.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import httpx
from pydantic import BaseModel

from backend import supabase_rest

from . import auth as _auth

logger = logging.getLogger(__name__)

# GitHub usernames: 1–39 chars, alphanumeric or single hyphens, no leading/
# trailing hyphen. We validate on write so a bad value can't reach a commit
# trailer. A leading '@' is tolerated and stripped.
_HANDLE_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")


def normalize_handle(raw: Optional[str]) -> Optional[str]:
    """Return a clean handle (no '@'), or None if empty/invalid."""
    if not raw:
        return None
    h = raw.strip().lstrip("@")
    if not h:
        return None
    return h if _HANDLE_RE.match(h) else None


def _rest(method: str, params: dict, json_body: Optional[dict] = None):
    url = _auth._supabase_url()
    key = _auth._service_role_key()
    if not (url and key):
        return None
    with httpx.Client(timeout=10.0) as cx:
        return cx.request(
            method, f"{url.rstrip('/')}/rest/v1/profiles",
            params=params, json=json_body, headers=supabase_rest.service_headers(key),
        )


def get_github_handle(user_id: str) -> Optional[str]:
    """The user's stored GitHub handle, or None (unset / unconfigured / error)."""
    if not user_id:
        return None
    try:
        r = _rest("GET", {"id": f"eq.{user_id}", "select": "github_handle"})
        if r is None or r.status_code != 200:
            return None
        rows = r.json()
        return normalize_handle(rows[0].get("github_handle")) if rows else None
    except httpx.RequestError as e:
        logger.warning(f"get_github_handle failed: {e}")
        return None


def set_github_handle(user_id: str, handle: Optional[str]) -> bool:
    """Write (or clear, when handle is None) the user's GitHub handle. Returns
    False when the service layer isn't configured; raises on a REST error."""
    r = _rest("PATCH", {"id": f"eq.{user_id}"}, {"github_handle": handle})
    if r is None:
        return False
    if r.status_code >= 400:
        raise RuntimeError(f"profiles patch failed: {r.status_code} {r.text}")
    return True


class EditableProfile(BaseModel):
    """The profile columns an owner may write. Its fields are the whole
    write set: `set_profile` writes these and nothing else."""
    username: str | None = None
    full_name: str | None = None
    bio: str | None = None
    github_handle: str | None = None


def set_profile(user_id: str, fields: EditableProfile) -> bool:
    """Owner-writes their editable profile fields via service-role — the
    `authenticated` role has no direct DML on profiles (see migration 0008),
    so the account page routes its save here instead of a client-side upsert.
    Empty strings become NULL. Returns False when the service layer isn't
    configured; raises on a REST error."""
    body = {k: (v or None) for k, v in fields.model_dump().items()}
    r = _rest("PATCH", {"id": f"eq.{user_id}"}, body)
    if r is None:
        return False
    if r.status_code >= 400:
        raise RuntimeError(f"profiles patch failed: {r.status_code} {r.text}")
    return True


def who(user_ids) -> dict[str, dict]:
    """`{user_id: {"email", "username"}}` for the admin pages that name the
    user behind a record: email from the auth admin API, username from
    profiles. A user the lookup cannot answer for is absent (the page shows
    the id); unconfigured → {}."""
    ids = sorted({u for u in user_ids if u})
    url = _auth._supabase_url()
    key = _auth._service_role_key()
    if not (ids and url and key):
        return {}
    out: dict[str, dict] = {}
    base = url.rstrip("/")
    with httpx.Client(timeout=10.0, headers=supabase_rest.service_headers(key)) as cx:
        try:
            r = cx.get(f"{base}/rest/v1/profiles",
                       params={"id": f"in.({','.join(ids)})", "select": "id,username"})
            if r.status_code == 200:
                for p in r.json() or []:
                    out.setdefault(p["id"], {})["username"] = p.get("username")
            for u in ids:
                r = cx.get(f"{base}/auth/v1/admin/users/{u}")
                if r.status_code == 200:
                    out.setdefault(u, {})["email"] = (r.json() or {}).get("email")
        except httpx.RequestError as e:
            logger.warning(f"who failed: {e}")
    return out
