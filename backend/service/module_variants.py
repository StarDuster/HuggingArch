"""user_module_variants data access (PostgREST, service role).

A module variant is a NAMED custom parameterization of a library component
(`component`, `dims`) that a user saved from the Playground. Rows are only
ever read/written here (never by the eval core); every query pins
`user_id` explicitly because the service role bypasses RLS. Returns
plain dicts / None / bools like `spec_generations` — HTTP shaping lives in
`backend.service.routes.modules`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from backend import supabase_env, supabase_rest

logger = logging.getLogger(__name__)

_TABLE = "user_module_variants"
_SELECT = "id,component,name,dims,base_model_id,updated_at"


def _svc() -> tuple[str, dict]:
    """(rest_base_url, service-role headers)."""
    url, key = supabase_env.service_role()
    return f"{url}/rest/v1/{_TABLE}", supabase_rest.service_headers(key)


def list_for_user(user_id: str, component: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """A user's own variants (component asc, name asc), narrowed to one
    component when given (the Playground's per-component "mine" section).
    None on backend failure — an empty LIST means genuinely nothing saved,
    and the route must be able to tell the two apart."""
    if not supabase_env.configured() or not user_id:
        return []
    base, headers = _svc()
    params: Dict[str, Any] = {
        "user_id": f"eq.{user_id}",
        "select": _SELECT,
        "order": "component.asc,name.asc",
    }
    if component:
        params["component"] = f"eq.{component}"
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(base, params=params, headers=headers)
        if r.status_code >= 400:
            logger.warning(f"module_variants list failed: {r.status_code} {r.text[:200]}")
            return None
        return r.json() or []
    except httpx.RequestError as e:
        logger.warning(f"module_variants list errored: {e}")
        return None


def create(user_id: str, *, component: str, name: str, dims: Dict[str, int],
           base_model_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Insert (or overwrite the (user, component, name) row) and return it.
    Upsert keyed on the natural unique tuple so re-saving a name refreshes
    the dims instead of erroring."""
    if not supabase_env.configured() or not user_id:
        return None
    base, headers = _svc()
    headers["Prefer"] = "return=representation,resolution=merge-duplicates"
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(base, params={"on_conflict": "user_id,component,name"},
                            json=[{"user_id": user_id, "component": component,
                                   "name": name, "dims": dims,
                                   "base_model_id": base_model_id}],
                            headers=headers)
        if r.status_code >= 400:
            logger.warning(f"module_variants create failed: {r.status_code} {r.text[:200]}")
            return None
        rows = r.json() or []
        return rows[0] if rows else None
    except httpx.RequestError as e:
        logger.warning(f"module_variants create errored: {e}")
        return None


def update(user_id: str, variant_id: str, *, name: str) -> Optional[bool]:
    """Rename one OWN row (user_id pinned). True = updated, False = not
    found / not owned, None = backend failure."""
    if not supabase_env.configured() or not user_id:
        return False
    base, headers = _svc()
    headers["Prefer"] = "return=representation"
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.patch(base, params={"id": f"eq.{variant_id}",
                                           "user_id": f"eq.{user_id}"},
                             json={"name": name}, headers=headers)
        if r.status_code == 409:
            raise _NameConflict(name)
        if r.status_code >= 400:
            logger.warning(f"module_variants update failed: {r.status_code} {r.text[:200]}")
            return None
        return bool(r.json())
    except httpx.RequestError as e:
        logger.warning(f"module_variants update errored: {e}")
        return None


class _NameConflict(Exception):
    """The new name collides with another of the user's variants (unique key)."""


def delete(user_id: str, variant_id: str) -> Optional[bool]:
    """Delete one OWN row (user_id pinned). True = deleted, False = not
    found / not owned, None = backend failure."""
    if not supabase_env.configured() or not user_id:
        return False
    base, headers = _svc()
    headers["Prefer"] = "return=representation"
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.delete(base, params={"id": f"eq.{variant_id}",
                                            "user_id": f"eq.{user_id}"},
                              headers=headers)
        if r.status_code >= 400:
            logger.warning(f"module_variants delete failed: {r.status_code} {r.text[:200]}")
            return None
        return bool(r.json())
    except httpx.RequestError as e:
        logger.warning(f"module_variants delete errored: {e}")
        return None
