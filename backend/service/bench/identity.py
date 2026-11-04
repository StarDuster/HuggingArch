"""Who a request to the Kernel Bench service is for (docs/measured_calibration.md
§5.1).

The service authenticates no one. The HA API does — a session, or a personal
access token where the route accepts one — and forwards the caller as two
headers it alone sets: `X-HA-User` (the user id; absent or empty for an
anonymous caller) and `X-HA-Admin` (`1` for an admin). Those headers are
believed only on a request that also carries the shared secret in
`X-HA-Internal` from a loopback peer (`server.InternalOnly`); the API strips
every inbound `X-HA-*` header before it sets its own, so a browser can never
name an identity.

The dependencies below answer as `backend.service.auth`'s do: 401 without a
caller where one is required, 403 for a non-admin on an admin route. A caller
is the claims-shaped dict the routes read (`{"sub": <user id>, "admin":
bool}`).
"""
from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, Request

from backend.measure.lookup import ADMIN_HEADER, USER_HEADER


def get_current_user(request: Request) -> Optional[dict]:
    """The caller the API named, None for an anonymous request."""
    user = (request.headers.get(USER_HEADER) or "").strip()
    if not user:
        return None
    return {"sub": user, "admin": request.headers.get(ADMIN_HEADER, "").strip() == "1"}


def require_user(request: Request) -> dict:
    user = get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Sign-in required")
    return user


def require_admin(request: Request) -> dict:
    user = require_user(request)
    if not user["admin"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def uid(user: dict) -> str:
    """The caller's user id; 401 when the caller names none."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user identity")
    return user_id


def is_admin(user: Optional[dict]) -> bool:
    return bool(user and user.get("admin"))
