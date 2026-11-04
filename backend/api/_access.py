"""Service-injected request guards for core routes.

The core API cannot import ``backend.service.auth`` without reversing the
core/service dependency. Service mode therefore installs callbacks at app
startup, each given only the request's bearer header and its
scheme-appropriate access cookie:

  * the admin gate for privileged core routes — without it, or when it
    decides auth is not configured, self-hosted routes keep their original
    behavior;
  * the owner resolver — the caller's user id for an estimate or sweep
    asking `calibrate_with_mine` (docs/measured_calibration.md §7.1), the
    owner of its measurement scope. Without it such a request is refused
    (501): only the service knows who the caller is.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional, TypeVar

import asyncio

from fastapi import Header, HTTPException, Request

from backend import supabase_env
from backend.api._models import InferenceEstimateRequest
from backend.measure import lookup as _measure_lookup


AdminAccessGate = Callable[[Optional[str], Optional[str]], Any]

_admin_access_gate: Optional[AdminAccessGate] = None


def set_admin_access_gate(gate: Optional[AdminAccessGate]) -> None:
    """Install the service-owned conditional admin check."""
    global _admin_access_gate
    _admin_access_gate = gate


def _session_cookie(request: Request) -> Optional[str]:
    """Mirror of `backend.service.auth.session_cookie` — core cannot import
    the service layer, so the scheme-appropriate pick is duplicated here.
    https honors only `__Host-ha_at`; plain-http dev accepts either."""
    if supabase_env.secure_request(request):
        return request.cookies.get("__Host-ha_at")
    return request.cookies.get("ha_at") or request.cookies.get("__Host-ha_at")


def require_admin_if_configured(
    request: Request,
    authorization: Optional[str] = Header(None),
) -> Any:
    """Require an admin in service mode; stay open in local core mode."""
    if _admin_access_gate is None:
        return None
    return _admin_access_gate(authorization, _session_cookie(request))


#: (authorization, cookie) -> the caller's user id, or None when the
#: request carries no session; raises (401) for credentials that do not verify.
OwnerResolver = Callable[[Optional[str], Optional[str]], Optional[str]]

_owner_resolver: Optional[OwnerResolver] = None


def set_owner_resolver(resolver: Optional[OwnerResolver]) -> None:
    """Install the service-owned session → user id resolver."""
    global _owner_resolver
    _owner_resolver = resolver


_Req = TypeVar("_Req", bound=InferenceEstimateRequest)


async def with_measurement_scope(req: _Req, request: Optional[Request] = None,
                                 authorization: Optional[str] = None) -> _Req:
    """`req` with its `measurement_scope` set (docs/measured_calibration.md
    §7.1) — in this process, before the request is offloaded, joined or
    keyed: the scope is part of every identity its result goes under.

    The owner: the caller's, from their session, when the request asks
    `calibrate_with_mine` and reads measurements (401 without a valid
    session, 501 without the service layer); None otherwise. The rest:
    `measure.lookup.resolve_scope` — the constant `NONE` for a request that
    reads no measurements (`reads_measurements`: `theoretical_only`, or a
    sweep's SOL answer — no request to the service, so theory never depends
    on it) or with no service configured; else one `/internal/bench/version`
    (503 `MeasurementsUnavailable` when the service does not answer). The
    resolver verifies a token (it may fetch keys) and the version is a
    request, so both run off the event loop."""
    reads = req.reads_measurements()
    owner = None
    if reads and req.calibrate_with_mine:
        resolver = _owner_resolver
        if resolver is None:
            raise HTTPException(status_code=501,
                                detail="calibration with your rows needs the service")
        cookie = _session_cookie(request) if request is not None else None
        owner = await asyncio.to_thread(resolver, authorization, cookie)
        if owner is None:
            raise HTTPException(status_code=401,
                                detail="sign in to calibrate with your rows")
    scope = await asyncio.to_thread(_measure_lookup.resolve_scope, reads=reads,
                                    owner=owner)
    return req.model_copy(update={"measurement_scope": scope})


__all__ = ["require_admin_if_configured", "set_admin_access_gate",
           "set_owner_resolver", "with_measurement_scope"]
