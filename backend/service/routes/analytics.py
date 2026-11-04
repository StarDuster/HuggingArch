"""Analytics + admin endpoints.

  • POST /analytics/event   — anonymous-safe usage event ingest. Silent-
    fails on Supabase outage so client UX isn't blocked by analytics.
  • GET  /api/admin/health  — admin-gated stub. /api/admin/* is the
    REST API namespace; /admin/ (no /api) serves the React admin page.
"""
from __future__ import annotations

from typing import Optional

from fastapi import Depends

from backend.api._app import app
from backend.service import auth as _auth
from backend.service.models import AnalyticsEventRequest


@app.post("/analytics/event")
def analytics_event(
    req: AnalyticsEventRequest,
    user: Optional[dict] = Depends(_auth.get_current_user),
):
    """Ingest one usage event. Anonymous-safe — `user_id` lands as NULL
    when no JWT is present. Silent-fails on Supabase outage."""
    user_id = user.get("sub") if user else None
    _auth.insert_analyze_log(
        endpoint=req.endpoint,
        model_id=req.model_id,
        duration_ms=req.duration_ms,
        metadata=req.metadata or {},
        user_id=user_id,
    )
    return {"ok": True}


@app.get("/api/admin/health")
def admin_health(user: dict = Depends(_auth.require_admin)):
    """Stub endpoint that proves the require_admin dependency chain works
    end-to-end. /api/admin/* is the REST namespace; the /admin/ prefix
    serves the React admin page. Returns 401 for anon, 403
    for non-admin, 200 for admin."""
    return {"admin": True, "user_id": user.get("sub")}
