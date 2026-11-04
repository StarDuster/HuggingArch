"""Service layer — auth + admin + analytics, and the Kernel Bench service.

This package is opt-in by deployment: when the SUPABASE_* env vars are
unset, every endpoint registered here returns a degraded response (e.g.
`/site-config.js` reports `authEnabled:false`) and the React SPA hides
login UI on its own.

The boundary with the OSS core (`backend.api`, `backend.arch`,
`backend.inference`, `backend.rendering`, `backend.hf`) is intentional:
nothing under `backend.service` is imported by the core. Conversely,
service routes import from `backend.api._app` (the shared FastAPI
instance) and `backend.api._helpers` only — never the other way
around. If you want to fork off a "no-service" build, dropping this
directory + the bottom lines of `backend/api/__init__.py` is enough.

Importing the package does nothing: the Kernel Bench service
(`backend.service.bench.server`, its own process) lives under it and must
not build the HA API. The HA API imports `backend.service.ha_api`, which:
  • registers the /site-config.js, /analytics/event, /api/admin/*,
    /api/account/* HTTP routes via `from . import routes`;
  • installs an in-process rate limiter middleware against /analyze
    and the threaded /inference/* routes (configurable via env, see rate_limit.py);
  • forwards /api/bench/* to the Kernel Bench service (`bench_proxy`).
  Auth / account / admin HTML is the React SPA (`spa_page_response`).
"""
