"""Misc meta endpoints: index page, health, dim glossary, debug.

These are the small endpoints that don't fit any business domain. Each
is independent and short. Service-only endpoints (e.g. /site-config.js)
live in backend/service/routes/meta.py.
"""
from __future__ import annotations

import ipaddress

from fastapi import Depends, HTTPException, Request
from fastapi.responses import RedirectResponse

from backend.agents import available_clis, get_backend
from backend.arch import dims
from backend.spec_worker import runtime as _rt

from .._access import require_admin_if_configured
from .._app import app, health_details, public_file_response, spa_page_response
from .._models import HealthResponse


@app.get("/")
async def root():
    """React SPA; `web/src/App.tsx` routes `/` to the Inspector page."""
    return spa_page_response()


# The Inspector owns `/`; its own URL stays for the links that name it.
@app.get("/inspector.html")
async def inspector_page():
    """React SPA; routes to `web/src/pages/Inspector.tsx`."""
    return spa_page_response()


@app.get("/gallery.html")
async def gallery_page():
    """React SPA; routes to `web/src/pages/Gallery.tsx`."""
    return spa_page_response()


# Compare is a page-internal mode of Inspector/Inference; old shareable
# links still land here, and `web/src/pages/CompareRedirect.tsx` remaps
# the legacy hash.
@app.get("/compare.html")
async def compare_page():
    return spa_page_response()


@app.get("/inference.html")
async def inference_page():
    """React SPA; routes to `web/src/pages/Inference.tsx`."""
    return spa_page_response()


@app.get("/playground.html")
async def playground_page():
    """React Playground SPA (module comparison)."""
    return spa_page_response()


@app.get("/fusion.html")
async def fusion_page():
    """React SPA; routes to `web/src/pages/Fusion.tsx`."""
    return spa_page_response()


@app.get("/kernel-corpus.html")
async def kernel_corpus_page():
    """React SPA; routes to `web/src/pages/Corpus.tsx`."""
    return spa_page_response()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", spec_jobs_running=_rt.running_job_count(),
                          detail=await health_details())


def _require_local_supervisor(request: Request) -> None:
    """Loopback callers only (IPv4, IPv6, and IPv4-mapped IPv6). A request
    relayed by the Cloudflare tunnel also arrives from loopback but carries
    `Cf-Connecting-Ip` / `X-Forwarded-For`, and is refused (404) like any
    remote caller."""
    try:
        addr = ipaddress.ip_address(request.client.host if request.client else "")
    except ValueError:
        raise HTTPException(status_code=404) from None
    if addr.version == 6 and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped  # ::ffff:127.0.0.1 on a dual-stack listener
    loopback = addr.is_loopback
    relayed = ("cf-connecting-ip" in request.headers
               or "x-forwarded-for" in request.headers)
    if not loopback or relayed:
        raise HTTPException(status_code=404)


@app.get("/health/drain")
async def health_drain_state(request: Request):
    """Restart drain state: whether new spec jobs are refused, and the jobs
    (generate / refine / review / fix, including ones still being admitted)
    a restart would interrupt. The local supervisor (`ha restart`) restarts
    only once `in_flight` is empty."""
    _require_local_supervisor(request)
    return _rt.drain_state()


@app.post("/health/drain")
async def health_drain_start(request: Request):
    """Stop admitting spec jobs (they get 503 "server restarting") so a
    restart can wait for the running ones without a new one slipping in."""
    _require_local_supervisor(request)
    _rt.set_draining(True)
    return _rt.drain_state()


@app.delete("/health/drain")
async def health_drain_cancel(request: Request):
    """Admit spec jobs again (the supervisor gave up waiting)."""
    _require_local_supervisor(request)
    _rt.set_draining(False)
    return _rt.drain_state()


@app.get("/account/profile.html")
async def account_profile_page():
    """React SPA; routes to `web/src/pages/AccountProfile.tsx`."""
    return spa_page_response()


@app.get("/account/workspace.html")
async def account_workspace_page():
    """React SPA; routes to `web/src/pages/AccountWorkspace.tsx`."""
    return spa_page_response()


@app.get("/account/bench-review.html")
async def account_bench_review_page():
    """Standalone batch-review popup; routes to `web/src/pages/BenchReview.tsx`."""
    return spa_page_response()


@app.get("/account/my-benchmarks.html")
async def account_benchmarks_page():
    """Benchmark management lives in the workspace page's #benchmarks
    panel, so this is a permanent redirect there, which keeps bookmarks
    and stale login `?next=` targets off a bare JSON 404."""
    return RedirectResponse("/account/workspace.html#benchmarks", status_code=301)


@app.get("/admin")
@app.get("/admin/")
@app.get("/admin/index.html")
async def admin_page():
    """React SPA; routes to `web/src/pages/Admin.tsx`."""
    return spa_page_response()


@app.get("/auth/login.html")
async def auth_login_page():
    """React SPA; routes to `web/src/pages/AuthLogin.tsx`."""
    return spa_page_response()


@app.get("/auth/verify.html")
async def auth_verify_page():
    """React SPA; routes to `web/src/pages/AuthVerify.tsx`."""
    return spa_page_response()


@app.get("/auth/reset-password.html")
async def auth_reset_page():
    """React SPA; routes to `web/src/pages/AuthReset.tsx`."""
    return spa_page_response()


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return public_file_response("favicon.ico")


@app.get("/robots.txt", include_in_schema=False)
async def robots_txt():
    return public_file_response("robots.txt")


@app.get("/privacy")
async def privacy_page():
    """Privacy policy at a clean /privacy URL, distinct from `/`.
    Required by Google OAuth's consent screen. React SPA; routes to
    `web/src/pages/Privacy.tsx`.
    """
    return spa_page_response()


@app.get("/dim-glossary")
async def dim_glossary():
    """Symbol-to-meaning map for dim bindings, read from the registry in
    `backend/arch/dims.yaml` — where each symbol is classified — so a
    tooltip and the shard policy cannot disagree. Cached at process load.
    """
    return {"items": dims.meanings()}


@app.get("/api/agents")
async def agent_backends():
    """The coding-agent backends this server registers (`backend.agents`),
    in registry order, each with the name pages show it by, and the one a
    spec run uses when nothing chooses another (`SPEC_AGENT_CLI`)."""
    return {
        "agents": [{"cli": cli, "label": get_backend(cli).display_name}
                   for cli in available_clis()],
        "default": _rt.get_agent_cli(),
    }


@app.get(
    "/debug/spec-locks",
    dependencies=[Depends(require_admin_if_configured)],
)
async def debug_spec_locks():
    """Snapshot of the in-process spec-worker concurrency state.
    Useful when /generate-spec keeps returning 'overloaded' despite a
    fresh restart — `semaphore_value` should equal `max_concurrent`
    when no agent runs are active. A drift means a leak; a stuck row
    in `_model_locks` shows which model_id leaked.
    """
    return {
        "max_concurrent": _rt._max_concurrent,
        "semaphore_value": _rt.semaphore._value,
        "model_locks": list(_rt._model_locks.keys()),
        "model_procs": list(_rt._model_procs.keys()),
        "supabase_lock_enabled": _rt._supabase_lock_enabled(),
        "host_id": _rt._host_id(),
        "boot_id": _rt.boot_id(),
    }


@app.get(
    "/debug",
    dependencies=[Depends(require_admin_if_configured)],
)
async def debug():
    """Diagnostic endpoint — checks import status."""
    try:
        from backend.arch.loader import SpecLoader

        loader = SpecLoader()
        specs = list(loader.all_specs().keys())

        return {
            "status": "ok",
            "imports": {
                "SpecLoader": SpecLoader is not None,
            },
            "arch_specs": specs,
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
