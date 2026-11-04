"""The Kernel Bench service: `python -m backend.service.bench.server`
(docs/measured_calibration.md §5.1).

A FastAPI app bound to 127.0.0.1 (port `HUGGINGARCH_BENCH_PORT`, default
8100), and the only process that opens the Kernel Bench lake: it runs the
serving gate (restore, lease, annotations, backups, maintenance — `serving`)
and every Kernel Bench endpoint (`routes`): the pages' `/api/bench/*`, which
the HA API forwards, and `/internal/bench/lookup` / `/internal/bench/version`,
which estimates read.

One process, always: the lake is local to one process (DuckDB locks its
catalog file), so this module runs uvicorn on the app object — no workers,
no reload — whatever `WEB_CONCURRENCY` the HA API next to it runs with.

Every request must carry the shared secret (`measurements.secret()`) in
`X-HA-Internal`, from a loopback peer (`InternalOnly`); anything else is 403
before it reaches a route, `/health` included. Identity is only ever taken
from `X-HA-User` / `X-HA-Admin` on such a request (`identity`). A service
started without a secret refuses to start.
"""
from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

from fastapi import FastAPI

from backend.measure.lookup import INTERNAL_HEADER, secret
from backend.service.bench import lake as _lake_mod
from backend.service.bench import routes

_LOG = logging.getLogger(__name__)

DEFAULT_PORT = 8100


def _loopback(host: Optional[str]) -> bool:
    try:
        addr = ipaddress.ip_address(host or "")
    except ValueError:
        _LOG.warning("Kernel Bench service: refused a peer that is not an IP address "
                     "(%r): not loopback", host)
        return False
    if addr.version == 6 and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr.is_loopback


class InternalOnly:
    """ASGI middleware: a request without the shared secret in
    `X-HA-Internal` (compared in constant time), or from a peer that is not
    loopback, is answered 403 and goes no further. `expected` is read per
    request so a test may set it; the secret is never logged."""

    def __init__(self, app: Any, expected: Callable[[], str] = secret):
        self.app = app
        self.expected = expected

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        want = self.expected()
        got = ""
        name = INTERNAL_HEADER.lower().encode()
        for k, v in scope.get("headers") or ():
            if k == name:
                got = v.decode("latin-1")
                break
        client = scope.get("client")
        ok = (bool(want) and hmac.compare_digest(got.encode(), want.encode())
              and _loopback(client[0] if client else None))
        if not ok:
            body = json.dumps({"detail": "internal endpoint"}).encode()
            await send({"type": "http.response.start", "status": 403,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode())]})
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


def create_app(*, start_lake: bool = True) -> FastAPI:
    """The service's app. `start_lake`: run the gate's startup in the
    background once the server is up (tests leave it to the first request,
    `Gate.reader`)."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if start_lake:
            routes.start()
        yield

    app = FastAPI(title="Kernel Bench", lifespan=lifespan, docs_url=None,
                  redoc_url=None, openapi_url=None)
    app.include_router(routes.router)
    app.add_exception_handler(_lake_mod.LakeUnavailable, routes.lake_unavailable)
    app.add_exception_handler(_lake_mod.CommitConflict, routes.lake_unavailable)

    @app.get("/health")
    def health() -> dict:
        """The lake's state (`serving.Gate.health`): never reads the lake,
        never fails."""
        try:
            lake = routes.health()
        except Exception as e:      # /health itself must never fail
            lake = {"error": f"{type(e).__name__}: {e}"}
        return {"status": "ok", "lake": lake, "lookups": routes.lookup_stats.report()}

    app.add_middleware(InternalOnly)
    return app


def main(argv: Optional[list] = None) -> int:
    from dotenv import load_dotenv
    import uvicorn

    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description="Run the Kernel Bench service")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("HUGGINGARCH_BENCH_PORT") or DEFAULT_PORT))
    args = parser.parse_args(argv)
    if not secret():
        parser.error("no shared secret: set HUGGINGARCH_BENCH_SECRET or "
                     "HUGGINGARCH_BENCH_SECRET_FILE (the HA API sends it)")
    logging.basicConfig(level=logging.INFO)
    # One process on the app object: no workers, no reload.
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port,
                timeout_graceful_shutdown=15)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
