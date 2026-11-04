"""CLI entry: `python -m backend.api`.

Parses --host / --port / --reload + spec_worker runtime knobs, then
hands off to uvicorn. Equivalent to `uvicorn backend.api:app --host
... --port ...`.
"""
from __future__ import annotations

import argparse

import uvicorn
from dotenv import load_dotenv

# Load .env from repo root before any submodule reads os.environ.
# Existing real env vars take precedence (override=False), so production
# deployments that inject vars via systemd / docker / k8s aren't shadowed.
load_dotenv(override=False)

from backend import spec_worker

from . import app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HuggingArch API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload", action="store_true",
        help="Auto-reload on source changes (dev only; requires import-string target)",
    )
    parser.add_argument("--spec-agent-cli", choices=spec_worker.VALID_AGENT_CLIS)
    parser.add_argument("--spec-agent-max-concurrent", type=int)
    parser.add_argument(
        "--ssl-certfile",
        help="Path to SSL certificate (PEM). Set together with --ssl-keyfile to serve HTTPS.",
    )
    parser.add_argument(
        "--ssl-keyfile",
        help="Path to SSL private key (PEM). Set together with --ssl-certfile to serve HTTPS.",
    )
    parser.add_argument(
        "--timeout-graceful-shutdown", type=int, default=15,
        help="Max seconds uvicorn waits for in-flight requests/background tasks on "
             "SIGINT/SIGTERM before cancelling them and exiting. Keeps `ha restart` "
             "from wedging forever on a stuck SSE/bench task (default: 15).",
    )
    args = parser.parse_args()

    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        parser.error("--ssl-certfile and --ssl-keyfile must be provided together")

    spec_worker.configure_runtime(
        agent_cli=args.spec_agent_cli,
        max_concurrent=args.spec_agent_max_concurrent,
    )

    # uvicorn takes the cert/key as separate paths (not Flask's
    # ssl_context tuple). When unset these stay None and uvicorn serves
    # plain HTTP, so the default dev workflow is unchanged.
    ssl_kwargs = {"ssl_certfile": args.ssl_certfile, "ssl_keyfile": args.ssl_keyfile}

    # uvicorn's --reload needs an import-string target, not the imported
    # `app` object. When --reload is set we pass "backend.api:app"
    # so the auto-reloader can re-import the module on file changes.
    run_kwargs = {
        "host": args.host,
        "port": args.port,
        "timeout_graceful_shutdown": args.timeout_graceful_shutdown,
        **ssl_kwargs,
    }

    if args.reload:
        uvicorn.run("backend.api:app", reload=True, **run_kwargs)
    else:
        uvicorn.run(app, **run_kwargs)


if __name__ == "__main__":
    main()
