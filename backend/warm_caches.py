"""Entry point of the out-of-process cache warm: `python -m backend.warm_caches`.

Loads `.env` (real env vars win), then runs `backend.api.warm.main`. It lives
outside `backend.api` on purpose: importing that package binds the cache
directories (`HA_GALLERY_CACHE_DIR`, `HA_SWEEP_CACHE_DIR`) and the remote
cache timeouts, so `.env` must be loaded before the first `backend.api`
import, the same order `python -m backend.api` uses for the server. The
`.env` is the one in the working directory or its nearest parent; `ha` runs
from the repo root.

Arguments: see `python -m backend.warm_caches --help`.
"""
from __future__ import annotations

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)

from backend.api.warm import main  # after .env, by design

if __name__ == "__main__":
    raise SystemExit(main())
