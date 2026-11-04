"""GET /supported-models — autocomplete data for the model-id input
(curated built-ins + cached entries + locally-registered customs).
"""
from __future__ import annotations

import json as _json


from backend.arch import snapshots as _snapshots
from backend.arch.loader import SpecLoader
from backend.custom_id import NAME_PREFIX as _CUSTOM_PREFIX
from backend.hf.forward_source import model_info_cache_path

from .._app import app
from backend.measure.cache import KeyedCache, MemoryBackend
from backend.measure.scope import make_key
from .._helpers import load_supported_model_list
from ..offload import offload_response

# Process cache for /supported-models (shared cache machinery — `backend.measure.cache`).
# The endpoint runs loader.match() over every snapshot + model_info cache entry
# (~1 s for ~40 models) and the Inference, Inspector and Playground pages fetch
# it on load, so uncached it makes each of them wait ~1 s. The result changes only when the
# snapshot set, the model_info cache, the curated list, or the custom registry
# changes — a cheap signature over all four is the cache key, so a change lands
# on a fresh key (max_entries=1 keeps just the latest). Source-code changes are
# covered by process restart (`ha restart` re-imports and clears the cache).
_SUPPORTED = KeyedCache("supported", MemoryBackend(max_entries=1))


@app.get("/supported-models")
async def supported_models():
    """Return suggested model IDs with source annotation.

    - `built-in`: ID is explicitly listed in
      `backend/arch/supported_models.txt` (git-tracked, manually
      curated — a verified architecture + checkpoint combination).
    - `cached`:   the ID has on-disk data we can analyze without going
      to HF — either a repo-tracked spec snapshot under
      `backend/arch/snapshots/` (shipped with the repo, typically
      written by `/generate-spec` then committed) or a user-level
      model_info cache from a previous CLI extract. Either way the ID
      stays `cached` until promoted to the curated list above.
    - `custom`:   locally registered via `python -m backend.custom_model
      register` — surfaced with the `custom:` prefix.
    """
    return await offload_response(_supported_models_payload)


def _supported_models_payload() -> dict:
    curated = set(load_supported_model_list())

    # Locally-registered custom models — surface with `custom:` prefix so
    # the same selector drives both HF-hub and user-registered models.
    try:
        from backend.custom_model import list_all as _list_custom
        custom_ids = [f"{_CUSTOM_PREFIX}{name}" for name in _list_custom()]
    except (ImportError, OSError):
        custom_ids = []

    # Cheap signature over everything the match() loop below depends on. If it
    # is unchanged since the last build, reuse the cached payload and skip the
    # ~1 s of loader.match() calls entirely.
    snap_ids = tuple(sorted(_snapshots.iter_snapshot_ids()))
    cache_dir = model_info_cache_path("x").parent
    try:
        cache_sig = int(cache_dir.stat().st_mtime_ns) if cache_dir.exists() else None
    except OSError:
        cache_sig = None
    sig = (tuple(sorted(curated)), snap_ids, cache_sig, tuple(sorted(custom_ids)))

    def _build() -> dict:
        loader = SpecLoader()
        cached_only: set = set()

        def _maybe_add(mid: str, cfg: dict) -> None:
            # Only surface IDs whose architecture resolves to a git-managed
            # spec — otherwise the user clicks one, /analyze fails, confusing.
            if mid in curated or mid in cached_only:
                return
            try:
                if loader.match(cfg, model_id=mid) is not None:
                    cached_only.add(mid)
            except (ValueError, KeyError):
                return

        # Repo-tracked spec snapshots. These ship with the codebase, so anyone
        # who clones the repo gets the same autocomplete list — no per-user
        # CLI extract required.
        for mid in snap_ids:
            snap = _snapshots.load_snapshot(mid)
            if snap:
                _maybe_add(mid, snap.get("config") or {})

        # User-level model_info cache (populated by `python -m backend.hf.model_info`).
        # Distinct directory from snapshots above and not guaranteed to overlap.
        if cache_dir.exists():
            for entry in cache_dir.iterdir():
                if entry.suffix != ".json":
                    continue
                mid = entry.stem.replace("__", "/")
                try:
                    data = _json.loads(entry.read_text())
                except (OSError, _json.JSONDecodeError):
                    continue  # skip corrupt cache entries silently
                _maybe_add(mid, data.get("config") or {})

        items = [{"model_id": mid, "source": "custom"} for mid in sorted(custom_ids)]
        items.extend({"model_id": mid, "source": "built-in"} for mid in sorted(curated))
        items.extend({"model_id": mid, "source": "cached"} for mid in sorted(cached_only))
        return {"items": items, "model_ids": [it["model_id"] for it in items]}

    return _SUPPORTED.get_or_compute(make_key("supported", sig), _build)
