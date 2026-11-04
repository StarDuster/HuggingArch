"""Git-managed snapshots for built-in models.

One JSON file per curated HF model_id captures everything the analyzer
needs to run offline: the model's config.json + the checkpoint's safetensors
header, every tensor under the name the file gives it.

Schema:
    {
      "schema_version": 4,
      "model_id": "meta-llama/Llama-3.1-8B",
      "config":   { ... raw config.json content ... },
      "metadata": { ... `hf.metadata.metadata_record`: the safetensors
                    header verbatim (`hf.header.SafetensorsHeader`) ... },
      "forward_source": {                 # optional
        "source": "local_transformers" | "hf_modeling_files" | ...,
        "paths":  [...],
        "files":  [{"path": ..., "text": ...}, ...]
      },
      "model_str": "<str(model)> output", # optional; print(model) tree
      "processor": {                      # optional; the repo's processor
        "preprocessor_config.json": {...},  # configs verbatim, by file name
        "video_preprocessor_config.json": {...},
        "processor_config.json": {...}
      },
      "release_date": "YYYY-MM-DD"        # optional; public availability date,
                                          # normally the repo's initial public
                                          # commit. Consumed by the gallery
                                          # timeline chart. Lives here (not on
                                          # the arch spec) because one spec serves
                                          # many checkpoints of different
                                          # vintages, each with its own HF repo.
    }

    `release_date` normally comes from the HF repo's oldest visible commit
    (or, for inaccessible gated repos, the vendor's public launch date), not
    the offline fetch path, so `write_snapshot` PRESERVES any existing value
    when a snapshot is regenerated (regen doesn't re-query the Hub). The repo
    metadata `createdAt` is deliberately not used: providers can create a
    private placeholder months before making the checkpoint public.

Analyzer + hf_metadata check this dir BEFORE hitting HF, so built-in
models work without network / auth tokens (critical for gated repos
like Llama).

Refresh a snapshot with:  python -m backend.hf.snapshot <id>
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from backend.custom_id import is_custom_id

_log = logging.getLogger(__name__)

_SNAP_DIR = Path(__file__).parent


# --- Forward-source path normalization -------------------------------------
#
# `forward_source.paths` and `forward_source.files[].path` carry *display*
# paths to the original modeling_*.py the agent / UI should reference.
# Persisted as-is, these would be absolute paths into the user's local
# .venv or HF cache and would churn between machines / cache-locations.
# We strip the user-local prefix at write time so committed snapshots stay
# stable. Three families seen in practice:
#
#   1. .venv/.../site-packages/transformers/.../modeling_X.py
#        → "site-packages/transformers/.../modeling_X.py"
#   2. ~/.cache/huggingface/hub/models--<owner>--<name>/snapshots/<hash>/<file>
#        → "hf-cache/<owner>/<name>/<file>"   (drop the version hash too —
#                                              hashes change when the repo
#                                              re-uploads, and the agent
#                                              only needs to know "this
#                                              file came from <owner>/<name>")
#   3. anything else (custom_models, project-relative, …)
#        → strip $HOME → "~"; otherwise leave untouched.

_RE_SITE_PACKAGES = re.compile(r".*?/site-packages/")
_RE_HF_HUB_CACHE = re.compile(
    r".*/\.cache/huggingface/hub/models--([^/]+)--([^/]+)/snapshots/[^/]+/"
)


def _normalize_path(p: Optional[str]) -> Optional[str]:
    """Strip user-local / venv-specific prefixes from a display path."""
    if not p or not isinstance(p, str):
        return p
    m = _RE_SITE_PACKAGES.match(p)
    if m:
        return "site-packages/" + p[m.end():]
    m = _RE_HF_HUB_CACHE.match(p)
    if m:
        owner, name = m.group(1), m.group(2)
        return f"hf-cache/{owner}/{name}/{p[m.end():]}"
    home = os.path.expanduser("~")
    if home and p.startswith(home + os.sep):
        return "~" + p[len(home):]
    return p


def _normalize_forward_source(fs: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a copy of `forward_source` with display paths normalized."""
    if not fs:
        return fs
    out = dict(fs)
    paths = fs.get("paths")
    if isinstance(paths, list):
        out["paths"] = [_normalize_path(p) for p in paths]
    files = fs.get("files")
    if isinstance(files, list):
        out["files"] = [
            {**f, "path": _normalize_path(f.get("path"))}
            if isinstance(f, dict) else f
            for f in files
        ]
    return out


def _fname(model_id: str) -> str:
    return model_id.replace("/", "__") + ".json"


def default_overlay_dir(overlay_dir: Optional[str] = None) -> Optional[str]:
    """The caller's overlay root, or the one `HA_SPEC_OVERLAY_DIR` names.

    An overlay is one run's or one stored record's private tree (`models/`,
    `components/drafts/`, `snapshots/`). Every reader of that tree resolves
    it here, so a spec and the snapshot it was judged against are read from
    the same place whichever entry point asked: an in-process caller passes
    the root, a sandboxed agent's subprocess inherits the env."""
    return overlay_dir or os.environ.get("HA_SPEC_OVERLAY_DIR") or None


def snapshot_path(model_id: str, overlay_dir: Optional[str] = None) -> Path:
    """Where `model_id`'s snapshot lives: the shared dir, or the
    `snapshots/` of the overlay root `overlay_dir` names."""
    if overlay_dir:
        return Path(overlay_dir) / "snapshots" / _fname(model_id)
    return _SNAP_DIR / _fname(model_id)


def iter_snapshot_ids():
    """Yield the canonical `owner/model` ID for every snapshot file in
    the directory. Mirror inverse of `_fname` — the autocomplete endpoint
    uses this to surface every repo-shipped model without having to know
    the `<owner>__<model>.json` filename convention itself."""
    if not _SNAP_DIR.is_dir():
        return
    for entry in _SNAP_DIR.iterdir():
        if entry.suffix != ".json" or not entry.is_file():
            continue
        yield entry.stem.replace("__", "/")


def load_snapshot(model_id: str, overlay_dir: Optional[str] = None
                  ) -> Optional[Dict[str, Any]]:
    """Return the full snapshot dict or None when no snapshot exists.

    Order: the shared snapshots dir, then the overlay's snapshots/
    (`default_overlay_dir`) — the spec worker stages the prefetched
    snapshot there so the sandboxed agent's validate loop runs offline,
    and serving a stored record stages the snapshot it was judged against.
    The shared dir wins: an overlay is agent-writable and must not shadow a
    curated snapshot.
    """
    paths = [snapshot_path(model_id)]
    overlay = default_overlay_dir(overlay_dir)
    if overlay:
        paths.append(snapshot_path(model_id, overlay))
    for p in paths:
        if not p.exists():
            continue
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            # The file exists but is unreadable/corrupt — NOT the same as
            # "no snapshot" (a missing file, handled by the exists() guard
            # above). Returning None here would let the caller silently fall
            # back as if the model were simply un-prefetched, hiding the
            # corruption. Surface it. (tests/meta/test_snapshot_json_parses
            # gates the git-managed store, so on shipped data this can only
            # fire for a corrupt overlay/local file.)
            raise ValueError(f"snapshot {p} exists but is unreadable: {e}") from e
    return None


def load_config(model_id: str, overlay_dir: Optional[str] = None
                ) -> Optional[Dict[str, Any]]:
    snap = load_snapshot(model_id, overlay_dir)
    return snap.get("config") if snap else None


def load_metadata(model_id: str, overlay_dir: Optional[str] = None
                  ) -> Optional[Dict[str, Any]]:
    """The metadata record stored in the snapshot file, or None.

    This is the on-disk record (the header verbatim); its totals are
    `hf.metadata.fetch_metadata`'s to derive.
    """
    snap = load_snapshot(model_id, overlay_dir)
    if not snap:
        return None
    return snap.get("metadata")


def load_forward_source(model_id: str, overlay_dir: Optional[str] = None
                        ) -> Optional[Dict[str, Any]]:
    snap = load_snapshot(model_id, overlay_dir)
    fs = snap.get("forward_source") if snap else None
    if fs and fs.get("files"):
        return fs
    return None


def load_processor(model_id: str, overlay_dir: Optional[str] = None
                   ) -> Optional[Dict[str, Any]]:
    """The processor configs the repo ships beside config.json — how a raw
    image, video or clip becomes what the model's forward reads — as the
    snapshot holds them, verbatim by file name; None when it holds none
    (the repo ships none, or the snapshot predates them)."""
    snap = load_snapshot(model_id, overlay_dir)
    return (snap.get("processor") if snap else None) or None


def load_model_str(model_id: str, overlay_dir: Optional[str] = None
                   ) -> Optional[str]:
    snap = load_snapshot(model_id, overlay_dir)
    return (snap.get("model_str") if snap else None) or None


def load_release_date(model_id: str) -> Optional[str]:
    """Return the per-checkpoint release date ("YYYY-MM-DD", from the HF repo's
    initial public commit), or None when the snapshot has no date recorded."""
    snap = load_snapshot(model_id)
    return (snap.get("release_date") if snap else None) or None


def _release_date_cache_path(model_id: str) -> Path:
    safe = re.sub(r"[^\w.\-]+", "__", model_id)
    root = Path(os.path.expanduser(
        os.environ.get("HUGGINGARCH_HF_RELEASE_DATE_DIR")
        or os.environ.get("HUGGINGARCH_HF_CREATED_AT_DIR")
        or "~/.cache/huggingarch/hf_release_dates"))
    return root / f"{safe}.json"


#: How long a Hub lookup that found no date (missing / private repo, or the
#: Hub unreachable) is remembered before it is asked again.
_RELEASE_DATE_NEGATIVE_TTL_S = 24 * 3600
_RELEASE_DATE_TIMEOUT_S = 5.0
_RELEASE_DATE_CACHE_SOURCE = "hf_initial_commit"


def _read_release_date_cache(model_id: str) -> Tuple[bool, Optional[str]]:
    """(cached, date). A cached miss is `(True, None)` until its TTL lapses."""
    p = _release_date_cache_path(model_id)
    if not p.is_file():
        return False, None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _log.warning("unreadable release-date cache %s: %s", p, exc)
        return False, None
    if not isinstance(data, dict) or data.get("source") != _RELEASE_DATE_CACHE_SOURCE:
        # v1 cached ModelInfo.createdAt, which can predate the public release.
        return False, None
    date = data.get("release_date") if isinstance(data, dict) else None
    checked_at = data.get("checked_at") if isinstance(data, dict) else None
    if isinstance(date, str) and len(date) >= 8:
        return True, date
    if (date is not None
            or isinstance(checked_at, bool)
            or not isinstance(checked_at, (int, float))):
        _log.warning("malformed release-date cache %s: %r", p, data)
        return False, None
    return time.time() - checked_at < _RELEASE_DATE_NEGATIVE_TTL_S, None


def _write_release_date_cache(model_id: str, date: Optional[str]) -> None:
    p = _release_date_cache_path(model_id)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "release_date": date,
            "checked_at": time.time(),
            "source": _RELEASE_DATE_CACHE_SOURCE,
        }))
    except OSError as exc:
        _log.warning("cannot write release-date cache %s: %s", p, exc)


def _iter_hf_commit_pages(model_id: str):
    """Yield JSON commit pages from the Hub, oldest commit on the last page."""
    from urllib.parse import quote

    from huggingface_hub import HfApi
    from huggingface_hub.utils import (
        build_hf_headers,
        get_session,
        hf_raise_for_status,
    )

    api = HfApi()
    encoded = quote(model_id, safe="/")
    url = f"{api.endpoint}/api/models/{encoded}/commits/main"
    headers = build_hf_headers()
    session = get_session()
    while url:
        response = session.get(
            url,
            headers=headers,
            timeout=_RELEASE_DATE_TIMEOUT_S,
        )
        hf_raise_for_status(response)
        page = response.json()
        if not isinstance(page, list):
            raise ValueError(f"Hub commit response for {model_id} is not a list")
        yield page
        url = response.links.get("next", {}).get("url")


def _fetch_hf_release_date(model_id: str) -> Optional[str]:
    """Date of the repo's oldest visible commit as YYYY-MM-DD.

    This is the closest Hub API proxy for "made public": `ModelInfo.createdAt`
    records private repo creation (Kimi-K3 was created 2026-06-13 but first
    appeared publicly with a 2026-07-27 initial commit), while `lastModified`
    keeps moving whenever the README or eval results change.
    """
    try:
        pages = list(_iter_hf_commit_pages(model_id))
    except Exception as exc:
        _log.info("Hub release-date lookup failed for %s: %s", model_id, exc)
        return None
    commits = [commit for page in pages for commit in page]
    if not commits:
        return None
    # HfApi.list_repo_commits parses this same field; do so here so every page
    # request carries the timeout above (that public wrapper has no timeout arg).
    raw = commits[-1].get("date")
    if not isinstance(raw, str):
        return None
    from huggingface_hub.utils import parse_datetime
    return parse_datetime(raw).date().isoformat()


def release_dates_for(model_ids) -> Dict[str, Optional[str]]:
    """YYYY-MM-DD per model_id: snapshot first, then a cached initial commit.

    A Hub lookup happens at most once per TTL per model: a miss is cached
    too, and custom ids (no Hub repo) and offline mode never ask."""
    out: Dict[str, Optional[str]] = {}
    ask = []
    for mid in model_ids:
        if not mid or mid in out:
            continue
        dated = load_release_date(mid)
        if dated is None:
            cached, dated = _read_release_date_cache(mid)
            if not cached and not is_custom_id(mid):
                ask.append(mid)
        out[mid] = dated
    if os.environ.get("HF_HUB_OFFLINE"):
        return out
    for mid in ask:
        dated = _fetch_hf_release_date(mid)
        _write_release_date_cache(mid, dated)
        out[mid] = dated
    return out


def has_forward_source(forward_source: Optional[Dict[str, Any]]) -> bool:
    """Whether a `forward_source` record carries any source text."""
    return any(isinstance(f, dict) and f.get("text")
               for f in (forward_source or {}).get("files") or [])


def write_snapshot(
    model_id: str,
    config: Dict[str, Any],
    metadata: Dict[str, Any],
    forward_source: Optional[Dict[str, Any]] = None,
    model_str: Optional[str] = None,
    release_date: Optional[str] = None,
    overlay_dir: Optional[str] = None,
    processor: Optional[Dict[str, Any]] = None,
) -> Path:
    """Assemble a snapshot from its parts and `store_snapshot` it.

    A snapshot freezes the forward source with the config and the header:
    the source is the only witness for what the checkpoint bytes cannot say
    (block topology, cache formats, recurrent-state dtypes), and every check
    that reads it passes a model it cannot read. One without source text is
    refused, not written."""
    if not has_forward_source(forward_source):
        raise ValueError(
            f"snapshot {model_id}: no forward source to freeze. A snapshot "
            f"is written with the modeling code it is judged against — fetch "
            f"it (the repo's own files, or the installed transformers module "
            f"for its model_type) before snapshotting.")
    payload = {
        "schema_version": 4,
        "model_id": model_id,
        "config": config,
        "metadata": metadata,
        "forward_source": _normalize_forward_source(forward_source),
    }
    if model_str:
        payload["model_str"] = model_str
    if processor:
        payload["processor"] = dict(processor)
    # release_date comes from the HF initial public commit (backfilled
    # separately, not via the offline fetch path) — carry forward the value so
    # regenerating a snapshot doesn't drop it.
    if release_date is None:
        release_date = load_release_date(model_id)
    if release_date:
        payload["release_date"] = release_date
    return store_snapshot(model_id, payload, overlay_dir)


def store_snapshot(model_id: str, payload: Dict[str, Any],
                   overlay_dir: Optional[str] = None) -> Path:
    """Write a whole snapshot (one `write_snapshot` built, or the one a
    stored record carries) where `load_snapshot` reads it: the overlay root
    `overlay_dir` keeps a run or a record out of the shared tree; default is
    the shared snapshots dir. Atomic, so a concurrent reader never sees half
    a file."""
    p = snapshot_path(model_id, overlay_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(tmp, p)
    return p
