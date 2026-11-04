"""Shared generated-spec persistence and the loader's miss hook.

The canonical shared tier is Supabase when it is configured. An auth-off
self-host deployment has no database, yet a successful anonymous generation
must still survive process restarts; those bundles land in an atomic,
host-local JSON cache. Signed-in users never read this global tier: their
artifacts stay in ``spec_generations`` until a human opens a PR.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend import supabase_env, supabase_rest
from backend.spec_worker.bundle import (
    FormalSource,
    SpecBundleError,
    normalize_bundle,
)
from backend.spec_worker.overlay import _safe


logger = logging.getLogger(__name__)
_commit_guard = threading.Lock()
_commit_locks: dict[str, threading.Lock] = {}


def _rpc(fn: str, body: dict) -> Optional[list]:
    """One cached-spec RPC, or None on any transport/HTTP/decode failure."""
    return supabase_rest.rpc(fn, body, label="shared_spec_cache", logger=logger)


def shared_spec_cache_root() -> Path:
    value = os.environ.get("HA_SHARED_SPEC_CACHE_ROOT")
    return Path(value) if value else Path.home() / ".huggingarch" / "shared_spec_cache"


def _local_leaf(model_id: str) -> Path:
    root = shared_spec_cache_root().resolve()
    leaf = (root / _safe(model_id)).resolve()
    if not leaf.is_relative_to(root):
        raise ValueError(f"shared spec cache path for {model_id!r} escapes its root")
    return leaf


def _commit_lock(model_id: str) -> threading.Lock:
    key = _safe(model_id)
    with _commit_guard:
        if key not in _commit_locks:
            _commit_locks[key] = threading.Lock()
        return _commit_locks[key]


def formal_model_exists(model_id: str) -> bool:
    arch = Path(__file__).resolve().parents[1] / "arch"
    return (arch / "models" / f"{_safe(model_id)}.yaml").is_file()


def upsert_after_success(
    model_id: str,
    spec_yaml: str,
    *,
    snapshot_json: Optional[dict] = None,
    new_component_yamls: Optional[list[dict]] = None,
    agent_cli: Optional[str] = None,
    generated_by: Optional[str] = None,
) -> bool:
    if not supabase_env.configured():
        return False
    result = _rpc("upsert_cached_spec", {
        "p_model_id": model_id,
        "p_spec_yaml": spec_yaml,
        "p_snapshot_json": snapshot_json or {},
        "p_new_component_yamls": new_component_yamls or [],
        "p_agent_cli": agent_cli,
        "p_generated_by": generated_by,
    })
    return result is not None


def write_local_after_success(
    model_id: str,
    spec_yaml: str,
    *,
    snapshot_json: Optional[dict] = None,
    new_component_yamls: Optional[list[dict]] = None,
    agent_cli: Optional[str] = None,
) -> None:
    """Atomically replace this model's host-local shared-cache entry."""
    with _commit_lock(model_id):
        bundle = normalize_bundle(
            model_id,
            spec_yaml,
            new_component_yamls or [],
            FormalSource.from_deployment(),
        )
        payload = {
            "model_id": model_id,
            "spec_yaml": bundle.spec_yaml,
            "snapshot_json": snapshot_json or {},
            "new_component_yamls": bundle.record_files(),
            "agent_cli": agent_cli,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "storage": "selfhost-local",
        }
        leaf = _local_leaf(model_id)
        leaf.mkdir(parents=True, exist_ok=True, mode=0o700)
        leaf.chmod(0o700)
        fd, temp_name = tempfile.mkstemp(
            prefix=".bundle-", suffix=".tmp", dir=leaf)
        temp = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            temp.chmod(0o600)
            os.replace(temp, leaf / "bundle.json")
        finally:
            temp.unlink(missing_ok=True)


def fetch(model_id: str) -> Optional[dict]:
    if not supabase_env.configured():
        return None
    rows = _rpc("get_cached_spec", {"p_model_id": model_id})
    return rows[0] if rows else None


def _read_local(model_id: str) -> Optional[dict]:
    path = _local_leaf(model_id) / "bundle.json"
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(row, dict) or not (row.get("spec_yaml") or "").strip():
            raise ValueError("bundle has no spec_yaml")
        return row
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(f"local shared spec cache miss for {model_id}: {exc}")
        return None


def _expire_local_if_formal(model_id: str) -> None:
    if not formal_model_exists(model_id):
        return
    shutil.rmtree(_local_leaf(model_id), ignore_errors=True)


def bump_hit(model_id: str) -> None:
    if not supabase_env.configured():
        return
    _rpc("bump_cached_spec_hit", {"p_model_id": model_id})


def materialize_cached_spec(model_id: str) -> Optional[str]:
    """Return a serving overlay for a DB or local shared-cache hit.

    Lookup order is fixed: multi-user refuses; a formal deployment model
    expires the cache; Supabase is consulted first; then the host-local
    fallback. Every failure is a miss, not a serving error.
    """
    from backend.spec_worker.overlay import materialize_user_spec

    if supabase_env.jwt_verification_available():
        return None
    if formal_model_exists(model_id):
        _expire_local_if_formal(model_id)
        return None
    try:
        row = fetch(model_id)
        db_hit = row is not None
        if row is None:
            row = _read_local(model_id)
        if row is None:
            return None
        bundle = normalize_bundle(
            model_id,
            row["spec_yaml"],
            row.get("new_component_yamls") or [],
            FormalSource.from_deployment(),
        )
        root = materialize_user_spec("_cached_specs", model_id, bundle,
                                     row.get("snapshot_json"))
        if db_hit:
            bump_hit(model_id)
        return root
    except Exception as exc:
        if isinstance(exc, SpecBundleError):
            detail = "; ".join(exc.errors)
            logger.warning(
                f"cached spec for {model_id} not materialized: {detail}")
        else:
            logger.warning(f"cached spec for {model_id} not materialized: {exc}")
        return None


__all__ = [
    "bump_hit",
    "fetch",
    "materialize_cached_spec",
    "shared_spec_cache_root",
    "upsert_after_success",
    "write_local_after_success",
]
