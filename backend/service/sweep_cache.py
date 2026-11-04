"""Durable cross-VM sweep cache backend (Supabase).

Wraps the three RPCs from supabase/migrations/0032_sweep_cache.sql:
    sweep_cache_get     — read-through (bumps hit_count/last_used_at)
    sweep_cache_upsert  — write-through + bounded prune
    sweep_cache_clear   — tests / manual ops

`SupabaseBackend` conforms to the `backend.measure.cache` backend duck type
(`get(key)` / `put(key, value)` / `clear()`), so it slots into a `KeyedCache` as
the gated `remote` tier behind memory + disk. Kept in the service layer so
`backend.measure.cache` stays DB-agnostic; `backend/service/__init__.py`
binds it.

The `sweep_cache` table is shared by two KeyedCache namespaces (keys are
sha256 material over disjoint part lists, so they never collide):
  * `_SWEEP` (inference routes) — the Scaling sweeps' theory halves and
    their SOL answers (`api/sweep_runtime.py`). Neither reads a measurement
    or a user's rows (a sweep that reads measurements is answered live and
    never stored), so every entry is shareable.
  * `_GALLERY` and `_FRONTIER` (api/gallery_overview.py) — per-model
    gallery facts and decode-frontier entries. Always shareable (no user
    input), but error entries are kept local via `remote_if`.

Silent-fail (return None / no-op) when SUPABASE is unconfigured or any network /
HTTP error hits — a cache is an optimization; a miss or an error must never break
the endpoint (same contract as DiskBackend, which swallows all IO errors).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from backend import supabase_rest


logger = logging.getLogger(__name__)


def _rpc(fn: str, body: dict | bytes, *, timeout: float = 5.0) -> Optional[list]:
    """A `sweep_cache_*` RPC: parsed JSON, or None on any failure — a miss.
    Mirrors shared_spec_cache._rpc."""
    return supabase_rest.rpc(fn, body, label="sweep_cache", logger=logger,
                             timeout=timeout)


def _encoded_within(obj: Any, limit: int) -> Optional[bytes]:
    """`obj` as JSON bytes, or None as soon as the encoding passes `limit` —
    an over-size payload is not encoded whole only to be dropped. ASCII
    output (`ensure_ascii`), so characters are bytes."""
    chunks, size = [], 0
    for chunk in json.JSONEncoder().iterencode(obj):
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return "".join(chunks).encode()


def _env_number(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


class SupabaseBackend:
    """Durable cold tier over the `sweep_cache` table. All methods silent-fail:
    an unconfigured or unreachable Supabase degrades the cache to whatever tiers
    sit in front of it (memory + disk), never an endpoint error.

    The read sits on a request's miss path, so its timeout is short
    (`HA_REMOTE_CACHE_READ_TIMEOUT_S`, default 2 s): an unreachable store
    costs a request at most that long before it computes. The write runs off
    the request (the service layer wraps this backend in a
    `WriteBehindBackend`), so it can afford a longer timeout for a large
    payload (`HA_REMOTE_CACHE_WRITE_TIMEOUT_S`, default 15 s).

    The store also has a size it can take: one jsonb value, written by one
    statement under Supabase's statement timeout. A gallery frontier entry is
    up to ~100 MB and a frontier walk over 1 GB; sent whole, every such write
    timed out after serializing it. A payload over `max_payload_bytes`
    (`HA_REMOTE_CACHE_MAX_BYTES`, default 8 MiB) stays on the local tiers —
    the namespaces that produce it need not know the store's limit."""

    def __init__(self, read_timeout_s: Optional[float] = None,
                 write_timeout_s: Optional[float] = None,
                 max_payload_bytes: Optional[int] = None):
        self._read_timeout = (
            read_timeout_s if read_timeout_s is not None
            else _env_number("HA_REMOTE_CACHE_READ_TIMEOUT_S", 2.0))
        self._write_timeout = (
            write_timeout_s if write_timeout_s is not None
            else _env_number("HA_REMOTE_CACHE_WRITE_TIMEOUT_S", 15.0))
        self._max_payload = (
            max_payload_bytes if max_payload_bytes is not None
            else int(_env_number("HA_REMOTE_CACHE_MAX_BYTES", 8 << 20)))

    def get(self, key: str) -> Optional[Any]:
        rows = _rpc("sweep_cache_get", {"p_key": key}, timeout=self._read_timeout)
        if not rows:
            return None
        # RETURNS TABLE(payload jsonb) → [{"payload": {...}}]; a stored NULL
        # payload is impossible (column is NOT NULL), so a present row is a hit.
        return rows[0].get("payload")

    def put(self, key: str, value: Any) -> None:
        body = _encoded_within({"p_key": key, "p_payload": value}, self._max_payload)
        if body is None:
            logger.info("sweep_cache: %s kept local (over %d bytes)",
                        key[:16], self._max_payload)
            return
        _rpc("sweep_cache_upsert", body, timeout=self._write_timeout)

    def clear(self) -> None:
        _rpc("sweep_cache_clear", {})
