"""On-disk JSON cache for fetch_metadata results.

Why: full metadata fetch for 200+-shard MoE checkpoints (e.g. GLM-5.1, 282
shards) takes 10–30s per call. The spec_worker pipeline fetches once when
building the prompt, then the subagent's `validate` subprocess fetches
again moments later for the same model_id. Without caching, subagents have
timed out and forced `HF_HUB_OFFLINE=1`, which silently zeroes all HF
reads (see logs/spec_worker/zai-org__GLM-5.1_*.log).

Keyed by model_id. TTL bounds staleness. Opt-out: HUGGINGARCH_HF_META_CACHE=0.

This module exposes raw read/write of a JSON dict; callers (metadata.py)
re-run the classifier on the loaded data to refresh totals against the
current quant scheme registry.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = "~/.cache/huggingarch/hf_meta"
_CACHE_TTL_SEC = 7 * 24 * 3600

# Bump whenever the SHAPE of the cached record changes. Entries without this
# field, or with a mismatched value, are a miss and get refetched.
#   v4: the record is the verbatim safetensors header (`metadata_record`);
#       earlier versions stored names folded on write.
_CACHE_SCHEMA_VERSION = 4


def _cache_dir() -> str:
    return os.path.expanduser(
        os.environ.get("HUGGINGARCH_HF_META_CACHE_DIR") or _DEFAULT_CACHE_DIR
    )


def _cache_path(model_id: str) -> str:
    return os.path.join(_cache_dir(), f"{model_id.replace('/', '__')}.json")


def cache_load_raw(model_id: str) -> Optional[Dict[str, Any]]:
    """Read the cached JSON dict for `model_id`, or None on miss / stale /
    schema-version mismatch / cache disabled / corrupt. The returned dict
    is RAW — caller must re-classify totals against the current quant
    scheme registry."""
    if os.environ.get("HUGGINGARCH_HF_META_CACHE") == "0":
        return None
    path = _cache_path(model_id)
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    if time.time() - st.st_mtime > _CACHE_TTL_SEC:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("header"):
            return None  # don't serve cached misses
        if data.get("_cache_schema_version") != _CACHE_SCHEMA_VERSION:
            # Built under an older normalization — refetch.
            logger.debug(f"[{model_id}] hf_metadata: cache schema stale")
            return None
        logger.debug(f"[{model_id}] hf_metadata: cache hit")
        return data
    except (OSError, json.JSONDecodeError) as e:
        logger.debug(f"[{model_id}] hf_metadata: cache read failed ({e})")
        return None


def cache_store(model_id: str, data: Dict[str, Any]) -> None:
    """Atomically persist `data` for `model_id` (write tmp + rename).
    Stamps the current cache schema version so a later normalization
    change auto-invalidates this entry on read."""
    if os.environ.get("HUGGINGARCH_HF_META_CACHE") == "0":
        return
    if not data.get("header"):
        return
    try:
        os.makedirs(_cache_dir(), exist_ok=True)
        path = _cache_path(model_id)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({**data, "_cache_schema_version": _CACHE_SCHEMA_VERSION}, f)
        os.replace(tmp, path)
    except OSError as e:
        logger.debug(f"[{model_id}] hf_metadata: cache write failed ({e})")
