"""Compatibility exports for the core shared-spec cache.

The implementation moved to ``backend.spec_worker.shared_spec_miss`` so the
OSS miss hook and self-host local cache do not depend on the optional service
package. This module remains the historical import for service code/tests.
"""
from backend.spec_worker.shared_spec_miss import (
    _rpc,
    bump_hit,
    fetch,
    materialize_cached_spec,
    shared_spec_cache_root,
    upsert_after_success,
    write_local_after_success,
)

__all__ = [
    "_rpc",
    "bump_hit",
    "fetch",
    "materialize_cached_spec",
    "shared_spec_cache_root",
    "upsert_after_success",
    "write_local_after_success",
]
