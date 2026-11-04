"""Playground: compose/compare model components on top of the arch spec.

Phase 1 (semantics B): isolated module comparison — bench one library
component's canonical flops / weight-bytes / activation-bytes / kvcache under a
fixed input contract, reusing the existing single-component eval path
(`isolate_resolved` / `project_block` / `kv_cache_topology`). No new cost
formulas, no serving pipeline, no persistence.
"""

from typing import Optional


def entry_key(name: str, model_id: Optional[str],
              call: Optional[str] = None) -> str:
    """Identity of one compare entry: the bare component name when unbound,
    `name@model_id` when bound — so the same component bound to two different
    models is two distinguishable columns — and `#call` after it when the
    entry names its call, so two calls of one model are two columns too.
    The request's duplicate check and the bench's row keys both read it."""
    key = f"{name}@{model_id}" if model_id else name
    return f"{key}#{call}" if call else key
