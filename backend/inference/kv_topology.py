"""KV cache topology — a view of `eval.cache` for one component.

The geometry lives on the eval cache layer. This module names that
geometry the way older inference callers asked for it (`CachedKVNode`,
per-token element totals). It does not walk the component graph again.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.arch.eval.cache import cache_of_component
from backend.arch.eval.transforms import CallSite, analyze_isolated
from backend.arch.eval.dim import value_of


@dataclass
class CachedKVNode:
    """One KV-cache contribution from an attention component.

    `name` is the graph-node name (as listed in the component's
    `kv_cached:`); `elements_per_token` is the product of that node's
    resolved `shape_out` evaluated at B=s_ctx=1 — i.e. the number of
    scalar values the node emits per token per layer.

    `inner_dim` is the trailing dim of shape_out — the per-channel
    scale unit. For typical cached primitives the shape is either
    (B, s_ctx, dim) or (B, heads, s_ctx, dim), so the last dim doubles as
    "channel". `heads = elements_per_token // inner_dim` (= 1 for
    MLA-style flat latents).
    """
    name: str
    elements_per_token: int
    inner_dim: int
    # Per-node KV dtype name (the `dtype:` its `cache.entries` item resolves to);
    # None means "use the deployment-wide kv_dtype". Lets one component
    # cache different nodes at different precisions — DSA's FP8 indexer
    # key alongside the BF16 MLA latent.
    dtype: Optional[str] = None
    # Per-node KV storage / read cardinality (SymPy Expr in `s_kv`), for the
    # case where a single layer caches multiple nodes with DIFFERENT
    # cardinalities — DSv4 CSA/HCA cache an SWA `kv_concat` (caps at
    # `s_window`) AND a time-compressed `compressor` latent (grows at
    # ⌈s_kv/m⌉).
    storage_positions_expr: Optional[Any] = None
    read_positions_expr: Optional[Any] = None


@dataclass
class KVCacheTopology:
    """Per-layer KV cache topology for a single attention component.

    `supported=True` iff `eval.cache.Cache.tensors` is non-empty. Callers
    (e.g. the eval cache layer) treat `supported=False` as a signal to
    fall back to the legacy config-dim heuristic.

    `kv_positions_per_query_expr` / `kv_storage_positions_per_query_expr`
    are the first cached tensor's read / stored laws when the component
    has one stream, and None when it has none. Mixed-stream components
    keep the per-node laws on `cached_nodes`.
    """
    component_name: str
    label: Optional[str]
    cached_nodes: List[CachedKVNode]
    elements_per_token: int
    supported: bool
    kv_positions_per_query_expr: Optional[Any] = None
    score_op_name: Optional[str] = None
    kv_storage_positions_per_query_expr: Optional[Any] = None


def _counted(extent: Any, tensor_name: str, what: str) -> int:
    """An integer the cache actually has, not 0 for one it could not value."""
    value = value_of(extent)
    if value is None:
        raise ValueError(
            f"cached tensor {tensor_name!r}: {what} has no numeric value "
            f"({extent!r}); it must not be counted as 0")
    return int(value)


def kv_cache_topology(
    source,
    component_name: str,
    dim_overrides: Optional[Dict[str, int]] = None,
    site: Optional[CallSite] = None,
) -> KVCacheTopology:
    """Cached tensors of one component, from the eval cache layer.

    `site` is the model's call the component is evaluated at
    (`CallSite.of`); None takes its only call, and a component the model
    calls more than once is refused until the caller names one.
    `dim_overrides` rebind model dims before isolation (hybrid templates
    that thin a head count). Resolution failures raise.
    """
    resolved = getattr(source, "resolved", source)
    entry = cache_of_component(
        analyze_isolated(resolved, component_name, dim_overrides, site),
        component_name,
    )
    tensors = entry.tensors if entry is not None else ()
    nodes = [
        CachedKVNode(
            name=tensor.name,
            elements_per_token=_counted(tensor.elements, tensor.name, "elements"),
            inner_dim=(
                _counted(tensor.inner, tensor.name, "inner")
                if tensor.inner is not None else 0
            ),
            dtype=tensor.dtype,
            storage_positions_expr=tensor.stored,
            read_positions_expr=tensor.read,
        )
        for tensor in tensors
    ]
    # Per TOKEN: a recurrent state is one matrix per request, listed among
    # the cached nodes and kept out of the per-token total and laws.
    per_token = [node for node, tensor in zip(nodes, tensors, strict=True)
                 if not tensor.recurrent]
    first = per_token[0] if per_token else None
    return KVCacheTopology(
        component_name=component_name,
        label=entry.label if entry is not None else None,
        cached_nodes=nodes,
        elements_per_token=sum(n.elements_per_token for n in per_token),
        supported=bool(entry is not None and entry.tensors),
        kv_positions_per_query_expr=(
            first.read_positions_expr if first else None),
        kv_storage_positions_per_query_expr=(
            first.storage_positions_expr if first else None),
    )
