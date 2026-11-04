"""Shared overlay-diff + L1 entry points.

`l1_of_nodes` is Playground's structure view (`backend.views.abstraction`);
`totals` is required so the Σ gate stays falsifiable. Inspector / Inference
block and serving graphs do not call it this step (L1 does not apply).

`diff_of_graphs` is the multi-entry overlay (`backend.views.diff`). Stream
pseudo-nodes (`op: stream_in` / `stream_out`) are dropped before `Entry`;
declared streams travel on `reads`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from backend.views.abstraction import L1View, Sums, abstract, has_anchor
from backend.views.diff import DiffRefused, Entry, GraphDiff


def l1_of_nodes(nodes: Sequence[Mapping[str, Any]], totals: Sums) -> Dict[str, Any]:
    """L1 boxes over a rendered node list. `totals` is the independent module
    total (Playground: the cost report); omitting it is an error."""
    if not isinstance(totals, Sums):
        raise TypeError("l1_of_nodes requires totals: Sums")
    return abstract(list(nodes), totals).to_dict()


class NoInnerTotals(ValueError):
    """An inner scope whose own totals cannot be stated per drawn iteration."""


def _inner_totals(node: Mapping[str, Any]) -> Optional[Sums]:
    """Totals for an inner scope, in the same units `abstract` sums.

    FLOPs / params come from the NodeReport copied onto the wrapper — the
    engine's figures, independent of the boxes. The wrapper's
    `weight_bytes` is not: the renderer adds it up from the children drawn
    inside it (`_memory._parent_weight_bytes`), the very sum the boxes
    add, so it is no leg of the gate (None: not compared).

    A repeated block's wrapper carries ALL repeats' totals while the
    children drawn inside it are ONE iteration — the Σ gate compares one
    iteration, which the engine's totals state exactly when they are a
    whole multiple of the repeat; anything else is refused, not rounded.
    """
    if "flops" not in node or "params" not in node:
        return None
    repeat = node.get("repeat") or 1
    if not isinstance(repeat, int) or repeat < 1:
        raise NoInnerTotals(f"repeat {repeat!r} is not a count of iterations")
    flops, params = int(node.get("flops") or 0), int(node.get("params") or 0)
    if flops % repeat or params % repeat:
        raise NoInnerTotals(
            f"totals (flops {flops}, params {params}) are not {repeat} "
            f"equal iterations")
    return Sums(flops // repeat, params // repeat, None)


def stamp_l1(nodes: Sequence[Any]) -> None:
    """Stamp `node['l1']` on inner scopes. Idempotent. Does not write root l1.

    Direct children with no anchor: omit the field. Anchor but no independent
    totals: `available=false`. Otherwise `l1_of_nodes(children, totals)`.
    A fused kernel that ate the score/apply still groups at the component:
    the renderer gave it the roles its absorbed anchors play.
    """
    _stamp_l1_scopes(nodes)


def _stamp_l1_scopes(nodes: Sequence[Any]) -> None:
    for node in nodes:
        kids = node.get("children") or []
        if not kids:
            continue
        _stamp_l1_scopes(kids)
        if "l1" in node:
            continue
        if not has_anchor(kids):
            continue
        try:
            totals = _inner_totals(node)
        except NoInnerTotals as exc:
            node["l1"] = L1View(False, str(exc)).to_dict()
            continue
        if totals is None:
            kid_tot = Sums()
            for k in kids:
                kid_tot.add(k)
            if kid_tot.flops or kid_tot.params or kid_tot.weight_bytes:
                node["l1"] = L1View(False, "no independent totals").to_dict()
                continue
            # Serving graphs carry topology without per-op cost: group
            # anyway, with nothing to reconcile the boxes against.
            totals = Sums(None, None, None)
        node["l1"] = l1_of_nodes(kids, totals)


def _without_stream_in(nodes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for node in nodes:
        if node.get("op") in ("stream_in", "stream_out"):
            continue
        item = dict(node)
        if item.get("children"):
            item["children"] = _without_stream_in(item["children"])
        out.append(item)
    return out


def entry_from_graph(*, key: str, graph: Mapping[str, Any],
                     dims: Optional[Mapping[str, Any]] = None) -> Entry:
    """One overlay entry from a drawn graph. Does not invent L1."""
    nodes = _without_stream_in(graph.get("nodes") or [])
    ids = {n["id"] for n in nodes}
    edges = [e for e in (graph.get("edges") or [])
             if e.get("src") in ids and e.get("dst") in ids]
    stripped = {
        "nodes": nodes,
        "edges": edges,
        "inputs": list(graph.get("inputs") or []),
        "outputs": list(graph.get("outputs") or []),
        "reads": list(graph.get("reads") or []),
    }
    l1 = graph.get("l1")
    if not isinstance(l1, Mapping):
        l1 = {}
    return Entry(
        key=key,
        block_graph=stripped,
        l1=l1,
        dims=dict(dims or {}),
        inputs=stripped["inputs"],
        outputs=stripped["outputs"],
        reads=stripped["reads"],
    )


def diff_of_graphs(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Overlay several graphs. Each item is `{key, graph, dims?}`."""
    if len(items) < 2:
        return GraphDiff.refused("a diff needs at least two entries")
    try:
        return GraphDiff.of([
            entry_from_graph(
                key=str(item["key"]),
                graph=item["graph"],
                dims=item.get("dims"),
            )
            for item in items
        ]).to_dict()
    except DiffRefused as exc:
        return GraphDiff.refused(str(exc))


def diff_representative_blocks(
    results: Sequence[Optional[Mapping[str, Any]]],
    resolved_list: Sequence[Any],
    keys: Sequence[str],
) -> Dict[str, Any]:
    """Inspector Compare: overlay each model's published representative block."""
    items = []
    missing = []
    for key, payload, resolved in zip(keys, results, resolved_list, strict=True):
        picked = (payload or {}).get("representative_template") or {}
        name = picked.get("name")
        block_graph = (payload or {}).get("block_graph") or {}
        if resolved is None or not name or name not in block_graph:
            missing.append(key)
            continue
        items.append({
            "key": key,
            "graph": block_graph[name],
            "dims": picked["dims"],
        })
    if missing:
        return GraphDiff.refused(
            "no representative_template for " + ", ".join(missing)
        )
    return diff_of_graphs(items)
