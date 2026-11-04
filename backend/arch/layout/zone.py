"""Routed-zone graph validation — the three rules that retired the route
state machine.

Route identity is implied by SSA structure, so the defense is structural:

  permute and combine consume the SAME routing operand (gate SSA
      value) — a combine fed by another dispatch's gate is the error the
      old route_key guarded against;
  no node inside the zone consumes a token-domain tensor from
      outside it. Unknown cases refuse: the only legal non-zone
      references are the dispatch itself and nodes whose PRODUCER is a
      parameter op — no name-based heuristics, an unknown external
      reference is a violation (the corpus needs zero exceptions);
  no routed edge escapes combine — the zone absorbs every consumer
      of a routed tensor, so an escape surfaces as one of two shapes,
      both caught: mixing in an outside token tensor, or a
      zone tensor reaching a component boundary (outputs, stream_out,
      secondary_outputs) without passing combine (checked here). The
      absorption argument additionally REQUIRES that combine terminates
      this zone: its data input must come from the dispatch's reachable
      set, else the pair is not a pair.

Contracts the corpus already satisfies exactly (checked, not assumed):
both seam ops take exactly two inputs — (tokens, routing) in, (zone
tail, routing) out. Exactly one dispatch pair per graph; nested routing
is a declared non-goal and is reported, loudly, not guessed about.
References of the form ``node.port`` resolve to ``node`` (hyper-connection
components expose multi-output ports); an unresolvable reference is external.
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
from pathlib import Path
from typing import Optional, Tuple

from backend.arch.ops import COMBINE_OP, DISPATCH_OP, is_tensor_source  # re-exported
#: Component-boundary keys whose tensors leave the graph. `outputs` is a
#: list of node refs; `stream_out`/`secondary_outputs` are alias -> node-ref
#: maps (the PRODUCER refs are the values, never the alias keys); cache
#: entries are persistent endpoints — a routed-zone tensor in any of them
#: is an escape.
BOUNDARY_KEYS = ("outputs", "stream_out", "secondary_outputs")


def _inputs(node: dict) -> Tuple[str, ...]:
    raw = node.get("in", ())
    if isinstance(raw, str):
        return (raw,)
    return tuple(raw or ())


def _producer(ref: str, graph: dict) -> Optional[str]:
    """Resolve a reference to its producing node: ``a.port`` -> ``a`` when
    ``a`` is a node; None when the reference is external to the graph."""
    if ref in graph:
        return ref
    base = ref.split(".", 1)[0]
    return base if base in graph else None


def validate_dispatch_region(graph: dict, boundary=()) -> list:
    """Returns violations as strings; empty == the graph is zone-clean.
    `boundary` is every tensor name that leaves the component (outputs +
    stream_out + secondary_outputs). Graphs without a dispatch pair pass
    vacuously."""
    permutes = [n for n, spec in graph.items()
                if spec.get("op") == DISPATCH_OP]
    combines = [n for n, spec in graph.items()
                if spec.get("op") == COMBINE_OP]
    if not permutes and not combines:
        return []
    if len(permutes) != 1 or len(combines) != 1:
        return [f"expected exactly one dispatch pair, found "
                f"permute={permutes} combine={combines} — nested routing "
                "is a declared non-goal and must not be guessed about"]
    dispatch, combine = permutes[0], combines[0]
    out = []

    # Two-input contract, then R1 by EXACT operand identity — set algebra
    # washed out duplicated and smuggled operands.
    d_in = _inputs(graph[dispatch])
    c_in = _inputs(graph[combine])
    if len(d_in) != 2:
        out.append(f"{dispatch}: dispatch must take exactly (tokens, "
                   f"routing), got {list(d_in)}")
    if len(c_in) != 2:
        out.append(f"{combine}: combine must take exactly (zone tail, "
                   f"routing), got {list(c_in)}")
    if len(d_in) == 2 and len(c_in) == 2 and d_in[1] != c_in[1]:
        out.append(f"dispatch {dispatch} routes on {d_in[1]!r} but "
                   f"combine {combine} on {c_in[1]!r} — the pair must "
                   "consume the same routing SSA operand")

    # Zone: forward reachability from dispatch, stopping at combine.
    # Consumers index resolves ``node.port`` references to their producer.
    consumers: dict = {}
    for name, spec in graph.items():
        for ref in _inputs(spec):
            prod = _producer(ref, graph)
            if prod is not None:
                consumers.setdefault(prod, []).append(name)
    zone = set()
    frontier = [dispatch]
    while frontier:
        node = frontier.pop()
        for consumer in consumers.get(node, ()):
            if consumer == combine or consumer in zone:
                continue
            zone.add(consumer)
            frontier.append(consumer)

    # The absorption argument needs combine to TERMINATE this zone: its
    # data input must come from the dispatch's reachable set.
    if len(c_in) == 2:
        tail = _producer(c_in[0], graph)
        if tail not in zone and tail != dispatch:
            out.append(f"combine {combine} consumes {c_in[0]!r}, which "
                       "is not reachable from the dispatch — this pair "
                       "does not bracket one zone")

    # Termination of every branch: the zone absorbs all of the
    # dispatch's successors, so a routed branch that never reaches combine
    # is dead routed data — schema-level dead-end warnings do not cover
    # routed tensors, and silence here would hide a modeling hole.
    reaches_combine = {combine}
    changed = True
    while changed:
        changed = False
        for node in zone:
            if node in reaches_combine:
                continue
            if any(c in reaches_combine for c in consumers.get(node, ())):
                reaches_combine.add(node)
                changed = True
    out.extend(f"routed node {node} never reaches combine — a "
               "dead routed branch is a modeling hole, not a warning"
               for node in sorted(zone - reaches_combine))

    # A zone node may reference the zone, the dispatch,
    # or a parameter-producing node. Everything else — including any
    # reference this graph cannot resolve — is a violation.
    for node in sorted(zone):
        for ref in _inputs(graph[node]):
            prod = _producer(ref, graph)
            if prod in zone or prod == dispatch:
                continue
            if prod is not None and is_tensor_source(graph[prod].get("op") or ""):
                continue
            out.append(f"zone node {node} consumes {ref!r} from "
                       "outside the routed zone — token-domain tensors "
                       "join after combine, never inside")

    # A routed tensor may reach a component boundary only through
    # combine. (No separate edge loop: zone discovery absorbs every
    # consumer, so mixing an outside token tensor is the check above,
    # and a flow-through escape taints the boundary tensor caught here.)
    for name in boundary:
        prod = _producer(name, graph)
        if prod in zone or prod == dispatch:
            out.append(f"boundary tensor {name!r} is a routed-zone "
                       "tensor — it must exit through combine")
    return out


def _refs(val) -> list:
    """Node references from any boundary shape: str, list of str/{node: ..},
    or alias->ref dict (values are the producers, keys are display names)."""
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, dict):
        return [r for v in val.values() for r in _refs(v)]
    out = []
    for item in val:
        if isinstance(item, dict):
            if "node" in item:
                out.append(item["node"])
        else:
            out.append(item)
    return out


def graph_boundary(doc: dict) -> Tuple[str, ...]:
    """Every tensor name that leaves a component document — declared
    outputs and streams, plus persistent cache entries (endpoints must
    never sit inside a routed zone)."""
    names = []
    for key in BOUNDARY_KEYS:
        names.extend(_refs(doc.get(key)))
    cache = doc.get("cache")
    if isinstance(cache, dict):
        names.extend(_refs(cache.get("entries")))
    return tuple(names)


def iter_component_graphs(root: Path):
    """Yield (path, graph, boundary) for every YAML under `root` that
    declares a graph — both .yaml and .yml, and a parse failure raises:
    a gate that silently skips its inputs is no gate."""
    for pattern in ("*.yaml", "*.yml"):
        for path in sorted(root.rglob(pattern)):
            doc = load_yaml(path.read_text())
            if isinstance(doc, dict) and isinstance(doc.get("graph"), dict):
                yield path, doc["graph"], graph_boundary(doc)
