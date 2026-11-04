"""Per-node memory accounting for the block-graph render.

Reads spec node info already populated by `_attach_shapes` (params,
shape_in/out cached as `_si` / `_so`) and `_quant_match` (weight_dtype
via materialize index),
then fills in:
  • `weight_bytes`    — params × stored bytes per parameter, their
                        quantisation scales included, plus the node's buffers
  • `out_bytes`       — output tensor size in bytes (activation dtype)
  • `access_bytes`    — `traffic.derive` total for that uid
  • `weight_dtype_bytes` / `act_dtype_bytes` — for UI tooltips

Component nodes aggregate bottom-up: sum children's weight_bytes, then
scale by repeat factor (`parent.params / sum(child.params)`) to correctly
handle MoE / expert banks where children show per-expert params but the
parent rolls up ×E.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, NamedTuple

from backend.arch.eval import capacity as capacity_mod
from backend.arch.eval.dim import value_of


def tensor_bytes(shape: Any, dtype_bytes: float) -> int:
    """Bytes required for an activation tensor of `shape` at `dtype_bytes`
    per element. `shape` may be a flat list of ints, a list-of-lists for
    multi-input ops, or empty/None (returns 0). Sub-byte dtypes (INT4=0.5)
    work because the result is `int()`-cast at the call boundary."""
    if not shape or not isinstance(shape, list):
        return 0
    if isinstance(shape[0], list):
        return sum(tensor_bytes(s, dtype_bytes) for s in shape)
    result: float = dtype_bytes
    for d in shape:
        if isinstance(d, (int, float)):
            result *= int(d)
    return int(result)


class StoredWeights(NamedTuple):
    """One copy of a node's stored weights, over the program nodes under it.

    `parameters`, `metadata` and `elements` are summed per copy of each
    tensor's own node, so they price the node's parameters at their width:
    an expert bank counts one expert's rows here and repeats them by its own
    parameter count. `buffers` is per copy of THIS node, because a buffer
    is not priced by parameters — it is what the node holds, whatever its
    parameter count."""

    parameters: float = 0.0
    metadata: float = 0.0
    elements: float = 0.0
    buffers: float = 0.0

    @property
    def width(self) -> float:
        """Stored bytes per parameter, scales left out — the dtype a node's
        tooltip names."""
        return self.parameters / self.elements

    @property
    def priced(self) -> float:
        """Stored bytes per parameter with the scales that come with them."""
        return (self.parameters + self.metadata) / self.elements

    def without(self, other: "StoredWeights") -> "StoredWeights":
        """The rows this node holds outside `other` (a rendered child)."""
        return StoredWeights(self.parameters - other.parameters,
                             self.metadata - other.metadata,
                             self.elements - other.elements,
                             self.buffers)


def _copies(node: Any, instances: Any) -> float:
    """How many copies of `node` the rank holds, as the walk published them.

    A refusal rather than a default of one: the ledger prices every copy, so
    a node the walk counted no copies of is a node whose per-copy width
    cannot be stated at all.
    """
    if node not in instances:
        raise ValueError(
            f"{node.display_path} holds stored weights but the rank "
            f"has no copy count for it")
    return float(instances[node])


def stored_weights(analysis: Any) -> Dict[int, StoredWeights]:
    """`{node uid → StoredWeights}`, read off the weight ledger's `by_uid`
    (`capacity.weights`) — the same bytes the entry's `weight_bytes` sums, so
    a drawn graph adds up to it.

    A quantisation scale goes with the weight it scales: it is priced into
    that weight's bytes per parameter. A buffer goes with the node that
    declares it. The element counts that turn bytes into a width come from
    the rank's tensor rows (`local_tensors`, the ledger's own face), counting
    parameters only, so `width` is the dtype the parameters are stored at
    and not raised by their scales. A packed format answers its true width
    here (FP4 in a U8 slot is half a byte per element).

    Every node also counts toward its ancestors, so a component the renderer
    shows collapsed (a MoE router whose linear is inside it) is priced by the
    rows of the leaves it contains.

    A node's copies are the walk's own count for it (`Analysis.instances`),
    never a tensor row's. The rows are one per (stored tensor × layer
    segment), so a node whose layers store one tensor at two saved formats —
    Step-5's expert bank, FP8 in 63 of its 66 layers and BF16 in 3 — answers
    with each segment's SHARE of the copies, and a width taken off one share
    would price the whole node at the three layers' dtype. For the same
    reason a tensor's elements count once per tensor and not once per
    segment: the segments are one tensor, at one shape, stored differently.
    The width that comes out is the mean over the node's copies — 69 bytes
    per 66 copies of one of Step-5's parameters — which is the one width a
    node drawn once can be drawn at.
    """
    ledger = capacity_mod.weights(analysis).by_uid
    instances = analysis.instances
    elements: Dict[int, float] = {}
    counted: set = set()
    for row in analysis.local_tensors():
        if row.is_metadata or row.is_buffer:
            continue
        # One tensor index, several rows: the layer segments of one tensor,
        # which hold the same elements and only differ in how they are saved.
        if (row.uid, row.tensor_index) in counted:
            continue
        counted.add((row.uid, row.tensor_index))
        elements[row.uid] = elements.get(row.uid, 0.0) + float(row.elements)
    out: Dict[int, StoredWeights] = {}
    for uid, held in ledger.items():
        node = analysis.program.by_uid(uid)
        count = _copies(node, instances)
        per_copy = StoredWeights(float(held.parameters) / count,
                                 float(held.metadata) / count,
                                 elements.get(uid, 0.0))
        buffers = float(held.buffers)
        while node is not None:
            acc = out.get(node.uid, StoredWeights())
            out[node.uid] = StoredWeights(
                acc.parameters + per_copy.parameters,
                acc.metadata + per_copy.metadata,
                acc.elements + per_copy.elements,
                acc.buffers + buffers / _copies(node, instances))
            node = node.parent
    return out


def _own_buffers(n: dict, stored: Mapping[int, StoredWeights]) -> float:
    held = stored.get(n.get("uid"))
    return held.buffers if held is not None else 0.0


def _weight_bytes(n: dict, stored: Mapping[int, StoredWeights]) -> int:
    """`params × bytes per parameter` (scales included) plus the node's
    buffers; a node that stores parameters the ledger has no row for is an
    error, not zero bytes."""
    params = n.get("params", 0)
    if not params:
        return round(_own_buffers(n, stored))
    uid = n.get("uid")
    held = stored.get(uid)
    if held is None or not held.elements:
        raise ValueError(
            f"{n.get('id')!r} stores {params} parameters but "
            f"the weight ledger has no row for uid {uid!r}")
    return round(params * held.priced + held.buffers)


def _parent_weight_bytes(n: dict, stored: Mapping[int, StoredWeights]) -> int:
    """Children's parameter bytes, plus the parameters no rendered child
    carries priced at the width of the rows those children do not cover, plus
    the node's buffers. A collapsed linear next to an F32 bias is priced as
    the linear it is, not as the bias's width scaled by a parameter ratio."""
    children = n["children"]
    buffers = _own_buffers(n, stored)
    # The children's own buffers are the parent's too, per copy of the
    # parent: `buffers` has them, so only their parameter bytes carry over.
    child_bytes = sum(c.get("weight_bytes", 0) - _own_buffers(c, stored)
                      for c in children)
    child_params = sum(c.get("params", 0) for c in children)
    params = n.get("params", 0)
    uncovered = params - child_params
    if uncovered <= 0:
        return round(child_bytes + buffers)
    uid = n.get("uid")
    if uid not in stored:
        raise ValueError(
            f"{n.get('id')!r} holds {uncovered} parameters outside its "
            f"rendered children but the weight ledger has no row for uid {uid!r}")
    rest = stored[uid]
    for c in children:
        if c.get("uid") in stored:
            rest = rest.without(stored[c["uid"]])
    if rest.elements <= 0:
        # Every row is already a child's: the extra parameters are the
        # children repeated (an expert bank counts its experts on the bank,
        # one expert on each child), so the bytes repeat with them.
        if child_params <= 0:
            raise ValueError(
                f"{n.get('id')!r}: {params} parameters, rows only on children "
                f"that hold none")
        return round(child_bytes * params / child_params) + round(buffers)
    return round(child_bytes) + round(uncovered * rest.priced) + round(buffers)


def _set_intensity(n: dict) -> None:
    """FLOPs per byte accessed, or None when either side is zero — the page
    used to divide the two fields itself."""
    flops, access = n.get("flops") or 0, n.get("access_bytes") or 0
    n["arithmetic_intensity"] = (flops / access) if flops > 0 and access > 0 else None


def _access_bytes(n: dict, traffic: Any) -> int:
    uid = n.get("uid")
    if uid is None:
        raise ValueError(
            f"{n.get('id')!r} is not a program node — presentation "
            f"inputs are stamped after pricing")
    row = traffic.values.get(uid)
    if row is None:
        raise ValueError(
            f"{n.get('id')!r} uid {uid!r} has no traffic.derive row")
    return int(value_of(row.total) or 0)


def compute_memory(nodes_list: List[dict], analysis: Any,
                   stored: Mapping[int, StoredWeights], traffic: Any) -> None:
    """Populate `weight_bytes` / `out_bytes` / `access_bytes` on every
    node (including component subtrees) in place. Reads `params`,
    `_si` / `_so` (shape stash from `_attach_shapes`), and `stored`
    (`stored_weights(analysis)`: the weight ledger by node uid).
    `weight_dtype` stays the label `_quant_match` attached; it no longer
    prices anything. Leaf `access_bytes` is `traffic.derive[uid].total`.

    Activation occupancy is the walk's edge width for the node
    (`ResolvedPrecision.edge_bytes`) — the same width HBM traffic and the
    capacity peak use.
    """
    for n in nodes_list:
        n.pop("_si", None)
        so = n.pop("_so", None)
        held = stored.get(n.get("uid"))
        if held is not None and held.elements:
            n["weight_dtype_bytes"] = held.width
        uid = n.get("uid")
        node = analysis.program.by_uid(uid) if uid is not None else None
        bpe = analysis.resolved_precision.edge_bytes(
            n.get("role"),
            analysis.program.stack_of(node) if node is not None else "")
        n["act_dtype_bytes"] = bpe
        n["out_bytes"] = tensor_bytes(so, bpe)

        if n.get("children"):
            compute_memory(n["children"], analysis, stored, traffic)
            n["weight_bytes"] = _parent_weight_bytes(n, stored)
            if n.get("uid") in traffic.values:
                n["access_bytes"] = _access_bytes(n, traffic)
            else:
                n["access_bytes"] = sum(
                    c.get("access_bytes", 0) for c in n["children"])
            if n.get("out_bytes", 0) == 0:
                n["out_bytes"] = sum(c.get("out_bytes", 0) for c in n["children"])
            _set_intensity(n)
        else:
            n["weight_bytes"] = _weight_bytes(n, stored)
            n["access_bytes"] = _access_bytes(n, traffic)
            _set_intensity(n)
