"""L1 structure view: group a component's detailed (L2) graph into the few
boxes a reader thinks in — the query path, the key path, the routing, the
experts — without changing the graph.

The pass is a view. It takes the L2 node list the renderer already produced
and returns a node -> box map plus per-box sums; the L2 data is untouched, so
a caller can show boxes, expand one box, or drop back to L2 at any time.

Nothing here is authored per component. Boxes come from two declared facts:

* anchors — ops that declare `OpSpec.input_roles` (attention score/apply,
  MoE permute/combine). A branch is everything upstream of one role's input.
* the graph's own edges — backward / forward closure, no names read.

What it refuses to do, rather than guess:

* no anchor, or more than one anchor core in one component;
* an anchor with a parameter the renderer folded that does not record the
  input position it held (the remaining inputs could not be lined up with
  the declared roles);
* boxes whose sums do not equal the module's totals (FLOPs, params, weight
  bytes). A refusal carries its reason; the caller shows L2.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import sympy

from backend.arch.law_text import read_law
from backend.arch.ops_base import INPUT_ROLES
from backend.rendering.kv_marks import merge_kv

@dataclass
class Sums:
    """The cost facts every box carries and every view must reconcile with
    the module's own totals. As totals, a fact with no independent figure is
    None and is not compared."""
    flops: Optional[int] = 0
    params: Optional[int] = 0
    weight_bytes: Optional[int] = 0

    def add(self, node: Mapping[str, Any]) -> None:
        self.flops += _value(node, "flops")
        self.params += _value(node, "params")
        self.weight_bytes += _value(node, "weight_bytes")

    def minus(self, other: Sums) -> Dict[str, int]:
        """Non-zero differences, by fact, over the facts `other` states."""
        a, b = asdict(self), asdict(other)
        return {k: a[k] - b[k] for k in a if b[k] is not None and a[k] != b[k]}

#: Box for the component's own inputs (zero cost, kept so the partition is
#: total), for what the core feeds, and for what reaches neither.
SOURCE_BOX = "input"
CORE_BOX = "core"
OUTPUT_BOX = "output"
OTHER_BOX = "other"


@dataclass
class Box:
    id: str
    members: List[str] = field(default_factory=list)
    sums: Sums = field(default_factory=Sums)
    ops: Dict[str, int] = field(default_factory=dict)
    flops_symbolic: Optional[str] = None
    #: Whether `flops_symbolic` still reads workload axes (a law over B, S…)
    #: rather than having collapsed to a number.
    flops_is_law: bool = False
    #: What a reader calls the box, and where it sits in reading order
    #: (`_named`) — both read off the declared roles, never off names.
    label: str = ""
    rank: int = 0
    #: The cached tensors the members write and read — the union of their
    #: renderer ``kv`` fields, gathered in the same pass as the sums because
    #: `to_dict` no longer holds the member nodes.
    kv: Optional[Dict[str, Any]] = None

    def add(self, node: Mapping[str, Any]) -> None:
        self.members.append(node["id"])
        self.sums.add(node)
        op = node.get("op")
        self.ops[op] = self.ops.get(op, 0) + 1
        self.kv = merge_kv([self.kv, node.get("kv")])

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "members": self.members, **asdict(self.sums),
                "ops": self.ops, "flops_symbolic": self.flops_symbolic,
                "flops_is_law": self.flops_is_law,
                "label": self.label, "rank": self.rank, "kv": self.kv}


@dataclass
class L1View:
    available: bool
    reason: Optional[str] = None
    boxes: List[Box] = field(default_factory=list)
    node_box: Dict[str, str] = field(default_factory=dict)
    totals: Sums = field(default_factory=Sums)
    #: What the core computes, read off the anchors' declared roles:
    #: "recurrence" when a query and a state decay are operands (linear
    #: attention), "attention" for a query alone, "experts" for a routing.
    core: Optional[str] = None
    #: Whether what runs beside the core (the "other" box) merges back into
    #: what the core feeds — a parallel branch (shared experts, an output
    #: gate) rather than a dead end.
    other_merges: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"available": self.available, "reason": self.reason,
                "boxes": [b.to_dict() for b in self.boxes],
                "node_box": self.node_box, "totals": asdict(self.totals),
                "core": self.core, "other_merges": self.other_merges}


def _value(node: Mapping[str, Any], key: str) -> int:
    """A node's own amount plus what the renderer folded into it."""
    own = int(node.get(key) or 0)
    return own + sum(int(a.get(key) or 0) for a in (node.get("absorbed_params") or []))


def _states(node: Mapping[str, Any], key: str) -> bool:
    """Whether the node (or what the renderer folded into it) states `key`."""
    return node.get(key) is not None or any(
        a.get(key) is not None for a in node.get("absorbed_params") or [])


def _roles(node: Mapping[str, Any]) -> Tuple[Optional[str], ...]:
    """The parts the node's operands play, as the renderer stamped them
    from the declaration (a fused kernel: the parts its absorbed anchors
    gave its outer inputs)."""
    return tuple(node.get("input_roles") or ())


def _input_role(node: Mapping[str, Any], position: int) -> Optional[str]:
    roles = _roles(node)
    if 0 <= position < len(roles):
        return roles[position]
    return None


def is_anchor(node: Mapping[str, Any]) -> bool:
    """Whether this node is an L1 anchor (`OpSpec.input_roles`)."""
    return bool(_roles(node))


def has_anchor(nodes: Iterable[Mapping[str, Any]]) -> bool:
    """Whether a scope's direct nodes include an L1 anchor.

    `stamp_l1` and `abstract` share this; do not reimplement it at a caller.
    """
    return any(is_anchor(n) for n in nodes)


def _scope_holders(nodes: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    """Every node drawn in this scope or inside one of its compounds, mapped
    to the node of this scope that holds it. A read of a member of a
    compound (a consumer rewired to a fused kernel inside the router, say)
    is an edge from the compound at this level, as the renderer anchors it
    (`_shapes._anchor`)."""
    out: Dict[str, str] = {}

    def walk(items: Iterable[Mapping[str, Any]], top: str) -> None:
        for item in items:
            out.setdefault(item["id"], top)
            walk(item.get("children") or (), top)

    for n in nodes:
        out[n["id"]] = n["id"]
    for n in nodes:
        walk(n.get("children") or (), n["id"])
    return out


def _producers(node: Mapping[str, Any],
               holders: Mapping[str, str]) -> List[str]:
    """The node of this scope producing each input. A ref to a named output
    port reads `<producer>.<port>` with the port beside it in `input_ports`
    (a fused kernel's `q_out` / `k_out`, a merged projection's `output_0`);
    the edge is to the producer, or to the compound of this scope that
    holds it, as the drawn graph has it."""
    ports = node.get("input_ports") or ()
    out = []
    for i, ref in enumerate(node.get("inputs") or ()):
        port = ports[i] if i < len(ports) else ""
        ref = ref[:-len(port) - 1] if port and ref.endswith("." + port) else ref
        out.append(holders.get(ref, ref))
    return out


def _ports(node: Mapping[str, Any],
           holders: Mapping[str, str]) -> Optional[List[Tuple[int, str]]]:
    """(declared position, producer) for each input the node still shows.
    The renderer folds a parameter into its consumer and drops it from
    `inputs`; the folded entry records the position it held. None when a
    folded entry does not say, so positions cannot be recovered."""
    folded: Set[int] = set()
    for a in node.get("absorbed_params") or []:
        if "input_positions" not in a:
            return None
        folded.update(a["input_positions"])
    shown = iter(_producers(node, holders))
    ports: List[Tuple[int, str]] = []
    position = 0
    for src in shown:
        while position in folded:
            position += 1
        ports.append((position, src))
        position += 1
    return ports


def _closure(start: Iterable[str], step: Mapping[str, List[str]],
             nodes: Mapping[str, Any], stop: Set[str]) -> Set[str]:
    seen: Set[str] = set()
    todo = [s for s in start]
    while todo:
        v = todo.pop()
        if v in seen or v not in nodes or v in stop:
            continue
        seen.add(v)
        todo.extend(step.get(v, ()))
    return seen


def _symbolic_sum(exprs: Iterable[Optional[str]]) -> Optional[sympy.Expr]:
    """Sum of member formulas, simplified. None when any member has no
    formula (a partial sum would read as the whole)."""
    terms = []
    for e in exprs:
        if not e:
            return None
        # Formula text is read the one way the arch reads authored text (a
        # dim named `beta` must stay a dim, not SymPy's beta function).
        # read_law raises ValueError on text it cannot read; abstract() turns
        # that into a refusal rather than a partial formula.
        terms.append(read_law(e))
    if not terms:
        return sympy.Integer(0)
    return sympy.simplify(sympy.Add(*terms))


_ROLE_INDEX = {role.key: i for i, role in enumerate(INPUT_ROLES)}
_ROLE_LABEL = {role.key: role.label for role in INPUT_ROLES}
_LAST_ROLE_RANK = len(INPUT_ROLES) ** 2


def _named(box_id: str, core_label: str,
           other_merges: bool) -> Tuple[str, int]:
    """A box's name and reading-order rank, from its roles alone.

    A box is named for what it produces: the path computing one operand
    ("Q path"), or an upstream several operands are read from ("Shared by
    K, V") — which holds none of them. Role boxes come in the declared
    reading order of their earliest role; among boxes with the same earliest
    role, one read by more roles first (it lies upstream of the others);
    then the core, what it feeds, and what runs beside it.
    """
    if box_id == SOURCE_BOX:
        return "input", -1
    if box_id == CORE_BOX:
        return core_label, _LAST_ROLE_RANK + 1
    if box_id == OUTPUT_BOX:
        return "Output", _LAST_ROLE_RANK + 2
    if box_id == OTHER_BOX:
        return ("Parallel branch" if other_merges else "Beside the core"), _LAST_ROLE_RANK + 3
    roles = sorted(box_id.split("+"), key=_ROLE_INDEX.__getitem__)
    shared = len(roles) > 1
    rank = len(INPUT_ROLES) * _ROLE_INDEX[roles[0]] + (len(INPUT_ROLES) - len(roles))
    names = ", ".join(_ROLE_LABEL[r] for r in roles)
    return (f"Shared by {names}" if shared else f"{names} path"), rank


def abstract(nodes: List[Mapping[str, Any]],
             totals: Sums) -> L1View:
    """Group one component's top-level L2 nodes into L1 boxes.

    `nodes`: the renderer's node dicts for the component (id, op, inputs,
    flops/params/weight_bytes, absorbed_params, flops_symbolic).
    `totals`: the module's own totals, from the cost report — the partition
    must add up to exactly these.
    """
    by_id = {n["id"]: n for n in nodes}
    holders = _scope_holders(nodes)
    ins = {nid: [i for i in _producers(n, holders) if i in by_id]
           for nid, n in by_id.items()}
    outs: Dict[str, List[str]] = {nid: [] for nid in by_id}
    for nid, srcs in ins.items():
        for s in srcs:
            outs[s].append(nid)
    sources = {nid for nid, n in by_id.items() if n.get("op") == "input"}

    anchors = [nid for nid, n in by_id.items() if is_anchor(n)]
    if not anchors:
        return L1View(False, "no anchor op with declared input roles", totals=totals)
    ports = {a: _ports(by_id[a], holders) for a in anchors}
    for a in anchors:
        if ports[a] is None:
            return L1View(False, f"anchor {by_id[a]['op']} had inputs folded by the renderer", totals=totals)

    # Anchors pair into cores: a later anchor that consumes an earlier one
    # (score -> apply, permute -> combine) belongs to the same core.
    down = {a: _closure(outs.get(a, ()), outs, by_id, set()) for a in anchors}
    heads = [a for a in anchors if not any(a in down[b] for b in anchors if b != a)]
    if len(heads) != 1:
        return L1View(False, f"{len(heads)} anchor cores in one component", totals=totals)
    head = heads[0]
    members = [head] + [a for a in anchors if a in down[head]]
    tail = [a for a in members if not any(b in down[a] for b in members if b != a)]
    core: Set[str] = set(members)
    for t in tail:
        core |= down[head] & _closure(ins.get(t, ()), ins, by_id, set())

    # Branch per declared role, upstream of the role's input and outside the core.
    branch: Dict[str, Set[str]] = {}
    for a in members:
        for i, src in ports[a]:
            role = _input_role(by_id[a], i)
            if role is None or src not in by_id or src in core:
                continue
            reach = _closure([src], ins, by_id, sources) - core
            branch.setdefault(role, set()).update(reach)

    # A box several roles reach is named by them in the order the anchors
    # declare them (q, k, v, …), so the name reads like the op's signature.
    declared_order: List[str] = []
    for a in members:
        for r in _roles(by_id[a]):
            if r is not None and r not in declared_order:
                declared_order.append(r)

    owners: Dict[str, Set[str]] = {}
    for role, reach in branch.items():
        for v in reach:
            owners.setdefault(v, set()).add(role)

    downstream: Set[str] = set()
    for t in tail:
        downstream |= _closure(outs.get(t, ()), outs, by_id, set())

    node_box: Dict[str, str] = {}
    for nid in by_id:
        if nid in sources:
            node_box[nid] = SOURCE_BOX
        elif nid in core:
            node_box[nid] = CORE_BOX
        elif nid in owners:
            node_box[nid] = "+".join(sorted(owners[nid], key=declared_order.index))
        elif nid in downstream:
            node_box[nid] = OUTPUT_BOX
        else:
            node_box[nid] = OTHER_BOX

    # A fact no node states (a serving graph draws topology without per-op
    # cost) is unknown on every box, not zero.
    stated = {fact: any(_states(n, fact) for n in nodes)
              for fact in ("flops", "params", "weight_bytes")}
    boxes: Dict[str, Box] = {}
    for nid, bid in node_box.items():
        boxes.setdefault(bid, Box(bid)).add(by_id[nid])
    for b in boxes.values():
        b.members.sort()
        b.ops = dict(sorted(b.ops.items()))
        try:
            # Each member's law in the block's names (`flops_law`), not its
            # own: a window's `s_pairs` is not the phase's.
            law = _symbolic_sum(by_id[m].get("flops_law") or
                                ("0" if stated["flops"]
                                 and not _value(by_id[m], "flops") else None)
                                for m in b.members)
        except ValueError as exc:
            # A formula the arch cannot read back is an upstream bug; the view
            # declines with it, and the module's own numbers still show.
            return L1View(False, f"a box formula does not read back: {exc}", totals=totals)
        if law is not None:
            b.flops_symbolic = str(law)
            b.flops_is_law = bool(law.free_symbols)

    for fact, known in stated.items():
        if not known:
            for b in boxes.values():
                setattr(b.sums, fact, None)

    got = Sums()
    for nid in by_id:
        got.add(by_id[nid])
    delta = got.minus(totals)
    if delta:
        return L1View(False, f"boxes do not add up to the module totals: {delta}",
                      totals=totals)
    declared = set(declared_order)
    if "q" in declared and "decay" in declared:
        core_kind, core_label = "recurrence", "Recurrence core"
    elif "q" in declared:
        core_kind, core_label = "attention", "Attention core"
    elif "routing" in declared:
        core_kind, core_label = "experts", "Routed experts"
    else:
        core_kind, core_label = None, "Core"
    other_merges = any(node_box[nid] == OTHER_BOX and node_box[d] == OUTPUT_BOX
                       for nid in by_id for d in outs[nid])
    for b in boxes.values():
        b.label, b.rank = _named(b.id, core_label, other_merges)
    return L1View(True, None, sorted(boxes.values(), key=lambda b: (b.rank, b.label)),
                  node_box, totals, core_kind, other_merges)
