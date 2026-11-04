"""Compute-graph diff: what several components share, and where they part.

The Playground compares components of different structure — GQA beside MLA,
MLA beside MLA with a sparse selection, a delta rule beside a state-space
scan — and the same component under two bindings. This view lines their
detailed (L2) graphs up node by node and reports three things: the skeleton
every entry has, the paths only some entries have, and the paired nodes whose
declared attributes differ.

Nothing here is authored per component, per family or per op. A node's
identity is the declaration card its renderer stamped from the leaf's own
declaration (`SpecIRNode.declaration`): for a component, the sequence of its
children's cards. The alignment walks
from anchors — the module inputs and outputs in declared order, and the ops
that declare operand roles — along declared ports, one step at a time, and
accepts only a candidate whose two ends are unique in that round. Where the
walk cannot go on, it stops; the rest is reported as paths, not guessed.

It reads no node id, label, tensor name, component name or dim value to
decide a pairing. Ids appear only as the keys the payload hands back.

Refusals, rather than guesses: a leaf that is neither a registered primitive,
a custom op of its spec, nor a component with children; an input list the
renderer folded without recording positions; classes that would hold two
nodes of one entry, or that order into a cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import sympy

from backend.arch.law_text import read_law
from backend.arch.op_names import is_view_name
from backend.rendering.kv_marks import merge_kv
from backend.views.abstraction import Sums, _ports, _scope_holders, _value

#: How a pair was found (payload `basis`), one word per rule of the walk.
INPUT = "input"
OUTPUT = "output"
ANCHOR = "anchor"
ANCHOR_SUBSTITUTED = "anchor-substituted"
BACKTRACK = "backtrack"
FORWARD = "forward"
SAME_SLOT = "same-slot"

#: How the skeleton is grouped: by the entries' L1 boxes, or — when some
#: entry has no L1 view — by topological layer.
BY_BOX = "l1"
BY_LAYER = "topological"

class DiffRefused(ValueError):
    """The diff declines, with the reason the caller shows."""


@dataclass(frozen=True)
class Declared:
    """What the registry (or the spec's custom op) declares about a node.

    `identity` compares the declaration itself: the registry object, the
    custom op's declared content, or the tuple of a component's children.
    `sigma` is the coarser declared signature — ops whose declarations
    coincide field for field share it — used only for a same-slot
    substitution.
    """
    identity: Any
    sigma: Any
    roles: Tuple[Optional[str], ...]
    view: bool
    output_kind: Optional[str]


def declared(node: Mapping[str, Any]) -> Declared:
    """Identity and signature of one rendered node, from its declaration card."""
    op = node["op"]
    children = node.get("children") or []
    if children:
        parts = [declared(c) for c in children]
        return Declared(tuple(p.identity for p in parts),
                        tuple(p.sigma for p in parts), (), is_view_name(op), None)
    card = node.get("declaration")
    if card is None:
        raise DiffRefused(
            f"node {node['id']!r}: op {op!r} carried no declaration card")
    # `sigma` already carries the output kind beside every other declared
    # field; a transport-only top-level spelling would leak back into the API.
    return Declared(card["identity"], tuple(card["sigma"]),
                    tuple(card["roles"]), card["view"], card["sigma"][1])


class Entry:
    """One compare entry's L2 graph, with its declared ports.

    `inputs` / `outputs` are the component's declared interface
    (`BlockGraph.inputs` / `.outputs`), in order; `reads` its declared stream
    reads; `dims` the dims its frame binds (`BlockReport.dims`).
    """

    def __init__(self, *, key: str, block_graph: Mapping[str, Any],
                 l1: Mapping[str, Any], dims: Mapping[str, Any],
                 inputs: Sequence[str], outputs: Sequence[str],
                 reads: Sequence[str]) -> None:
        self.key = key
        self.l1 = l1
        self.dims = dict(dims)
        self.reads = list(reads)
        self.nodes: Dict[str, Mapping[str, Any]] = {n["id"]: n for n in block_graph["nodes"]}
        self.order = {nid: i for i, nid in enumerate(self.nodes)}
        self.decl: Dict[str, Declared] = {}
        # The interface: the formal inputs and the streams the component
        # declares it reads (`reads`). The drawing shows a read stream as a
        # source named after it; like a formal input it has no declaration
        # to compare. A drawn source the interface does not name is refused
        # by `declared`.
        interface = list(inputs) + [r for r in self.reads if r not in inputs]
        for nid, n in self.nodes.items():
            if n["op"] != INPUT and nid not in interface:
                self.decl[nid] = declared(n)
        # ins[v][port] = source; outs[s] = [(port at the consumer, consumer)]
        self.ins: Dict[str, Dict[Any, str]] = {}
        self.outs: Dict[str, List[Tuple[Any, str]]] = {nid: [] for nid in self.nodes}
        holders = _scope_holders(block_graph["nodes"])
        for nid, n in self.nodes.items():
            ports = _ports(n, holders)
            if ports is None:
                raise DiffRefused(f"node {nid!r} had inputs folded without their positions")
            self.ins[nid] = {}
            for pos, src in ports:
                if src not in self.nodes:
                    continue
                port = self.port(nid, pos)
                if port in self.ins[nid]:
                    raise DiffRefused(f"node {nid!r} declares operand role {port!r} twice")
                self.ins[nid][port] = src
                self.outs[src].append((port, nid))
        self.sources = [i for i in interface if i in self.nodes and i not in self.decl]
        by_name = {n.get("tensor_name"): nid for nid, n in self.nodes.items()}
        self.sinks = [by_name[o] for o in outputs if o in by_name]

    def port(self, nid: str, position: int) -> Any:
        """A declared operand role where the op declares one, else the
        input position."""
        roles = self.decl[nid].roles if nid in self.decl else ()
        if position < len(roles) and roles[position]:
            return roles[position]
        return position

    def anchors(self) -> List[str]:
        return [nid for nid, d in self.decl.items() if any(d.roles)]

    def identity(self, nid: str) -> Any:
        return self.decl[nid].identity if nid in self.decl else INPUT

    def has_l1(self) -> bool:
        return bool(self.l1.get("available"))

    def box_of(self, nid: str) -> Optional[str]:
        return self.l1["node_box"].get(nid) if self.has_l1() else None

    def box(self, box_id: str) -> Mapping[str, Any]:
        return next(b for b in self.l1["boxes"] if b["id"] == box_id)

    def box_sums(self) -> Dict[str, Sums]:
        return {b["id"]: Sums(b["flops"], b["params"], b["weight_bytes"])
                for b in self.l1["boxes"]} if self.has_l1() else {}

    def totals(self) -> Sums:
        t = self.l1["totals"]
        return Sums(t["flops"], t["params"], t["weight_bytes"])


class Alignment:
    """One pair of entries, lined up by the anchored walk.

    `pairs` maps a node of `a` to its partner in `b`; `basis` says which rule
    found each pair. Every round accepts only candidates whose two ends are
    unique in that round and whose already-paired neighbours sit at the same
    ports, so the result does not depend on the order anything is visited in:
    `Alignment.of(b, a)` is the mirror of `Alignment.of(a, b)`.
    """

    def __init__(self, a: Entry, b: Entry) -> None:
        self.a, self.b = a, b
        self.pairs: Dict[str, str] = {}
        self.back: Dict[str, str] = {}
        self.basis: Dict[str, str] = {}

    @classmethod
    def of(cls, a: Entry, b: Entry) -> Alignment:
        al = cls(a, b)
        al._walk()
        return al

    def _walk(self) -> None:
        a, b = self.a, self.b
        self._accept({p: INPUT for p in zip(a.sources, b.sources, strict=False)})
        self._accept({p: OUTPUT for p in zip(a.sinks, b.sinks, strict=False)})
        anchored = {}
        for u in a.anchors():
            for v in b.anchors():
                if set(filter(None, a.decl[u].roles)) & set(filter(None, b.decl[v].roles)):
                    same = a.identity(u) == b.identity(v)
                    anchored[(u, v)] = ANCHOR if same else ANCHOR_SUBSTITUTED
        self._accept(anchored)
        while self._accept(self._neighbours()) or self._accept(self._same_slot()):
            pass

    def _neighbours(self) -> Dict[Tuple[str, str], str]:
        """R1 (backtrack) and R2 (forward) candidates of one round."""
        a, b = self.a, self.b
        back: Set[Tuple[str, str]] = set()
        forward: Set[Tuple[str, str]] = set()
        for u, v in list(self.pairs.items()):
            for port, su in a.ins[u].items():
                sv = b.ins[v].get(port)
                if sv is not None and a.identity(su) == b.identity(sv):
                    back.add((su, sv))
            fu = self._fresh_consumers(a, u, self.pairs)
            fv = self._fresh_consumers(b, v, self.back)
            for k, cs in fu.items():
                ds = fv.get(k, [])
                if len(cs) == 1 and len(ds) == 1:
                    forward.add((cs[0], ds[0]))
        out = {p: FORWARD for p in forward}
        out.update({p: BACKTRACK for p in back})
        return out

    @staticmethod
    def _fresh_consumers(g: Entry, nid: str, paired: Mapping[str, str]
                         ) -> Dict[Tuple[Any, Any], List[str]]:
        found: Dict[Tuple[Any, Any], List[str]] = {}
        for port, c in g.outs[nid]:
            if c not in paired:
                found.setdefault((port, g.identity(c)), []).append(c)
        return found

    def _same_slot(self) -> Dict[Tuple[str, str], str]:
        """R3: unpaired producers at the same port of a paired pair, with
        equal signatures and every input of their own already paired
        port for port."""
        a, b = self.a, self.b
        out: Dict[Tuple[str, str], str] = {}
        for u, v in list(self.pairs.items()):
            for port, su in a.ins[u].items():
                sv = b.ins[v].get(port)
                if sv is None or su in self.pairs or sv in self.back:
                    continue
                if su not in a.decl or sv not in b.decl or a.decl[su].sigma != b.decl[sv].sigma:
                    continue
                qa, qb = a.ins[su], b.ins[sv]
                if qa and set(qa) == set(qb) and all(self.pairs.get(qa[k]) == qb[k] for k in qa):
                    out[(su, sv)] = SAME_SLOT
        return out

    def _accept(self, candidates: Mapping[Tuple[str, str], str]) -> int:
        live = [p for p in candidates if p[0] not in self.pairs and p[1] not in self.back]
        na: Dict[str, int] = {}
        nb: Dict[str, int] = {}
        for u, v in live:
            na[u] = na.get(u, 0) + 1
            nb[v] = nb.get(v, 0) + 1
        good = [(u, v) for u, v in live
                if na[u] == 1 and nb[v] == 1 and self._consistent(u, v)]
        for u, v in good:
            self.pairs[u] = v
            self.back[v] = u
            self.basis[u] = candidates[(u, v)]
        return len(good)

    def _consistent(self, u: str, v: str) -> bool:
        """Every neighbour already paired on both sides is paired to its
        counterpart at the same port. One side unpaired is an inserted step."""
        a, b = self.a, self.b
        pu, pv = a.ins[u], b.ins[v]
        for port in set(pu) | set(pv):
            su, sv = pu.get(port), pv.get(port)
            if su in self.pairs and (sv is None or (sv in self.back and self.pairs[su] != sv)):
                return False
            if sv in self.back and su is None:
                return False
        for mine, theirs, fwd, rev, nid in ((a, b, self.pairs, self.back, u),
                                             (b, a, self.back, self.pairs, v)):
            for port, c in mine.outs[nid]:
                d = fwd.get(c)
                if d is None:
                    continue
                other = theirs.ins[d].get(port)
                if other is not None and other in rev:
                    return False
        return True


def _kv_summary(kv: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The cache facts a comparison reads off a node's (or a group's) `kv`."""
    if not kv:
        return None
    return {"writes": [e["tensor"] for e in kv["writes"]],
            "reads": [e["tensor"] for e in kv["reads"]],
            "write_bytes_per_token": kv["write_bytes_per_token"],
            "read_bytes_per_token": kv["read_bytes_per_token"],
            "state_bytes_per_request": kv["state_bytes_per_request"]}


def _symbols(value: Any) -> Set[str]:
    """Names in what `read_law` returned: an expression, or a shape list."""
    if isinstance(value, (list, tuple)):
        return set().union(*(_symbols(v) for v in value)) if value else set()
    if isinstance(value, sympy.Basic):
        return {str(s) for s in value.free_symbols}
    return set()


def _sums_dict(s: Sums) -> Dict[str, int]:
    return {"params": s.params, "flops": s.flops, "weight_bytes": s.weight_bytes}


def _delta(a: Sums, b: Sums) -> Dict[str, int]:
    return {"params": a.params - b.params, "flops": a.flops - b.flops,
            "weight_bytes": a.weight_bytes - b.weight_bytes}


class GraphDiff:
    """N entries lined up into classes of corresponding nodes.

    Pairs come from the pairwise alignments; a pair is kept only when every
    other entry agrees (the two nodes have the same partner there, or both
    none), and the kept pairs merge into classes. The result does not depend
    on the order of the entries.
    """

    def __init__(self, entries: List[Entry]) -> None:
        self.entries = entries
        self.alignments: Dict[Tuple[int, int], Alignment] = {}
        self.classes: List[List[Optional[str]]] = []
        self.class_of: List[Dict[str, int]] = [{} for _ in entries]
        self._edges: Optional[Dict[Tuple[int, int, Any], List[int]]] = None

    @staticmethod
    def refused(reason: str) -> Dict[str, Any]:
        return {"available": False, "reason": reason}

    @classmethod
    def of(cls, entries: List[Entry]) -> GraphDiff:
        if len(entries) < 2:
            raise DiffRefused("a diff needs at least two entries")
        gd = cls(entries)
        gd._align()
        gd._merge()
        return gd

    # -- alignment ---------------------------------------------------------

    def _align(self) -> None:
        for i, j in combinations(range(len(self.entries)), 2):
            self.alignments[(i, j)] = Alignment.of(self.entries[i], self.entries[j])

    def partner(self, i: int, nid: str, j: int) -> Optional[str]:
        if i < j:
            return self.alignments[(i, j)].pairs.get(nid)
        return self.alignments[(j, i)].back.get(nid)

    def _merge(self) -> None:
        n = len(self.entries)
        parent: Dict[Tuple[int, str], Tuple[int, str]] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for (i, j), al in self.alignments.items():
            for u, v in al.pairs.items():
                if all(self.partner(i, u, z) == self.partner(j, v, z)
                       for z in range(n) if z not in (i, j)):
                    parent[find((i, u))] = find((j, v))
        groups: Dict[Tuple[int, str], List[Tuple[int, str]]] = {}
        for i, e in enumerate(self.entries):
            for nid in e.nodes:
                groups.setdefault(find((i, nid)), []).append((i, nid))
        raw = []
        for members in groups.values():
            row: List[Optional[str]] = [None] * n
            for i, nid in members:
                if row[i] is not None:
                    raise DiffRefused(f"one class would hold two nodes of entry "
                                      f"{self.entries[i].key!r}: {row[i]!r} and {nid!r}")
                row[i] = nid
            raw.append(row)
        self.classes = self._topological(raw)
        for c, row in enumerate(self.classes):
            for i, nid in enumerate(row):
                if nid is not None:
                    self.class_of[i][nid] = c

    def _topological(self, raw: List[List[Optional[str]]]) -> List[List[Optional[str]]]:
        """Classes in an order every entry's edges respect; ties by where
        their members sit in the entries' own node lists."""
        where: Dict[Tuple[int, str], int] = {}
        for c, row in enumerate(raw):
            for i, nid in enumerate(row):
                if nid is not None:
                    where[(i, nid)] = c
        succ: Dict[int, Set[int]] = {c: set() for c in range(len(raw))}
        indeg = {c: 0 for c in range(len(raw))}
        for i, e in enumerate(self.entries):
            for v, ports in e.ins.items():
                for src in ports.values():
                    s, d = where[(i, src)], where[(i, v)]
                    if d not in succ[s]:
                        succ[s].add(d)
                        indeg[d] += 1

        def rank(c):
            return min((i, self.entries[i].order[nid]) for i, nid in enumerate(raw[c])
                       if nid is not None)

        ready = sorted((c for c in indeg if not indeg[c]), key=rank)
        out: List[int] = []
        while ready:
            c = ready.pop(0)
            out.append(c)
            for d in succ[c]:
                indeg[d] -= 1
                if not indeg[d]:
                    ready.append(d)
            ready.sort(key=rank)
        if len(out) != len(raw):
            raise DiffRefused("the classes order into a cycle")
        return [raw[c] for c in out]

    # -- facts per class ---------------------------------------------------

    def owners(self, c: int) -> List[int]:
        return [i for i, nid in enumerate(self.classes[c]) if nid is not None]

    def is_skeleton(self, c: int) -> bool:
        return len(self.owners(c)) == len(self.entries)

    def edges(self) -> Dict[Tuple[int, int, Any], List[int]]:
        """(source class, consumer class, port) -> the entries with that edge."""
        if self._edges is not None:
            return self._edges
        out: Dict[Tuple[int, int, Any], List[int]] = {}
        for i, e in enumerate(self.entries):
            for v, ports in e.ins.items():
                for port, src in ports.items():
                    key = (self.class_of[i][src], self.class_of[i][v], port)
                    out.setdefault(key, []).append(i)
        self._edges = dict(sorted(out.items(), key=lambda kv: (kv[0][1], kv[0][0], str(kv[0][2]))))
        return self._edges

    def _basis(self, c: int) -> List[Dict[str, Any]]:
        """How each pair of the class's owners was found, one record per pair."""
        own = self.owners(c)
        return [{"pair": [i, j], "basis": self.alignments[(i, j)].basis.get(self.classes[c][i])}
                for i, j in combinations(own, 2)]

    def _attributes(self, i: int, nid: str) -> Dict[str, Any]:
        e = self.entries[i]
        n = e.nodes[nid]
        d = e.decl.get(nid)
        return {"op": n["op"],
                "output_kind": d.output_kind if d else None,
                "tensor_in": n.get("tensor_in"),
                "tensor_out": n.get("tensor_out"),
                "params": _value(n, "params"),
                "flops": _value(n, "flops"),
                "weight_dtype": n.get("weight_dtype"),
                "weight_bytes": _value(n, "weight_bytes"),
                "kv": _kv_summary(n.get("kv"))}

    def _diff(self, c: int) -> Dict[str, List[Any]]:
        own = self.owners(c)
        if len(own) < 2:
            return {}
        rows = {i: self._attributes(i, self.classes[c][i]) for i in own}
        idents = {i: self.entries[i].identity(self.classes[c][i]) for i in own}
        out: Dict[str, List[Any]] = {}
        for key in rows[own[0]]:
            values = [rows[i][key] if i in rows else None for i in range(len(self.entries))]
            if key == "op":
                changed = len({idents[i] for i in own}) > 1
            else:
                changed = any(rows[i][key] != rows[own[0]][key] for i in own)
            if changed:
                out[key] = values
        return out

    def _because(self, c: int) -> Dict[str, Any]:
        """The dims in the members' own formulas whose values differ between
        the entries, whether the declarations differ, and the members'
        stored weight dtypes when those differ — each entry's weights are
        priced at its own checkpoint's formats, so one node stores FP8 in
        one checkpoint and BF16 in another with no dim or op changed.
        None when they agree."""
        own = self.owners(c)
        names: Set[str] = set()
        for i in own:
            node = self.entries[i].nodes[self.classes[c][i]]
            for text in (node.get("flops_symbolic"), node.get("params_symbolic"),
                         node.get("tensor_in_symbolic"), node.get("tensor_out_symbolic")):
                if not text:
                    continue
                try:
                    names |= _symbols(read_law(text))
                except ValueError as exc:
                    # A formula the arch cannot read back is an upstream bug;
                    # the diff declines with it rather than fail the bench.
                    raise DiffRefused(f"node {node['id']!r}: a formula does not "
                                      f"read back: {exc}") from exc
        dims = {}
        for name in sorted(names):
            values = [self.entries[i].dims.get(name) if i in own else None
                      for i in range(len(self.entries))]
            if len({values[i] for i in own}) > 1:
                dims[name] = values
        declaration = len({self.entries[i].identity(self.classes[c][i]) for i in own}) > 1
        # `weight_dtype` is the RESIDENT width (`_quant_match` stamps it off
        # `TensorRow.format`, which is `Representation.resident`): what the
        # rank holds, not what the file stores. The two differ wherever a
        # framework loads a checkpoint narrower than it is saved (gemma-2's
        # F32) or a precision profile converts it, so the label beside it
        # says "runs at" rather than "stored as".
        loaded = [self.entries[i].nodes[self.classes[c][i]].get("weight_dtype")
                  if i in own else None for i in range(len(self.entries))]
        differs = len({loaded[i] for i in own}) > 1
        return {"dims": dims, "declaration": declaration,
                "weight_dtype": loaded if differs else None}

    def _view(self, c: int) -> bool:
        """Zero-cost view: folded into its box when the box is collapsed."""
        for i in self.owners(c):
            e, nid = self.entries[i], self.classes[c][i]
            if nid not in e.decl or not e.decl[nid].view:
                return False
            n = e.nodes[nid]
            if _value(n, "params") or _value(n, "flops"):
                return False
        return True

    def _op_label(self, c: int) -> str:
        ops = []
        for i in self.owners(c):
            op = self.entries[i].nodes[self.classes[c][i]]["op"]
            if op not in ops:
                ops.append(op)
        return " | ".join(ops)

    # -- grouping ----------------------------------------------------------

    def grouping(self) -> str:
        return BY_BOX if all(e.has_l1() for e in self.entries) else BY_LAYER

    @cached_property
    def _layers(self) -> List[int]:
        """Longest-path depth of each class; classes are already in
        topological order, so one pass over edges by source settles it."""
        level = [0] * len(self.classes)
        for (s, d, _port) in sorted(self.edges(), key=lambda k: k[0]):
            level[d] = max(level[d], level[s] + 1)
        return level

    @cached_property
    def _rank_index(self) -> Dict[int, int]:
        ranks = sorted({b["rank"] for e in self.entries if e.has_l1() for b in e.l1["boxes"]})
        return {r: k for k, r in enumerate(ranks)}

    def _box_rank(self, box_id: str) -> int:
        """A box's reading-order rank. Ranks are read off declared roles
        alone, so one box id has one rank in every entry."""
        return next(b["rank"] for e in self.entries if e.has_l1()
                    for b in e.l1["boxes"] if b["id"] == box_id)

    def _class_box_rank(self, c: int) -> int:
        i = self.owners(c)[0]
        e = self.entries[i]
        return e.box(e.box_of(self.classes[c][i]))["rank"]

    def paths(self) -> List[List[int]]:
        """Connected runs of non-skeleton classes with the same owners."""
        adj: Dict[int, Set[int]] = {}
        for (s, d, _port) in self.edges():
            adj.setdefault(s, set()).add(d)
            adj.setdefault(d, set()).add(s)
        seen: Set[int] = set()
        out = []
        for c in range(len(self.classes)):
            if c in seen or self.is_skeleton(c):
                continue
            own = self.owners(c)
            comp, todo = set(), [c]
            while todo:
                x = todo.pop()
                if x in comp:
                    continue
                comp.add(x)
                todo.extend(y for y in adj.get(x, ()) if y not in comp
                            and not self.is_skeleton(y) and self.owners(y) == own)
            seen |= comp
            out.append(sorted(comp))
        return out

    # -- payload -----------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        cid = [f"c{k}" for k in range(len(self.classes))]
        paths = self._path_records(cid)
        payload: Dict[str, Any] = {
            "available": True, "reason": None,
            "entries": [e.key for e in self.entries],
            "grouping": self.grouping(),
            "classes": [self._class_record(c, cid) for c in range(len(self.classes))],
            "edges": [{"src": cid[s], "dst": cid[d], "port": p, "owners": own}
                      for (s, d, p), own in self.edges().items()],
            "paths": paths,
            "groups": self._groups(paths),
            "skeleton_boxes": self._skeleton_boxes(cid),
            "reads": [e.reads for e in self.entries],
            "boxes": self._box_table(),
        }
        totals = self._totals()
        if totals is not None:
            payload["totals"] = totals
        return payload

    def _partition(self, c: int) -> int:
        """Layout band: the rank of the class's L1 box among all box ranks,
        or its topological layer when some entry has no L1 view."""
        if self.grouping() == BY_BOX:
            return self._rank_index[self._class_box_rank(c)]
        return self._layers[c]

    def _class_record(self, c: int, cid: List[str]) -> Dict[str, Any]:
        row = self.classes[c]
        diff = self._diff(c)
        record = {
            "id": cid[c],
            "members": row,
            "basis": self._basis(c),
            "boxes": [e.box_of(nid) if nid is not None else None
                      for e, nid in zip(self.entries, row, strict=True)],
            "diff": diff,
            "kv": [_kv_summary(e.nodes[nid].get("kv")) if nid is not None else None
                   for e, nid in zip(self.entries, row, strict=True)],
            "view": self._view(c),
            "partition": self._partition(c),
        }
        if diff:
            record["because"] = self._because(c)
        return record

    def _path_records(self, cid: List[str]) -> List[Dict[str, Any]]:
        edges = self.edges()
        out = []
        for k, members in enumerate(self.paths()):
            inside = set(members)
            own = self.owners(members[0])
            sources = sorted({s for (s, d, _p) in edges if d in inside and s not in inside})
            into = sorted({(d, p) for (s, d, p) in edges if s in inside and d not in inside},
                          key=lambda t: (t[0], str(t[1])))
            sums: List[Optional[Dict[str, int]]] = [None] * len(self.entries)
            # Per owner, its members' KV marks merged — the node `kv` shape.
            kv: List[Optional[Dict[str, Any]]] = [None] * len(self.entries)
            boxes: List[str] = []
            for i in own:
                e = self.entries[i]
                nodes = [e.nodes[self.classes[c][i]] for c in members]
                total = Sums()
                for node in nodes:
                    total.add(node)
                    b = e.box_of(node["id"])
                    if b is not None and b not in boxes:
                        boxes.append(b)
                sums[i] = _sums_dict(total)
                kv[i] = merge_kv(node.get("kv") for node in nodes)
            out.append({
                "id": f"p{k}", "classes": [cid[c] for c in members], "owners": own,
                "from": [cid[s] for s in sources],
                "into": [{"class": cid[d], "port": p} for d, p in into],
                "boxes": sorted(boxes, key=self._box_rank),
                "chain": [self._op_label(c) for c in members if not self._view(c)],
                "sums": sums, "kv": kv, "group": None,
                "partition": min(self._partition(c) for c in members),
            })
        return out

    @staticmethod
    def _groups(paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Paths arriving at one port are alternatives to one another: they
        form a group, and a path sharing a port with a grouped one joins it.
        Sets each grouped path's `group`."""
        parent = list(range(len(paths)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        first: Dict[Tuple[str, Any], int] = {}
        for k, p in enumerate(paths):
            for t in p["into"]:
                key = (t["class"], t["port"])
                if key in first:
                    parent[find(k)] = find(first[key])
                else:
                    first[key] = k
        clusters: Dict[int, List[int]] = {}
        for k in range(len(paths)):
            clusters.setdefault(find(k), []).append(k)
        groups = []
        for ks in clusters.values():
            if len(ks) < 2:
                continue
            gid = f"g{len(groups)}"
            into: List[Dict[str, Any]] = []
            for k in ks:
                paths[k]["group"] = gid
                into.extend(t for t in paths[k]["into"] if t not in into)
            groups.append({"id": gid, "into": into, "paths": [paths[k]["id"] for k in ks]})
        return groups

    def _skeleton_boxes(self, cid: List[str]) -> List[Dict[str, Any]]:
        """The skeleton grouped by the first entry's L1 boxes (every entry
        owns a skeleton class); empty when some entry has no L1 view."""
        if self.grouping() != BY_BOX:
            return []
        first = self.entries[0]
        by_box: Dict[str, List[int]] = {}
        for c in range(len(self.classes)):
            if self.is_skeleton(c):
                by_box.setdefault(first.box_of(self.classes[c][0]), []).append(c)
        out = []
        for bid, cs in sorted(by_box.items(), key=lambda kv: first.box(kv[0])["rank"]):
            box = first.box(bid)
            out.append({"id": bid, "label": box["label"], "rank": box["rank"],
                        "partition": self._rank_index[box["rank"]],
                        "classes": [cid[c] for c in cs],
                        "chain": [self._op_label(c) for c in cs if not self._view(c)]})
        return out

    def _box_table(self) -> List[Dict[str, Any]]:
        """Per L1 box, which entries have it and each one's difference from
        the first entry. The per-entry figures stay in each row's `l1`."""
        if self.grouping() != BY_BOX:
            return []
        sums = [e.box_sums() for e in self.entries]
        ids: Dict[str, Tuple[int, str]] = {}
        for e in self.entries:
            for b in e.l1["boxes"]:
                ids.setdefault(b["id"], (b["rank"], b["label"]))
        return [{"id": bid, "label": label, "rank": rank,
                 "present": [bid in s for s in sums],
                 "delta": [_delta(s.get(bid, Sums()), sums[0].get(bid, Sums())) for s in sums]}
                for bid, (rank, label) in sorted(ids.items(), key=lambda kv: kv[1])]

    def _totals(self) -> Optional[Dict[str, Any]]:
        if any("totals" not in e.l1 for e in self.entries):
            return None
        totals = [e.totals() for e in self.entries]
        base = totals[0]
        delta = [_delta(t, base) for t in totals]
        regrouped = False
        if self.grouping() == BY_BOX and len({t.params for t in totals}) == 1:
            sums = [e.box_sums() for e in self.entries]
            ids = set().union(*sums)
            regrouped = any(len({s.get(b, Sums()).params for s in sums}) > 1 for b in ids)
        return {"delta": delta, "regrouped": regrouped}
