"""The model per declaration: derived groups shown under the declaration they
were split from.

A template that reads the prefix of its stack's `read: all` stream sees a
history whose length grows along the stack, so block enumeration splits its
declaration into one invocation per length (`BlockMeta.visible`) — Kimi-K3's
`block_kda_moe` becomes eight. The engine keeps them apart, and so does every
raw report: their shapes and costs differ. A reader, though, thinks in the
declarations the spec wrote. This view joins the groups back under their
declaration for display — a parent carrying the sums and the `m -> layers`
histogram, the groups as its children — and changes nothing it is given.

Nothing is read out of a group's name: each row and drawn node carries its
group as fields (`BlockMeta.derived_group`), and this view groups on those.
A model without derived groups passes through untouched.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from backend.inference.declaration_groups import declaration_key, declaration_terms
from backend.inference.intensity import arithmetic_intensity, intensity_formula
from backend.rendering.kv_marks import merge_kv

@dataclass
class RowSums:
    """The per-op row columns a parent row shows as the sum of its
    children's: the step's work, traffic and time, and how many times the op
    runs. Each field is a row column of the same name."""
    flops_per_rank: float = 0
    access_bytes_per_rank: float = 0
    t_compute_s: float = 0
    t_memory_s: float = 0
    t_comm_s: float = 0
    t_sol_s: float = 0
    t_op_s: float = 0
    count_per_step: float = 0

    def add(self, row: Mapping[str, Any]) -> None:
        for name, value in asdict(self).items():
            setattr(self, name, value + (row.get(name) or 0))

def histogram(groups: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """`m -> layers` over some derived groups of one declaration, by m, with
    the layers each length covers."""
    by_m: Dict[int, Dict[str, Any]] = {}
    for g in groups:
        entry = by_m.setdefault(g["visible"], {
            "visible": g["visible"], "layers": 0, "layer_indices": []})
        entry["layers"] += g["count"]
        entry["layer_indices"].extend(g["layer_indices"])
    for entry in by_m.values():
        entry["layer_indices"] = sorted(set(entry["layer_indices"]))
    return [by_m[m] for m in sorted(by_m)]


def _declaration(group: Mapping[str, Any]) -> Dict[str, Any]:
    return {**group["declaration"], "stream": group["stream"],
            "axis": group["axis"]}


def per_op_rows(rows: Sequence[Mapping[str, Any]],
                measured: Optional[Mapping[str, Optional[float]]] = None
                ) -> List[Dict[str, Any]]:
    """The per-op rows of derived groups, one parent per op of each
    declaration.

    A parent joins the rows of one op (its path in the block and its op) in
    every group of one declaration (`declaration_groups.declaration_terms`,
    the one grouping). It carries `children` (indices into `rows`, in row
    order), the `RowSums` columns summed over them, the histogram and the
    layer total, and the input / output shape when every child has the same.
    Rows of other invocations get no parent. Parents are in the order of
    their first child.

    `measured` is the measured time per parent, as the one measured formula
    states it (`batch_sweep.apply_measured` → `Folded.subtotal`: a share only
    when every term under it is measured). Nothing is summed here — a sum of
    children would be a second measured aggregation, and it would stop being
    the formula's answer the day a term's weight or the coverage rule
    changes. No `measured` at all means no measurement was read."""
    out: List[Dict[str, Any]] = []
    for name, indices in declaration_terms(rows):
        children = [rows[i] for i in indices]
        path = children[0]["display_path"]
        op = str(children[0].get("op") or "")
        groups = [c["derived_group"] for c in children]
        sums = RowSums()
        for c in children:
            sums.add(c)
        parent: Dict[str, Any] = {
            "declaration": _declaration(groups[0]),
            "path": path,
            "op": op,
            "children": indices,
            "histogram": histogram(groups),
            "layers": sum(g["count"] for g in groups),
            **asdict(sums),
            "bound": None,
            # A sum of groups has no single winning bound; its intensity is
            # the summed FLOPs over the summed bytes.
            "arithmetic_intensity": arithmetic_intensity(
                sums.flops_per_rank, sums.access_bytes_per_rank),
            "arithmetic_intensity_formula": intensity_formula(
                sums.flops_per_rank, sums.access_bytes_per_rank),
            "t_measured_s": (measured or {}).get(name),
        }
        for col in ("shape_in", "shape_out"):
            values = [c.get(col) for c in children]
            parent[col] = values[0] if all(v == values[0] for v in values) else None
        out.append(parent)
    return out


def fold_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    """The serving graph with the top-level block stacks of each declaration's
    derived groups folded into one stack for the declaration.

    The folded box stands where its groups stood: it reads what the first
    group read and feeds what the last group fed, carries the layer total as
    its `count`, the declaration and histogram as `derived_groups`, and the
    groups — unchanged — as its children, joined by the root edges that ran
    between them. Two or more consecutive stacks of one declaration fold, and
    a fold is named by its declaration and its first layer, since groups of
    one declaration that another declaration's layers interleave fold apart;
    the graph is otherwise returned as it came."""
    nodes: List[Dict[str, Any]] = graph["nodes"]
    runs: List[List[int]] = []
    for index, node in enumerate(nodes):
        group = node.get("derived_group") if node.get("op") == "block_stack" else None
        if group is None:
            continue
        if (runs and runs[-1][-1] == index - 1
                and declaration_key(nodes[runs[-1][-1]]["derived_group"])
                == declaration_key(group)):
            runs[-1].append(index)
        else:
            runs.append([index])
    runs = [run for run in runs if len(run) > 1]
    if not runs:
        return graph

    folded_into: Dict[str, str] = {}
    replace_at: Dict[int, Dict[str, Any]] = {}
    drop: set = set()
    for run in runs:
        members = [nodes[i] for i in run]
        groups = [m["derived_group"] for m in members]
        decl = _declaration(groups[0])
        fold_id = ("__declaration__" + ".".join(map(str, decl["layer_path"]))
                   + f":{decl['assignment_index']}"
                   + f"@{min(groups[0]['layer_indices'])}")
        hist = histogram(groups)
        total = sum(g["count"] for g in groups)
        first, last = members[0], members[-1]
        ms = [h["visible"] for h in hist]
        span = f"{ms[0]}" if len(ms) == 1 else f"{ms[0]}–{ms[-1]}"
        fold: Dict[str, Any] = {
            "id": fold_id,
            "op": "block_stack",
            "label": (f"{groups[0]['stack_name'] or decl['base_template']} × {total}  [{decl['base_template']} · "
                      f"{decl['axis']} {span}]"
                      + (f"  [{first['aux_kind']}]" if first.get("aux_kind") else "")),
            "style": first.get("style"),
            "count": total,
            "aux_kind": first.get("aux_kind"),
            "is_moe": any(m.get("is_moe") for m in members),
            "inputs": list(first.get("inputs") or []),
            "input_ports": list(first.get("input_ports") or []),
            "children": members,
            "expandable": True,
            "derived_groups": {"declaration": decl, "histogram": hist,
                               "layers": total},
            "hover": "\n".join(
                f"{decl['axis']}={h['visible']}: {h['layers']} layer(s) "
                f"{h['layer_indices']}" for h in hist),
        }
        for key in ("tensor_in", "tensor_in_full"):
            if first.get(key):
                fold[key] = first[key]
        for key in ("tensor_out", "tensor_out_full"):
            if last.get(key):
                fold[key] = last[key]
        kv = merge_kv(m.get("kv") for m in members)
        if kv is not None:
            fold["kv"] = kv
        for m in members:
            folded_into[m["id"]] = fold_id
        replace_at[run[0]] = fold
        drop.update(run[1:])

    def outside(ref: str) -> str:
        return folded_into.get(ref, ref)

    out_nodes: List[Dict[str, Any]] = []
    for index, node in enumerate(nodes):
        if index in drop:
            continue
        if index in replace_at:
            node = replace_at[index]
            node["inputs"] = [outside(r) for r in node["inputs"]]
            out_nodes.append(node)
            continue
        refs = [outside(r) for r in node.get("inputs") or []]
        if refs != list(node.get("inputs") or []):
            node = {**node, "inputs": refs}
        out_nodes.append(node)

    # Root edges: between two groups of one fold they move inside it; one end
    # in a fold is redrawn to the fold, once per (src, dst).
    by_fold: Dict[str, List[Dict[str, Any]]] = {}
    out_edges: List[Dict[str, Any]] = []
    seen: set = set()
    for edge in graph.get("edges") or []:
        src, dst = edge["src"], edge["dst"]
        sf, df = folded_into.get(src), folded_into.get(dst)
        if sf is not None and sf == df:
            by_fold.setdefault(sf, []).append(edge)
            continue
        if sf is None and df is None:
            out_edges.append(edge)
            continue
        src, dst = outside(src), outside(dst)
        if (src, dst) in seen:
            continue
        seen.add((src, dst))
        # The id is `src->dst` and the input's position; the position stays.
        tail = edge["id"][len(f"{edge['src']}->{edge['dst']}"):]
        out_edges.append({**edge, "id": f"{src}->{dst}{tail}",
                          "src": src, "dst": dst})
    for fold in replace_at.values():
        fold["child_edges"] = by_fold.get(fold["id"], [])
    return {**graph, "nodes": out_nodes, "edges": out_edges}
