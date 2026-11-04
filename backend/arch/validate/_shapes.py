"""Cross-input geometry validation on multi-input primitive ops.

For every multi-input op in every used block template, verifies that the
upstream-derived input shapes are mutually consistent in the way the op
needs:
  • `axis_concat`        — all inputs share rank; non-axis dims agree
  • exact-template ops   — every axis the op's signature contracts away has
                           one extent across the inputs that carry it

Returns a list of `shape_mismatches` dicts with `{template, node, op,
kind, shapes, msg}`. Surfaces dataflow geometry bugs at validate time
that would otherwise only show up at frontend render time.
"""
from __future__ import annotations

from dataclasses import replace

from backend.arch.eval.dim import value_of
from backend.arch.shape_node import ShapeProbe, _Ref

from ._finding import Finding


def _input_shapes_of(report) -> list:
    """Coerce NodeReport.shape_in into a list of input shapes (one per
    upstream input). Single-input ops carry a flat list, multi
    carry list-of-lists."""
    si = report.shape_in
    if si is None:
        return []
    if si and isinstance(si[0], list):
        return [list(s) for s in si]
    return [list(si)]


def _equal_dim(a, b) -> bool:
    """Two evaluated extents agree. `value_of` is the one reading of a
    `Dim` / number this package makes; a symbol that never resolved is
    compared as itself and is not equal to a number."""
    return a == b or value_of(a) == value_of(b)


def _shape_kind_of(resolved, op_name):
    """The op's ShapeKind from the registered primitives, else the spec's
    `custom_ops` table — both compile to the same DimMap, so they share the
    same cross-input consistency rules here. None when the op has none."""
    from backend.arch.ops import OP_REGISTRY, ShapeKind

    op_spec = OP_REGISTRY.get(op_name)
    if op_spec is not None:
        return op_spec.shape_kind
    custom = (resolved.spec.custom_ops or {}).get(op_name)
    if custom is None or not custom.shape_kind:
        return None
    # `CustomOpSpec._validate_shape_kind` rejected anything else at load.
    return ShapeKind(custom.shape_kind)


def _check_axis_concat(nr, shapes) -> list:
    """Non-axis dims must match (or broadcast); ranks must agree."""
    issues = []
    ranks = {len(s) for s in shapes}
    if len(ranks) > 1:
        issues.append(Finding(
            kind="rank_mismatch", node=nr.name, op=nr.op, shapes=shapes,
            detail=f"input ranks {sorted(ranks)} disagree",
        ))
        return issues
    a_shape = shapes[0]
    rank = next(iter(ranks))
    ax = nr.axis if nr.axis is not None else -1
    ax = ax if ax >= 0 else rank + ax
    for j, b_shape in enumerate(shapes[1:], start=1):
        for i in range(rank):
            if i == ax:
                continue
            if _equal_dim(a_shape[i], b_shape[i]):
                continue
            if value_of(a_shape[i]) == 1 or value_of(b_shape[i]) == 1:
                continue
            issues.append(Finding(
                kind="concat_dim_mismatch", node=nr.name, op=nr.op,
                shapes=shapes, axis=ax,
                detail=f"axis {i}: input0={a_shape[i]} vs input{j}={b_shape[i]} "
                    f"(only axis {ax} is concat-summed)",
            ))
            return issues
    return issues


def _check_contract_pair(nr, shapes) -> list:
    """Every axis the op contracts away has one extent across the inputs
    that carry it. The contracted axes are read from the op's own signature
    (`signature_for`: the `shape_in` einsum template plus `shape_kind`) —
    a label present at two or more input positions and absent from the
    output. Only where the op declares that template a contract
    (`OpSpec.input_rank_is_exact`); an illustrative template says nothing
    about which axes meet. A template that does not fit the wired ranks is
    `edge_consistency_errors`' finding, not this one."""
    from backend.arch.axis_propagation import signature_for
    from backend.arch.ops import OP_REGISTRY

    op_spec = OP_REGISTRY.get(nr.op)
    if op_spec is None or not op_spec.input_rank_is_exact:
        return []
    rank_out = len(nr.shape_out) if nr.shape_out is not None else None
    sig = signature_for(
        op_spec, _declared_rank_node(nr, rank_out or 0, [len(s) for s in shapes]),
        rank_out)
    if sig is None:
        return []
    for label in sorted(sig.consumed()):
        at = [(i, j) for i, j in sig.positions(label)
              if i < len(shapes) and j < len(shapes[i])]
        if len(at) < 2:
            continue
        (i0, j0), rest = at[0], at[1:]
        for i, j in rest:
            if _equal_dim(shapes[i0][j0], shapes[i][j]):
                continue
            return [Finding(
                kind="contracted_axis_mismatch", node=nr.name, op=nr.op,
                shapes=shapes,
                detail=f"{nr.op} contracts axis {label}: input{i0}[{j0}]="
                    f"{shapes[i0][j0]} vs input{i}[{j}]={shapes[i][j]} — the "
                    f"two must be equal. Usually an upstream head_broadcast / "
                    f"view_split is missing, or {label} is bound to a "
                    f"different dim on one side.",
            )]
    return []


def _check_node(resolved, nr, parent, issues) -> None:
    """Cross-input geometry rules for one node; recurses into components."""
    from backend.arch.ops import ShapeKind

    path = f"{parent}.{nr.name}" if parent else nr.name
    if nr.is_component or not nr.op:
        for c in (nr.children or []):
            _check_node(resolved, c, path, issues)
        return
    kind = _shape_kind_of(resolved, nr.op)
    if kind is None:
        return
    # Rank / dim agreement only matters cross-input. Single-input
    # ops can't have an inconsistency by themselves.
    shapes = _input_shapes_of(nr)
    if len(shapes) < 2:
        return

    # AXIS_CONCAT: non-axis dims must match (or broadcast) — catches
    # concatenated tensors from different head-rank paths (the MLA k_pe
    # broadcast bug). Everything else: the axes the op's own signature
    # contracts must agree across inputs — catches a K whose d_head is not
    # Q's (a k_eq_v path bypassing head_broadcast on DSv4-Pro), and the
    # same on every attention variant whose template is a contract.

    if kind is ShapeKind.AXIS_CONCAT:
        found = _check_axis_concat(nr, shapes)
    else:
        found = _check_contract_pair(nr, shapes)
    issues.extend(replace(issue, template=parent) for issue in found)


def _run_shape_consistency_check(resolved, block_results) -> list:
    """For every multi-input primitive op in every used block template,
    verify the upstream-derived input shapes are mutually consistent
    in the way the op needs:

      * `axis_concat`  — all inputs share rank; non-axis dims agree
                         pairwise (axis dim sums).
      * exact-template ops — every label the op's `shape_in` signature
                         contracts away (on two or more inputs) has one
                         extent: d_head between Q and K, s_kv between the
                         probabilities and V.
      * `preserve`-multi (add/mul) — all inputs share full shape.
      * `explicit` / `source` — no generic cross-input rule; the
        DimMap is the declaration.

    Returns a list of issue dicts. Empty list when geometry holds.
    Catches the MLA k_pe pattern (broadcasting forgotten → concat of
    [B,n_heads,s_ctx,d_head] with [B,s_ctx,d_qk_rope] mismatches
    rank/dim) and similar.
    """

    issues: list = []
    for tmpl_name, _count, br in block_results:
        for nr in br.nodes:
            _check_node(resolved, nr, parent=tmpl_name, issues=issues)
    return issues



# --- per-op DimMap consistency (intra-node, all ops) -----------------------
def _rank_node(out_rank, axes, axis, in_ranks) -> ShapeProbe:
    """A rank-only `ShapeNode` view for `axis_propagation.dim_map`: output rank,
    permute axes / replace axis, and input ranks (each input a rank-only src)."""
    return ShapeProbe(
        out_shape=tuple(range(out_rank)), axes=axes, axis=axis,
        inputs_obj=tuple(_Ref(src=ShapeProbe(out_shape=tuple(range(r))))
                         for r in in_ranks))


def _declared_rank_node(nr, out_rank, in_ranks) -> ShapeProbe:
    """`_rank_node` for a report's node. The permutation and the replace-axis
    are DECLARED, on the IR node the DTO already references — read the
    declaration, never reconstruct it from display labels (a concretized axis
    renders as its bare extent, and two same-sized axes made the
    reconstruction refuse: 260 transposes across the corpus lost their
    signature to a recovery of a fact that was never lost)."""
    ir = nr.ir_node
    axes = ir.axes if ir is not None else None
    if axes is None:
        axes = nr.axes
    axis = ir.axis if ir is not None else None
    if axis is None:
        axis = nr.axis
    return _rank_node(out_rank, axes, axis, in_ranks)


def run_dim_map_consistency_check(resolved, block_results) -> list:
    """For every primitive op, verify the per-op DimMap's size relations
    against the engine-resolved shapes — a `from` copies its input dim, a
    `merge` equals the product of its inputs, a `split`'s parts multiply back
    to the input dim. Generalises `_run_shape_consistency_check` (which only
    covers a few multi-input families' cross-input agreement) to EVERY mapped
    op's input→output correspondence — so a `view_split` on a flat dim that
    isn't n_heads·d_head, or a node whose resolved input shape doesn't fit the op (a
    mis-wired edge / missing reshape), surfaces at validate time. Positional
    (never reads an axis label), so a concretized head axis is checked the same.
    """
    from backend.arch.axis_propagation import (
        edge_consistency_errors,
        shape_consistency_errors,
    )

    issues: list = []

    def _ints(sh):
        """Every axis evaluated to a number (a symbol left behind is not one)."""
        return bool(sh) and all(isinstance(value_of(x), int) for x in sh)

    def _ins(nr):
        si = nr.shape_in
        if si is None:
            return []
        if si and isinstance(si[0], (list, tuple)):
            return [list(s) for s in si]
        return [list(si)]

    def _check(nr, parent):
        if nr.is_component or not nr.op:
            for c in (nr.children or []):
                _check(c, f"{parent}.{nr.name}" if parent else nr.name)
            return
        out = [int(x) for x in nr.shape_out] if nr.shape_out else None
        if not _ints(out):
            return
        ins = [[int(x) for x in s] for s in _ins(nr) if _ints(s)]
        if not ins:
            return
        node = _declared_rank_node(nr, len(out), [len(s) for s in ins])
        issues.extend(Finding(
            kind="dim_map_mismatch", template=parent, node=nr.name, op=nr.op,
            shapes=[*ins, out], detail=e,
        ) for e in shape_consistency_errors(nr.op, node, ins, out))
        issues.extend(Finding(
            kind="edge_rank_mismatch", template=parent, node=nr.name,
            op=nr.op, shapes=ins, detail=e,
        ) for e in edge_consistency_errors(nr.op, ins))

    for tmpl, _count, br in block_results:
        for nr in br.nodes:
            _check(nr, parent=tmpl)
    return issues
