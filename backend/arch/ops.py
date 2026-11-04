"""Op library — query functions over the registered `OpSpec` table.

The base classes and the table itself live in `ops_base`; this module holds
the consumer-facing predicates that read `OP_REGISTRY`, and imports
`ops_registry` (last) to fill it. Splitting the base out is what lets
`ops_registry` import `OpSpec` / `register_op` without closing an
`ops -> ops_registry -> ops` cycle.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, Optional, Tuple

from .ops_base import *  # noqa: F403 — re-export the op library base
from .ops_base import Lookup, OP_REGISTRY, OpSpec, ShapeKind  # read by the query functions below


def get_op(name: str) -> OpSpec:
    if name not in OP_REGISTRY:
        raise KeyError(f"Unknown primitive op: {name!r}. Available: {sorted(OP_REGISTRY)}")
    return OP_REGISTRY[name]


def is_primitive(name: str) -> bool:
    return name in OP_REGISTRY


def is_identity_source(name: str) -> bool:
    """`OpSpec.is_identity_source` by op name; False for a name the registry
    does not know (a component, a custom op).

    Four readers ask this — traffic (who is billed for the bytes), the
    evaluator (what reaches a node), the layout rule (what state arrives),
    kernel_bench (whose dtype a GEMM's weight is) — and each used to carry
    its own three-line copy behind a function-body import of this module.
    One question, one place.
    """
    spec = OP_REGISTRY.get(name)
    return bool(spec is not None and spec.is_identity_source())


def binding_dims(name: str) -> Tuple[str, ...]:
    """`OpSpec.binding_dims` by op name: the model-scope dims the op reads
    off its binding. Empty for a name the registry does not know."""
    spec = OP_REGISTRY.get(name)
    return spec.binding_dims if spec is not None else ()


def passes_a_lone_operand(name: str) -> bool:
    """`OpSpec.passes_a_lone_operand` by op name; False for a name the
    registry does not know (a component, a custom op)."""
    spec = OP_REGISTRY.get(name)
    return bool(spec is not None and spec.passes_a_lone_operand)


def reads_own_output(name: str) -> bool:
    """`OpSpec.reads_own_output` by op name; False for a name the registry
    does not know (a component, a custom op)."""
    spec = OP_REGISTRY.get(name)
    return bool(spec is not None and spec.reads_own_output)



def gather_source(name: str) -> Optional[int]:
    """`OpSpec.gather_source` by op name: the input port the op's output is
    an index gather of. None for a name the registry does not know."""
    spec = OP_REGISTRY.get(name)
    return None if spec is None else spec.gather_source


def repeats_input(name: str) -> bool:
    """`OpSpec.repeats_input` by op name: a view that repeats its input's
    elements. False for a name the registry does not know."""
    spec = OP_REGISTRY.get(name)
    return bool(spec is not None and spec.repeats_input)


def reads_input_shape_only(name: str) -> bool:
    """`OpSpec.reads_input_shape_only` by op name; False for a name the
    registry does not know (a component, a custom op)."""
    spec = OP_REGISTRY.get(name)
    return bool(spec is not None and spec.reads_input_shape_only)


# The registration table lives in ops_registry.py; importing it here (last)
# is what fills OP_REGISTRY on `import backend.arch.ops`.
import backend.arch.ops_registry as _ops_registry  # noqa: F401


def is_tensor_source(name: str) -> bool:
    """`OpSpec.is_tensor_source` by op name; False for a name that is not a
    registered primitive (a component or a custom op)."""
    spec = OP_REGISTRY.get(name)
    return spec is not None and spec.is_tensor_source()


def is_gemm_op(name: str) -> bool:
    """`OpSpec.gemm` by op name; False for anything not registered. A node
    whose declaration may be a spec's own custom op asks
    `ops_base.declaration_is_gemm`, which this is the primitive half of."""
    spec = OP_REGISTRY.get(name)
    return spec is not None and spec.gemm


def is_comm_op(name: str) -> bool:
    """`OpSpec.is_comm_op` by op name; False for anything not registered."""
    spec = OP_REGISTRY.get(name)
    return spec is not None and spec.is_comm_op


def attention_kernel(name: Optional[str]) -> Optional[str]:
    """`OpSpec.attention_kernel` by op name: `"score"`, `"apply"`, or None
    for any op that is neither half of an attention kernel (and for a name
    that is not registered)."""
    spec = OP_REGISTRY.get(name or "")
    return spec.attention_kernel if spec is not None else None


def is_dsa_op(name: str) -> bool:
    """`OpSpec.dsa` by op name; False for anything not registered."""
    spec = OP_REGISTRY.get(name)
    return spec is not None and spec.dsa


def rotary_ops() -> FrozenSet[str]:
    """Every op that rotates by position (`OpSpec.rotary_position_axes`)."""
    return frozenset(name for name, spec in OP_REGISTRY.items()
                     if spec.rotary_position_axes)


def hosts_views(ops: Iterable[str]) -> bool:
    """Whether one kernel running `ops` takes an adjacent zero-cost view (a
    reshape, a transpose, a head split) into its own prologue or epilogue,
    as vLLM's elementwise, normalisation and rotation kernels do.

    Derived from the declarations: at least one op computes in the layout
    it reads (not a view, a GEMM, an attention kernel or a collective, and
    its output keeps its input's axes: `preserve`, `broadcast`,
    `axis_concat`), and none is a GEMM or an attention kernel, which read
    prepared tensors through strides. A primitive asks with its own name;
    a fused kernel with the ops its pattern matches
    (`NaivePattern.ops`)."""
    specs = [OP_REGISTRY[op] for op in ops if op in OP_REGISTRY]
    if any(spec.gemm or spec.attention_kernel for spec in specs):
        return False
    return any(
        not (spec.is_view_op or spec.is_comm_op)
        and spec.shape_kind in (ShapeKind.PRESERVE, ShapeKind.BROADCAST,
                                ShapeKind.AXIS_CONCAT)
        for spec in specs)


def op_labels() -> Dict[str, str]:
    """Display name per op, for the ops whose author gave one."""
    return {name: spec.label for name, spec in OP_REGISTRY.items() if spec.label}


def lookup_of(decl) -> Optional[Lookup]:
    """The row gather a leaf's declaration makes (`OpSpec.lookup`): where
    its ids and its table are. None for None, for a custom op, and for every
    op that gathers from no table."""
    return decl.lookup if isinstance(decl, OpSpec) else None


def owns_lookup_table(decl) -> bool:
    """A lookup into its own stored table — an embedding table, every one
    (the main one, MTP's, a speech bank), not only the input boundary."""
    lookup = lookup_of(decl)
    return lookup is not None and lookup.owns_table


def is_out_in_matrix(decl) -> bool:
    """A declaration whose weight is `[out, in]` — what a `W: in → out`
    detail describes. False for None and for a declaration without a
    weight."""
    return decl is not None and tuple(decl.weight_shape or ()) == ("out", "in")


def is_side_buffer(decl) -> bool:
    """A declaration that branches off the backbone — a precomputed position
    table or RoPE buffer the top-level graph wires as a standalone source,
    never the residual stream. False for None and for a custom op (a
    model-root custom op is a legal construct without this field)."""
    return isinstance(decl, OpSpec) and decl.is_side_buffer


def is_dispatch_boundary(decl) -> bool:
    """A declaration that is a dispatch-zone boundary — the `permute` /
    `combine` around a MoE expert bank. False for None and for a custom op."""
    return isinstance(decl, OpSpec) and decl.is_dispatch_boundary


def sp_axis_effect(name: str) -> str:
    """What a collective does to the sequence-parallel axis: `"divide"`,
    `"multiply"` or `""` (also `""` for a name that is not a primitive)."""
    return OP_REGISTRY[name].sp_axis_effect if name in OP_REGISTRY else ""


def collective_schedule(name: str) -> str:
    """The cross-node time schedule a collective may use: `"hierarchical"`,
    `"point_to_point"` (one hop between two members) or `""` (also `""`
    for a name that is not a primitive)."""
    return OP_REGISTRY[name].collective_schedule if name in OP_REGISTRY else ""
