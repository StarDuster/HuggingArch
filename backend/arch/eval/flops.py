"""FLOPs derived from the shape, instead of alongside it.

The disagreement
----------------
An op says two things about its geometry and says them separately. `shape_in`
declares the tensors it consumes — `("B", "n_heads", "s_query", "d_head")` — and
`flops_fn` declares its arithmetic — `2*B*n_heads*s_pairs*d_head`.
Both are written in dim symbols, and nothing makes them agree, because the
lambda reads those symbols from the GLOBAL dim binding while the shape is
whatever the upstream node actually produced.

Where the two diverge, the cost is wrong and nothing says so. Measured over the
shipped corpus, they diverge on 60-odd nodes:

  * `rope` on a `[B, n_heads, s_ctx, d_head]` tensor whose spec binds
    `dim: d_head` costs `3*B*s_ctx*d_head` — one head's worth, an
    `n_heads`-fold under-count, and `n_heads` is 8 to 64 across the corpus;
  * an expert-bank `silu` whose activation is `[M, 1, d_ffn_expert]` while
    `dim` is bound to the shared-expert width costs several times what it should;
  * `rmsnorm` used as a per-head q/k norm has the same shape as `rope` and the
    same under-count.

These are small terms — an elementwise op is a handful of FLOPs per element next
to a matmul's `2K` — but the class of error is not small: the same split is what
let a read path assume one dtype while the storage path used another for the
same tensor.

The fix, and why it needs no new notation
-----------------------------------------
`shape_in` is already an einsum signature: the labels ARE the dim symbol names,
and two inputs sharing `B`/`n_heads`/`d_head` already say those axes are the
same index. So the lambda does not have to change and no op has to be
re-declared. What changes is where its symbols get their VALUES: bound from the
actual input tensors, positionally, rather than looked up in the node's dim
environment.

    2*B*s_ctx*in*out   evaluated with B, s_ctx, in read off the tensor that
                       actually arrived, and only `out` — a label no input
                       carries — still coming from dims

Binding is trailing-aligned, and leading axes the template does not cover are
folded in as a batch factor. That is what makes one rank-3 template serve a
rank-4 tensor: every `flops_fn` in the registry is linear in its batch axes, so
an uncovered leading axis multiplies the cost exactly once. It is also what
fixes `rope` — the head axis is uncovered, so it multiplies, instead of being
silently absent.

Positional, never by name. A head axis concretized to `64` by a headwise gate
binds the same as one labelled `n_heads`, which is the same property that makes
the DimMap robust and the same reason neither may key on the symbol.

What a FLOP is
--------------
The coefficient in every `flops_fn` counts the scalar arithmetic of the op's
defining formula (its registry `formula`), per output element, summed over
the elements the shape says exist:

  * `+ − × ÷`, max / min, and a compare-select each count 1;
  * a fused multiply-add counts 2 — a matmul MAC is 2, as `linear` and
    `attention_score` write it;
  * one transcendental evaluation (`exp`, `log`, `tanh`, `erf`, `sqrt`,
    `rsqrt`, `sin`, `cos`) counts 1;
  * a reduction counts 1 per reduced element; a per-row scalar (one `rsqrt`
    per row) amortises to 0;
  * a value read from a table or a constant — a cos/sin cache, a norm gain,
    `scale_const`'s constant — is bytes, not FLOPs.

The count is the formula's, not a kernel's: a fused kernel's cost is the sum
of its parts (`Cost.combine`), and work one implementation adds on top (a
recomputed sin/cos, an extra scale) belongs to the measured track, not to the
speed-of-light figure. RoPE under this rule: `x' = x·cos − y·sin`,
`y' = y·cos + x·sin` is 4 multiplies and 2 adds per rotated pair, 3 per
rotated element (vLLM `csrc/libtorch_stable/pos_encoding_kernels.cu:31-32`).

Coefficients not yet re-derived under this rule are known debt, not a second
convention: `exp` (registry 2, rule 1), `neg_exp` (3, 2), `silu` (4, 5),
`softplus` (5, 3), `tanh` (5, 1), `gelu` (8, 5), `gelu_tanh` (8, 9), `situ`
(11, 8), `rmsnorm` (2, 4), `rmsnorm_unscaled` (2, 3), `l2norm` (2, 3),
`layernorm` (2, 7); and `clamp` counts 1 where a two-sided bound is 2.

What this does NOT do
---------------------
It does not touch `params`. A weight's size is a property of the module, not of
the activation passing through it, and binding it to an upstream extent would
make a batch-size change move the parameter count.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple, Optional

import sympy as sp

from .dim import (Dim, ReadTracking, expr_of, is_unit, same_law, value_of,
                  varies)
from .values import Extent


def _as_extent(value: Any) -> Any:
    """`Dim` when `value` is an axis length; otherwise leave it.

    Node dims also carry lists (a weight_shape, an axes vector). Those are
    not extents, and wrapping them would raise.
    """
    if isinstance(value, (list, tuple, dict)):
        return value
    try:
        return Dim.of(value)
    except TypeError:
        return value


def _any_input_voted(
    op: Any,
    in_shapes: Sequence[Sequence[Extent]],
) -> bool:
    """Whether any arriving tensor is tall enough for its `shape_in` template.

    A shorter tensor is not a contradiction: for most ops it does not vote at
    all, and for a separable op (`_separable`) it votes only for the trailing
    labels it carries and never for the batch remainder — so it cannot make
    remainders disagree, and it is not counted here. Two full-height voters
    with disagreeing remainders is the case `bind_labels` used to swallow.
    """
    for template, shape in zip(_templates(op, in_shapes), in_shapes, strict=False):
        if not template or not shape:
            continue
        if len(template) <= len(tuple(shape)):
            return True
    return False


def _separable(op: Any) -> bool:
    """An op whose cost is per element of what arrives: its output is its
    input's shape (`preserve`) or their broadcast (`broadcast`) and no axis is
    coupled (`couples_on` empty). Such an op on a lower-rank tensor — `neg_exp`
    on a per-head `A_log` — costs that tensor's elements, not a batch-times-
    sequence the tensor does not have."""
    return op.shape_kind in ("preserve", "broadcast") and not op.couples_on


def _templates(op: Any, in_shapes: Sequence[Any] = ()
               ) -> Tuple[Tuple[str, ...], ...]:
    """`shape_in` as one template per input port.

    The registry writes a single-input template flat (`("B", "s_ctx", "in")`)
    and a multi-input one nested, so a consumer that does not normalise sees a
    3-input op where there is a rank-3 tensor. A join's templates follow the
    axis the node joins along (`OpRef.input_templates`), one per arriving
    input.
    """
    from .op_ref import OpRef

    if isinstance(op, OpRef) and in_shapes:
        joined = op.input_templates(in_shapes)
        if joined:
            return joined
    raw = op.shape_in_template if isinstance(op, OpRef) else (op.shape_in or ())
    if not raw:
        return ()
    if all(isinstance(t, (tuple, list)) for t in raw):
        return tuple(tuple(str(x) for x in t) for t in raw)
    return (tuple(str(x) for x in raw),)


def _agreement(left: Extent, right: Extent) -> str:
    """How two extents claiming one label relate: `"same"`, `"differ"`, or
    `"unstated"`.

    They are the same length when their VALUES agree — a `Dim` and the `int`
    it carries are one extent, and comparing the objects made that a conflict.
    They differ only when both state a length and the lengths disagree; that
    is the contradiction the strict rule refuses, because two concrete kernel
    buffers cannot both be right about one axis.

    An extent with no number states no length. In the symbolic domain every
    extent is an expression, so `d_lin_k_head` meeting `d_lin_v_head` under
    one label is two laws, not two lengths, and there is nothing to refuse:
    the label is dropped and the environment answers, as it did before the
    strict rule existed. One law under one label is one extent (`same_law`).
    """
    if same_law(left, right):
        return "same"
    lv, rv = _length(left), _length(right)
    if lv is None or rv is None:
        return "unstated"
    return "same" if lv == rv else "differ"


def _length(extent: Extent) -> Any:
    """The extent's number of positions, or `None` when it is an expression
    rather than a number.

    The question is asked of the VALUE, not the extent: a tracked `Dim` in the
    numeric domain carries an axis name beside a number and is perfectly
    concrete, while a symbolic value is a law for a length — the strict rule
    is about two concrete kernel buffers, and a law states none.
    """
    value = value_of(extent)
    return None if varies(value) else value


def _remainder_key(remainder: Tuple[Extent, ...]) -> tuple:
    """Remainders compare by value where they have one, so an `int` axis and a
    `Dim` carrying the same value are the same remainder."""
    return tuple(
        value_of(axis) if value_of(axis) is not None else str(axis)
        for axis in remainder)


def bind_labels(
    op: Any,
    in_shapes: Sequence[Sequence[Extent]],
    *,
    require_consistent_labels: bool = True,
    remainder_out: Optional[List[Any]] = None,
) -> Tuple[Dict[str, Extent], Extent]:
    """Extents for the labels `shape_in` names, plus the uncovered batch factor.

    Returns `({label: extent}, batch)`. A template shorter than the tensor it
    describes aligns to the TRAILING axes — the feature end is what an op's
    formula is written about — and the leading remainder becomes `batch`, the
    product of the axes no label covers.

    A label appearing twice with different extents is a conflict and raises.
    The walk and the per-rank projection read the same shapes, so they apply
    the same rule; when the walk alone dropped the label and fell back to the
    dim environment, the two could price one node differently and only the
    projection would say so. ``require_consistent_labels=False`` keeps the
    lenient reading for a caller that states it wants the environment's number.

    The batch factor is only taken when every input that the template fits
    leaves the SAME remainder. Disagreement means the declaration does not
    describe these tensors — a rank-3 template fed a rank-4 tensor — and one
    input's leading axis would
    otherwise be folded in as everyone's batch, tripling the op. A template that
    does not describe the tensors is not evidence about them, so nothing is
    derived and the environment's binding stands.
    """
    labels: Dict[str, Extent] = {}
    #: Both sides state a length and the lengths disagree — refused as a
    #: contradiction under the strict rule.
    contradictions: List[Tuple[str, Extent, Extent]] = []
    #: One side states no length (the symbolic domain). Not a contradiction,
    #: but not an answer either: the label is dropped either way.
    unstated: List[str] = []
    remainders: List[Tuple[Extent, ...]] = []
    is_broadcast = op.shape_kind == "broadcast"
    separable = _separable(op)

    for template, shape in zip(_templates(op, in_shapes), in_shapes, strict=False):
        if shape is None:
            continue
        shape = tuple(shape)
        broadcast_operand = bool(
            separable and template and shape and len(template) > len(shape))
        if broadcast_operand:
            # A separable op aligns a lower-rank tensor from the trailing
            # axis, as its shape rule broadcasts it; the leading labels are
            # axes this tensor does not have. It votes for the labels it
            # carries and says nothing about the batch remainder.
            template = template[len(template) - len(shape):]
        if not template or len(template) > len(shape):
            # The template describes a rank the tensor does not have — a
            # broadcast operand, or a mis-declared port. Aligning the overlap
            # would pair the wrong axes, so this input does not vote.
            continue
        offset = len(shape) - len(template)
        if not broadcast_operand:
            remainders.append(tuple(shape[:offset]))
        for label, extent in zip(template, shape[offset:], strict=False):
            if not label.isidentifier():        # a literal axis, e.g. "1"
                continue
            if label in labels:
                agreement = _agreement(labels[label], extent)
                if agreement != "same":
                    if is_broadcast and is_unit(labels[label]):
                        labels[label] = extent      # the unit side broadcasts
                        continue
                    if is_broadcast and is_unit(extent):
                        continue
                    if agreement == "differ":
                        contradictions.append((label, labels[label], extent))
                    else:
                        unstated.append(label)
            labels[label] = extent
    # Inconsistent remainders mean the declaration does not describe these
    # tensors at all, and a label clash under a misaligned template is a
    # symptom of that, not a second fact. Decided before the conflicts are.
    if not remainders or len({_remainder_key(r) for r in remainders}) > 1:
        return ({} if remainders else labels), 1
    if contradictions and require_consistent_labels:
        label, first, second = contradictions[0]
        raise ValueError(
            f"input label {label!r} has conflicting extents "
            f"{first!r} and {second!r}"
        )
    for label, _first, _second in contradictions:
        labels.pop(label, None)
    for label in unstated:
        labels.pop(label, None)
    batch: Extent = 1
    for extent in remainders[0]:
        batch = batch * extent
    if remainder_out is not None:
        # The axes whose product is the batch factor — recorded so a consumer
        # displaying the resulting formula can bind their identities, not just
        # the folded product.
        remainder_out.extend(remainders[0])
    return labels, batch


def record_binding(record: Dict[Any, Any], extent: Extent, *,
                   overwrite: bool = False) -> None:
    """One entry of a formula's substitution record: identity → value used.

    The key is the extent's IDENTITY (`expr_of` — a bare symbol like
    `n_heads` or a composite like `n_heads*d_head`), because that is the
    vocabulary the formula expression is written in. Two extents claiming one
    identity at different values drop the entry: a binding that might belong
    to either scope is no binding.

    `overwrite` is the precedence rule between the two tiers of one
    evaluation: the dim environment is recorded with it off, the labels and
    batch axes — the extents measured off the tensors that actually arrived —
    with it on, mirroring the `{**dims, **labels}` override the evaluation
    itself applies. Under CP the env's `s_ctx` is the shard while the query
    axis still carries the full length; the axis is what the formula read.
    """
    key = expr_of(extent)
    if not isinstance(key, sp.Basic) or not key.free_symbols:
        return
    if any(symbol.is_integer and symbol.is_nonnegative
           for symbol in key.free_symbols):
        # An untracked extent's "identity" is its value: a law in the
        # workload's counts (`Workload.symbolic`) names no axis, as a bare
        # number names none.
        return
    value = value_of(extent)
    if key in record:
        # One identity bound to two laws is ambiguous for the formula that
        # reads it, whatever the two come to at a point (`same_law`).
        if same_law(record[key], value):
            return
        if overwrite:
            record[key] = value
        else:
            del record[key]
        return
    record[key] = value


def merge_bindings(parts: Sequence[Dict[Any, Any]]) -> Dict[Any, Any]:
    """Union of substitution records, dropping keys the parts disagree on.

    The `Layer.rebase` combiner for binding records: a fused node inherits
    every absorbed sibling's record, and a sibling pair that bound one
    identity at two values keeps neither.
    """
    seen: Dict[Any, set] = {}
    for part in parts:
        for key, value in part.items():
            seen.setdefault(key, set()).add(value)
    return {key: next(iter(values))
            for key, values in seen.items() if len(values) == 1}


def shape_derived_dims(
    op: Any,
    dims: Mapping[str, Any],
    in_shapes: Sequence[Sequence[Extent]],
    *,
    require_consistent_labels: bool = True,
) -> Tuple[Dict[str, Any], Extent]:
    """`dims` with every label the input tensors determine overwritten by what
    they actually carry, and the batch factor the templates left uncovered.

    Only labels the templates NAME are overwritten. A symbol the op reads but no
    input carries — `out` on a projection, `s_window` on a sliding-window
    attention — is not something a tensor can answer, and keeps the value the
    environment bound.
    """
    labels, batch = bind_labels(
        op,
        in_shapes,
        require_consistent_labels=require_consistent_labels,
    )
    if not labels:
        return dict(dims), batch
    return {**dims, **labels}, batch


def flops_over_operands(op: Any, dims: Mapping[str, Any],
                        in_shapes: Sequence[Sequence[Extent]], evaluate,
                        **kwargs: Any) -> Extent:
    """`flops_from_shape`, where an op that repeats its last input (a row
    scatter writing several media inputs' rows,
    `OpSpec.repeats_last_input`) costs its call on each repeated operand."""
    arity = op.arity()
    if not (op.repeats_last_input and len(in_shapes) > arity):
        return flops_from_shape(op, dims, in_shapes, evaluate, **kwargs)
    total: Extent = 0
    for operand in in_shapes[arity - 1:]:
        total = total + flops_from_shape(
            op, dims, (*in_shapes[:arity - 1], operand), evaluate, **kwargs)
    return total


def flops_from_shape(
    op: Any,
    dims: Mapping[str, Any],
    in_shapes: Sequence[Sequence[Extent]],
    evaluate,
    *,
    require_consistent_labels: bool = True,
    record: Optional[Dict[Any, Any]] = None,
    env_guard: Optional[Any] = None,
) -> Extent:
    """`evaluate(dims)` against shape-derived extents, times the batch factor.

    `evaluate` is the op's own cost — `OpRef.flops` — so this changes the inputs
    to the existing formula and not the formula. An op whose cost is zero stays
    zero: the batch factor multiplies, and multiplying zero is zero.

    `record`, when given, collects the substitution THIS evaluation used —
    the dims the formula actually read (tracked through the merged map it was
    handed), plus every batch-factor axis, keyed by identity (`expr_of`) — so
    a formula display binds the values the engine substituted instead of
    reconstructing them afterwards. A dim that sat in the environment unread
    is not part of the evaluation and does not enter the record.

    `env_guard`, when given, is called as `env_guard(name, extent)` for each
    read dim that came from the environment rather than from an arriving
    tensor — the tier where a stale global binding can otherwise pass for a
    per-rank value. The guard's refusal policy is the caller's; this function
    only knows which tier each read came from.
    """
    remainder_axes: Optional[List[Any]] = [] if record is not None else None
    labels, batch = bind_labels(
        op,
        in_shapes,
        require_consistent_labels=require_consistent_labels,
        remainder_out=remainder_axes,
    )
    # An input whose template is taller than the tensor does not vote for the
    # batch remainder: a separable op's lower-rank input votes only for the
    # trailing labels it carries, any other op's does not vote at all. Empty
    # labels then means "no evidence", not a broken declaration. Full-height
    # voters that still produced no labels means the templates do not
    # describe these tensors together — that used to fall back to Binding.
    if (
        not labels
        and require_consistent_labels
        and _any_input_voted(op, in_shapes)
    ):
        raise ValueError(
            f"shape_in {_templates(op)!r} does not describe the arriving "
            f"tensors {in_shapes!r}"
        )
    if labels and _separable(op):
        # A template label no arriving tensor carries is an axis the op does
        # not run over — one, not the environment's global extent.
        for template in _templates(op):
            for label in template:
                if label.isidentifier() and label not in labels:
                    labels[label] = 1
    merged = dict(dims) if not labels else {**dims, **labels}
    # A lambda that writes `2 * d["B"] * d["s_ctx"] * ...` puts a sympy
    # Symbol on the left whenever the dim environment holds symbols. Sympy
    # then converts a `Dim` through `_sympy_`, which is the identity
    # half — so a unit axis `Dim(x, 1)` leaked into the value
    # that formulas substitute. Wrapping every operand keeps arithmetic
    # on `Dim`, where the value half folds the 1.
    derived = {name: _as_extent(value) for name, value in merged.items()}
    if record is None:
        return _times_batch(evaluate(derived), batch)
    tracked = ReadTracking(derived)
    cost = evaluate(tracked)
    # Sorted for determinism: `read` is a set, and `record_binding`'s
    # drop-on-conflict semantics are order-sensitive for three or more
    # disagreeing entries.
    for name in sorted(tracked.read):
        if name not in merged:
            continue
        if env_guard is not None and name not in labels:
            env_guard(name, merged[name])
        record_binding(record, merged[name])
    for extent in (*labels.values(), *(remainder_axes or ())):
        record_binding(record, extent, overwrite=True)
    return _times_batch(cost, batch)


def _times_batch(cost: Extent, batch: Extent) -> Extent:
    """The cost times the batch factor. A tracked factor multiplies even at
    value 1: `B` bound to 1 is still the axis the cost grows along, and
    dropping it would leave the cost an expression free of `B`."""
    return cost * batch if isinstance(batch, Dim) or batch != 1 else cost
