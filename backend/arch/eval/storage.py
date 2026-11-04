"""`StoredTensor` — what a node puts on disk, asked of the op that owns it.

The question storage was never asked
------------------------------------
An op declares what it computes (`shape_kind`, `out_key`), what it costs
(`flops`, `params`) and — for non-linear weight geometry — what it stores
(`weight_shape`). What no op declares
is which of its dims are its WEIGHT's two dims, and every consumer that needs
that has had to work it out for itself:

  * `spec_tree` used to guess by symbol name, with the pair for one op
    hardcoded — `in_key = "n_indexer_heads" if out_key == "Out" else "in"`.
    A second fused op with different dim names needed a second line there.
    That guess is gone: a weight-bearing custom op declares `weight_shape`,
    or it uses the linear `(out, in)` default, or it raises.
  * the quant-metadata cardinality formulas are written in `in` and `out`, and
    resolving them against whatever is bound under those names in the
    surrounding scope reads another tensor's dims. It resolves happily: gpt-oss
    came out 8600x under with nothing saying so.
  * `lt.op == "linear"` is compared as a string in `sharding` and
    `weight_breakdown` to decide whether a tensor is a shardable matrix,
    which is why `spec_tree` emits `op="linear"` for a custom op that is not one.

Declared, not recovered
-----------------------
`OpSpec.tensors` lists every tensor an op owns, in checkpoint order, each with
its own shape and the dim that gates its existence. That is the whole mechanism.

It replaces two pieces of arithmetic that stood in for it, and both were the
same mistake in different places: recovering a declared fact from a scalar.

    in   = params / out                     the weight's input extent
    bias = params - elements(weight_shape)  everything else the op owns

The first ran 4 times in 1,618 across the corpus — `weight_shape` already
answered the other 1,614 — and those 4 were one custom op that had simply never
declared its `weight_shape`. The fix was that declaration, not a fallback: a
fallback keeps a missing declaration working, so nothing ever makes it missed.

The second is worse than a fallback. A subtraction has no name for what it
finds, so anything an op owns beyond its weight lands in one bucket labelled
`.bias` — it does not model the second tensor, it hides the difference. Which
axis a bias follows is genuinely per-op (`linear` biases the first axis of
`(out, in)`, `headwise_linear` the first two, `conv1d` the first of
`(C_out, C_in, K)`), and having no positional rule is the argument for declaring
it, not for subtracting.

Where an op declares `tensors`, its parameter count is the PROJECTION of that
list and not a second declaration beside it. Two independent declarations is
exactly how a bias came to be counted in `params` and absent from storage at the
same time, in eleven of thirty-nine models.

How far that goes, stated as a number rather than as an aspiration
-----------------------------------------------------------------
Two ops declare `tensors` today: `linear` and `layernorm`. Everything else still
answers `params` from its own `params_fn` and its shape from `weight_shape` —
two declarations, unchecked against each other, 3,649 times across the corpus
against 1,618 through the projection.

So the disagreement is closed for the ops that carry a bias, which is where it
had actually opened, and it remains POSSIBLE everywhere else. That is a coverage
number and a to-do, not a design: the end state is every weight-bearing op
declaring its tensors, at which point `params_fn` has nothing left to say and
`weight_shape` is the one-tensor shorthand rather than a parallel channel.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Tuple

from .. import dims as dim_registry
from ..ops_base import weight_leading_axes
from ..spec_tree.projections import compose_module
from .dim import Dim, InexactDivision, names_of
from .values import Extent, product


class StorageKind(Enum):
    """What KIND of thing is stored, as distinct from which op stored it.

    The distinction exists because three modules compare `leaf.op == "linear"`
    to answer a question about the tensor rather than about the op: is this a
    matrix that can be sharded along either axis, a per-channel vector that can
    only be split with its channels, or a lookup table indexed by token. An op
    name is a poor proxy — it forced `spec_tree` to emit `op="linear"` for a
    fused MXFP4 op so that sharding would keep working — and it is a closed
    world, so a new weight-bearing op is invisible to all three until each is
    edited.
    """

    MATRIX = "matrix"          # (out, in) — shardable on either axis
    HEADWISE = "headwise"      # (heads, out, in) — one matrix per head
    VECTOR = "vector"          # (d,) — per-channel, follows its channels
    TABLE = "table"            # (vocab, d) — indexed by token, not contracted
    BUFFER = "buffer"          # precomputed cache: occupies bytes, not learnable
    METADATA = "metadata"      # a scale / zero point a FORMAT implies, not an op

    @classmethod
    def of(cls, shape: Tuple[int, ...], op: Any = None) -> "StorageKind":
        """Inferred from rank and from what the op declares about the weight.

        Rank is not a heuristic here: a 1-D weight has no contraction to shard
        and a 3-D one is a stack of matrices. The only 2-D case that is not a
        contraction matrix is a lookup table, and what distinguishes it is that
        a reader INDEXES it rather than contracting against it — a fact about
        the graph, not about the op storing it: an embedding indexes its own
        weight, DeepSeek-V4 `hash_route` indexes a `parameter_tensor` handed
        to it. `Program.as_stored` marks every gathered tensor TABLE from
        `Program.lookups`; a bare `parameter_tensor` nobody indexes keeps the
        kind its rank says it is.
        """
        # Declared, and it outranks the rank: a RoPE inverse-frequency table is
        # a real tensor with real bytes that is not a parameter. Putting it on
        # the KIND rather than in a `Cost.buffer_params` field is the point —
        # `Cost` was about to grow a bucket per reason-for-exclusion, which is
        # the `aux_params` / `attention_params` mistake a group-by already
        # replaced twice in this package.
        if op is not None and op.is_buffer:
            return cls.BUFFER
        rank = len(shape)
        if rank >= 3:
            return cls.HEADWISE
        if rank == 2:
            return cls.MATRIX
        return cls.VECTOR

    @property
    def stores_out_in(self) -> bool:
        """Whether a 2-D tensor of this kind is `(out, in)` as a format reads it:
        rows first, each row a run of `in` contiguous elements.

        A contraction matrix is, and so is a lookup table — `(rows, width)`,
        where a lookup gathers whole rows. That is the geometry a block or a
        group tiles, so a quant scheme describes both with one formula: an FP8
        table with one scale per 32 elements of a row is `block_h: 1,
        block_w: 32`, and the scale grid is `[rows, width / 32]`, the same
        rows the table has (DeepSeek-V4.1 `ParallelEngramEmbedding`).
        """
        return self in (StorageKind.MATRIX, StorageKind.TABLE)


@dataclass(frozen=True)
class StoredTensor:
    """One tensor a node owns on disk.

    `shape` is the LOGICAL shape — elements as the architecture sees them, not
    as the checkpoint packs them. A packed MXFP4 weight is `(out, in)` here and
    becomes `in*out/2` bytes through its `TensorFormat`, which is the one place
    packing belongs.

    The name is a MODULE and a SUFFIX, not a string
    -----------------------------------------------
    A checkpoint entry is "this module, this role inside it": `q_proj` +
    `.weight`, `q_proj` + `.bias`, `q_proj` + `.weight_scale_inv`. Holding the
    two halves makes the sibling relation structural — same module, different
    suffix — where a single string makes it string surgery, and string surgery
    is what there were FIVE implementations of: `storage._sibling`, a regex in
    the checkpoint test, `evaluator._named_as_stored`, and two in the cutover
    dump scripts. Two of the five were wrong in opposite directions, and both
    were wrong about the same thing: what a suffix replaces.

    It also removes a trap. `.weight` is what the OP declares, and a quant
    scheme may say the bytes on disk are `.weight_packed` instead — so a name
    composed at construction was incomplete until something else fixed it up,
    and two call sites read it before anything did. With the suffix a field, the
    scheme's answer is `replace(t, suffix=...)` and there is no window in which
    `name` is wrong.
    """

    module: str
    suffix: str = ".weight"
    shape: Tuple[Extent, ...] = ()
    kind: StorageKind = StorageKind.MATRIX
    #: For a tensor that carries its own — a scale is F32 whatever the weight
    #: it belongs to is stored as. `None` means "ask the format".
    dtype: Optional[str] = None
    #: What the slot holds when that is not the slot type (a UE8M0 exponent in
    #: a U8). `None` means the slot type is the type. Only metadata siblings
    #: set it; a weight's logical type is the format's business.
    logical_dtype: Optional[str] = None
    #: What the OP called it, kept when a scheme renames the bytes.
    #
    # Two facts, not one: `.weight` is the module's tensor and
    # `.weight_packed` is the encoding a scheme stores it as, and consumers
    # need both — a byte total keys on what is on disk, while a lookup that has
    # to survive the rename keys on what the op declared. `weight_breakdown`
    # expresses the same pair as `name` and `source_logical_name`; keeping it
    # as two suffixes on one module says it without a second string to compose.
    logical_suffix: str = ".weight"
    #: Projection onto the stored weight's axes.  Declared by ``TensorDecl``
    #: for siblings such as bias; ``None`` is identity for the stored tensor.
    axis_refs: Optional[Tuple[int, ...]] = None
    #: For a METADATA tensor, what the scheme declared it to be — the
    #: `kind:` of the `metadata_siblings` entry it was made from (`scale`,
    #: `zero_point`, ...). A reader that wants "the scale" asks this, not
    #: the suffix.
    sibling_kind: Optional[str] = None
    # How many copies of it the model holds is NOT a field here. A tensor is
    # what one copy is, the count belongs to the node that owns it, and the walk
    # resolves that count against the binding it ran at — so carrying it on the
    # tensor too would be the same number arriving twice, free to disagree the
    # moment one of the two is per-rank. `Analysis.tensors()` pairs them.

    @property
    def name(self) -> str:
        """The on-disk name, and it is one from the moment this exists."""
        return f"{self.module}{self.suffix}"

    @property
    def logical_name(self) -> str:
        """The name before any scheme renamed the bytes — what the op declared.

        Equal to `name` for everything a scheme leaves alone, which is most
        tensors; they differ exactly where a packed format is in play.
        """
        return f"{self.module}{self.logical_suffix}"

    def sibling(self, suffix: str, shape: Tuple[Extent, ...],
                kind: "StorageKind", dtype: Optional[str] = None,
                logical_dtype: Optional[str] = None,
                sibling_kind: Optional[str] = None,
                ) -> "StoredTensor":
        """Another tensor of the same module — a bias, a scale, a zero point.

        The only way to make one, so "what does a suffix replace" is asked
        once and cannot be answered differently in the next module. It replaces
        nothing: the module is the module, and the suffix says which of its
        tensors this is.
        """
        return StoredTensor(module=self.module, suffix=suffix,
                            logical_suffix=suffix, shape=shape,
                            kind=kind, dtype=dtype, logical_dtype=logical_dtype,
                            sibling_kind=sibling_kind)

    @property
    def is_parameter(self) -> bool:
        """Whether the checkpoint's parameter total counts it.

        A buffer does not — and it still occupies every byte it occupies.
        Neither does quantisation metadata, for the same reason and a different
        one: the scales are real bytes on the device that the model card's
        parameter count excludes.
        """
        return self.kind not in (StorageKind.BUFFER, StorageKind.METADATA)

    @property
    def elements(self) -> Extent:
        return product(self.shape)

    def bytes(self) -> Extent:
        """Only for a tensor carrying its OWN dtype — a scale, a zero point.

        A weight's width is its format's answer and is asked there; raising
        here rather than defaulting is the same rule `dtypes` follows, and for
        the same reason: a tensor priced at a guessed width is a device budget
        that looks right until the load fails.
        """
        from ..dtypes import dtype_bytes
        if self.dtype is None:
            raise ValueError(
                f"{self.name} carries no dtype of its own — ask its "
                f"`TensorFormat`, which is where a weight's width lives")
        return self.elements * dtype_bytes(self.dtype)

    @property
    def matrix_dims(self) -> Optional[Tuple[Extent, Extent]]:
        """`(in, out)` for a contraction matrix or a lookup table's
        `(width, rows)`, else None.

        This is what a quant scheme's cardinality formula is written in, and
        having it come from the stored shape is the whole point: `in` and `out`
        are then the weight's own dims by construction, not two names looked up
        in an enclosing scope that may be describing something else.
        """
        if self.kind.stores_out_in and len(self.shape) >= 2:
            # The matrix is the last two axes; any axis before them is a
            # batch axis (`torch.matmul`'s rule), one matrix per index.
            return self.shape[-1], self.shape[-2]
        if self.kind is StorageKind.HEADWISE and len(self.shape) >= 2:
            # The checkpoint stores a per-head bank FLATTENED — every leading
            # axis folded into the rows ([n_heads * out, in] on disk), and a
            # scale layout tiles that stored 2D geometry, not one head's
            # slice. Dropping the leading axes under-counted a grouped
            # output-LoRA's tile scales 16x, and `in * out` then also failed
            # to multiply back to `params` — the two dims must be the stored
            # matrix's by construction, all of it.
            rows = self.shape[-2]
            for axis in self.shape[:-2]:
                rows = rows * axis
            return self.shape[-1], rows
        return None

    def as_dims(self) -> dict:
        """The `{in, out}` binding a cardinality formula needs, or empty."""
        pair = self.matrix_dims
        return {} if pair is None else {"in": pair[0], "out": pair[1]}


def is_expert_axis(extent: Extent) -> bool:
    """Whether an axis is an expert axis: its identity names a symbol
    `dims.yaml` classifies `kind: expert_copies` (`n_expert_copies`, or a
    pool's own copies symbol). Asked of the axis's `Dim` identity, never of
    its position: a weight is an expert bank because its declared `shape:`
    leads with that symbol, and an activation is routed because the dispatch
    declared it so (`permute`'s `[n_expert_copies, M, d_model]`)."""
    return any((declared := dim_registry.get(name)) is not None
               and declared.kind == "expert_copies"
               for name in names_of(extent))


def leads_with_expert_axis(tensor: "StoredTensor") -> bool:
    """Whether a stored tensor's axis 0 is an expert axis (`is_expert_axis`)."""
    return bool(tensor.shape) and is_expert_axis(tensor.shape[0])


def module_path(ir: Any, prefix: str) -> str:
    """The checkpoint path of the MODULE a node's tensors live in.

    No suffix: which tensor of that module this is belongs to `StoredTensor`,
    and the op's `.weight`-or-nothing is one such answer rather than part of the
    path. Composing them here is what made a name that a quant scheme then had
    to correct.
    """
    return compose_module(ir, prefix)


def from_declaration(op: Any, dims: Mapping[str, Any], params: Extent,
                     module: str, resolve_expr) -> Tuple[StoredTensor, ...]:
    """What this op stores, from what it already declares.

    PLURAL, and that is the correction. A `linear` with a bias puts two tensors
    on disk under two names, and the checkpoint lists them separately — so
    answering with one `StoredTensor` leaves the bias with no bytes, no dtype
    and no entry anywhere, while `params` counts it. That gap is small and it is
    everywhere: eleven of the thirty-nine snapshots, every one of them exactly
    one bias short per biased node.

    The weight, first, by precedence of specificity:

      1. an explicit `weight_shape` — the op states its geometry outright,
         which is the only way to express a shape that is not derivable from
         `out_key` and a parameter count (a norm gain, an embedding table);
      2. an `out_key` plus a parameter count — linear geometry, where `out` is
         the declared output axis and `in` follows by division. Exact, and it
         names no symbol;
      3. neither — the op owns nothing.

    Everything here is ONE copy. How many copies the model holds is the walk's
    answer, not the op's, and folding it in would make `in * out == params`
    false for every expert bank.
    """
    own = op.weight_tensor_suffix
    decls = op.tensors or ()
    lead = weight_leading_axes(op, dims)
    if decls:
        return tuple(
            StoredTensor(
                module=module,
                # A declaration saying `.weight` means "the op's own suffix",
                # which is `""` for a naked `nn.Parameter`. Anything else names
                # a sibling outright.
                suffix=(own if not decl.suffix or decl.suffix == ".weight"
                        else decl.suffix),
                logical_suffix=(own if not decl.suffix
                                or decl.suffix == ".weight" else decl.suffix),
                shape=(*lead, *shape),
                # A sibling carries the weight's leading axes too, ahead of
                # the ones it follows.
                axis_refs=(None if decl.axis_refs is None
                           else (*range(len(lead)),
                                 *(ref + len(lead) for ref in decl.axis_refs))),
                # What KIND of tensor each matrix of it is; the leading axes
                # only say how many there are.
                kind=(StorageKind(decl.kind) if decl.kind
                      else StorageKind.of(shape, op)))
            for decl, shape in _resolve_declarations(op, dims))

    shape = _declared_shape(op, dims, resolve_expr)
    if shape is None or not shape:
        return ()
    return (StoredTensor(
        module=module, suffix=own, logical_suffix=own,
        shape=(*lead, *shape),
        kind=StorageKind.of(tuple(shape), op)),)


class Undeclarable(ValueError):
    """A tensor an op says it owns, whose shape will not resolve.

    Raised, not absorbed. The two places that used to absorb it did so in
    opposite directions and so covered for each other: `from_declaration`
    dropped the tensor from storage, `declared_elements` handed `params` back to
    `params_fn` — and `Σ stored elements == params`, the invariant they exist to
    make true, stayed true while both answers were wrong. An invariant that
    survives its own violation is not checking anything.
    """


def _resolve_declarations(op: Any, dims: Mapping[str, Any]) -> Tuple:
    """`(decl, shape)` for every tensor the op owns HERE.

    A `when:` that is off means the tensor does not exist at this binding — a
    `linear` with no bias owns one tensor, not one and a broken one. Anything
    else failing to resolve is a defect in the declaration, and it is raised.
    """
    out = []
    for decl in op.tensors or ():
        if not decl.flag(dims):
            continue
        from backend.arch.ops import resolve_shape_template

        try:
            # No `dict(dims)`: the resolver only subscripts, and copying a
            # ReadTracking map would mark every dim read.
            shape = resolve_shape_template(decl.shape, dims)
        except InexactDivision as exc:
            raise Undeclarable(
                f"{getattr(op, 'name', op)!r} declares {decl.suffix!r} but "
                f"{exc}") from exc
        if not shape:
            raise Undeclarable(
                f"{getattr(op, 'name', op)!r} declares {decl.suffix!r} with a "
                f"shape that will not resolve at these dims")
        out.append((decl, tuple(shape)))
    return tuple(out)


def declared_elements(op: Any, dims: Mapping[str, Any]) -> Optional[Extent]:
    """Total parameters, PROJECTED from the tensor list — or `None` when the op
    declares no list and something else has to answer.

    This is the direction that matters. `params_fn` and the storage geometry
    used to be two independent declarations of one fact, checked against each
    other by nobody, and that is precisely how a bias came to be counted in
    `params` and absent from storage at the same time in eleven models. Making
    one the projection of the other removes the disagreement as a possibility
    rather than as a bug.

    `None` means the op declares no tensors AT ALL, which is not a failure and
    is still the common case — see this module's header for the coverage. A
    declaration that exists and will not resolve raises instead: falling back to
    `params_fn` there is the disagreement coming back through the door the
    projection was built to close.
    """
    if not (op.tensors or ()):
        return None
    total: Extent = Dim.of(0)
    for _decl, shape in _resolve_declarations(op, dims):
        own: Extent = Dim.of(1)
        for extent in shape:
            own = own * extent
        total = total + own
    for extent in weight_leading_axes(op, dims):
        total = total * extent
    return total


def _declared_shape(op: Any, dims: Mapping[str, Any],
                    resolve_expr) -> Optional[Tuple]:
    """A `weight_shape` the op states for itself."""
    del resolve_expr
    template = op.weight_shape
    if not template:
        return None
    from backend.arch.ops import resolve_shape_template
    return resolve_shape_template(template, dict(dims))

