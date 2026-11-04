"""CustomOpSpec — agent-authored novel primitive ops declared in spec YAML."""
from __future__ import annotations

from typing import (Any, Dict, List, Optional, Tuple)

from pydantic import (BaseModel, Field, model_validator)

from backend.arch.ops_base import INPUT_ROLE_VOCABULARY
# Declared with the op registry, which registered ops share with custom ops.
from backend.arch.ops_base import AXIS_USE_CATEGORIES as AXIS_USE_CATEGORIES
from backend.arch.ops_base import AxisUseDecl as AxisUseDecl




class CustomOpSpec(BaseModel):
    """Agent-authored novel primitive op, declared in spec YAML.

    Scoped to the enclosing ArchSpec: any node whose `op` matches a key in
    `ArchSpec.custom_ops` is evaluated against this definition (not against
    `backend.arch.ops.OP_REGISTRY`). Use this when a model introduces a
    compute primitive not representable by existing registered ops — declare
    it here rather than touching Python. Once reviewed, it can be promoted
    to `ops.py::register_op` for library-wide reuse.

    This declares exactly what `ops.build_custom_op_spec()` needs to
    CONSTRUCT an `ops.OpSpec` — not a field-level copy of one. An op's
    vocabulary has a single home (`OpSpec`); what an author writes here are
    its inputs, and the registration flow supplies the rest. A behavior an
    author cannot express in these fields is a behavior that belongs in
    `ops.py`, where it can carry a DimMap or ring-attention
    `internal_comm` that made it inexpressible in the first place.

    Fields:
      flops     — SymPy expression over bound dims (B, S, and spec dims).
      params    — SymPy expression; "0" for param-less ops.
      signature — Ordered map of operand_name -> shape list. Declares the op's
                  arity (checked against every call site's `in:`) and each
                  operand's shape; the engine's positional input shapes are
                  this map's values, in order.
      shape_kind — How the output shape derives from the inputs, from the
                  `ops.ShapeKind` taxonomy. The engine compiles it to the
                  same DimMap registered primitives use, so a custom op
                  participates in upstream-driven shape inference and gets
                  the same coverage from validate_spec's cross-input
                  consistency check.
      formula   — One-line math description shown in the node UI.
      description — Why this can't decompose into existing primitives.
    """
    model_config = {"extra": "forbid"}

    flops: str = "0"
    params: str = "0"
    signature: Dict[str, List[Any]]
    formula: str = ""
    description: str = ""
    shape_kind: str
    out_key: Optional[str] = None
    axis_locked: Optional[int] = None
    # Multi-output port specifications. Each entry maps a port name to a SHAPE
    # TEMPLATE — list of dim symbol names, e.g.
    # `[B, s_ctx, n_hidden_copies, n_hidden_copies]`.
    # Downstream nodes reference secondary outputs as `<node>.<port>`,
    # disambiguated against `secondary_outputs` keys at load time. Empty
    # (default) means single-output (referenced as bare `<node>` or
    # `<node>.out`). The main `out` port still comes from `shape_kind`.
    secondary_outputs: Dict[str, List[Any]] = {}
    # The op is a recurrence streaming a state across decode steps — see
    # `ops.OpSpec.carries_recurrent_state`. Reachable from YAML because a
    # component may implement its recurrence with a custom op, and every node
    # running one must be named as a `state:` entry's `carrier:`.
    # `gla_simple_recurrence` is one.
    carries_recurrent_state: bool = False
    # What each operand IS TO THE OP, aligned with the signature's operand
    # order — see `ops.OpSpec.input_roles`. Reachable from YAML because a
    # custom op can be the attention core itself (GLA's recurrence reads a
    # query, keys and values), and a reader of the graph — the fusion
    # matcher telling Q's rope from K's — finds the operand a node feeds by
    # the role the core declares for it. None at a position that plays no
    # part a reader tells apart; the whole field None for an op that is no
    # anchor. Left out of the dump when undeclared, so declaring it on one op
    # does not change the serialized form of every other.
    input_roles: Optional[List[Optional[str]]] = Field(
        default=None, exclude_if=lambda roles: roles is None)
    # The shape labels this op is NOT separable along — see
    # `ops.OpSpec.couples_on`, whose three states this inherits exactly.
    # Reachable from YAML because a custom op is a first-class primitive and
    # only its author knows what it reduces, joins or re-partitions. An empty
    # list explicitly declares a fully separable op.
    #
    # The names are the `signature:` keys' SHAPE LABELS, the same vocabulary
    # a registered op's `shape_in` uses — `In` for an op declaring
    # `x: [B, s_ctx, In]`, not `x`.
    couples_on: List[str]
    # On-disk weight-tensor-name suffix. Default `.weight` matches
    # nn.Linear / nn.Embedding storage; empty `""` matches naked
    # nn.Parameter idiom (no `.weight` suffix on the checkpoint
    # tensor). Used by `spec_tree.ir_to_leaf` when the custom op
    # also declares a weight (rare — most custom ops are zero-param
    # algorithmic kernels).
    weight_tensor_suffix: str = ".weight"
    # On-disk weight tensor shape as a dim-name template. Set ONLY for
    # weight-bearing custom ops whose geometry is not the default `(out, in)`
    # linear — e.g. `[d_head]` for a norm scale, `[n_vocab, d_model]` for an embedding.
    # Resolved against the node's dims.
    weight_shape: Optional[List[Any]] = None
    # How the op couples along each axis in `couples_on`, for the layout
    # walk. Declared by the author for the recurrent and geometry families;
    # for the linear family (a `[Out, In]` weight) it is derived — the in
    # axis contracts by sum, like every linear — and must not be repeated.
    axis_use: Dict[str, AxisUseDecl] = {}

    # --- OpView totalisation -------------------------------------------------
    # A custom op is a YAML-declared compute kernel: no Python lambdas, no
    # TensorDecl list. These six attributes that OpSpec carries are absent by
    # construction. Expose their honest custom-op values as read-only
    # @property (NOT fields) so a YAML author still cannot set them
    # (`extra="forbid"` rejects the keys) and they never enter `model_dump`
    # (keeping the calibration spec-hash stable). Each mirrors what OpRef's own
    # `if self._custom:` methods already returned.
    @property
    def is_view_op(self) -> bool:
        return False  # a custom op is never a view op (op_ref documents this)

    @property
    def is_buffer(self) -> bool:
        return False  # a compute kernel, never a precomputed buffer

    @property
    def processor_step(self) -> Optional[str]:
        return None  # a compute kernel, never a step of a media processor

    @property
    def tensors(self) -> Tuple:
        return ()  # weights are declared via weight_shape, not a TensorDecl list

    @property
    def shape_in(self) -> Tuple:
        """The axis labels of the op's FIRST operand, the vocabulary
        `couples_on` names (see its own note above). A registered op states
        these as `OpSpec.shape_in`; a YAML op states them as its first
        `signature:` entry, and a reader of either asks one attribute rather
        than probing for two."""
        first = next(iter(self.signature.values()), None)
        return tuple(first) if first else ()

    @property
    def alt_shape_in(self) -> Tuple:
        return ()  # a YAML op has no alternate-rank signatures

    @property
    def permute_to(self) -> Optional[Tuple[int, ...]]:
        return None  # not a permute-axes op

    @property
    def linear_labels(self) -> Optional[Tuple[str, str]]:
        """`(out, in)` when the op is a linear map — `x[..., in] -> [..., out]`
        with a stored `[out, in]` weight — else None. The output label is
        `out_key` under the axis rewrite that puts it there; the weight is
        either declared `weight_shape: [out, in]` or the default `(out, in)`
        geometry a weight-bearing op with no `weight_shape` gets, whose in
        axis is the operand's last label. A 2-D weight alone is not enough:
        an embedding table is `[V, d]` and a lookup, not a contraction."""
        if self.shape_kind != "axis_replace" or self.out_key is None:
            return None
        out = str(self.out_key)
        shape = tuple(str(x) for x in (self.weight_shape or ()))
        if shape:
            return (out, shape[1]) if len(shape) == 2 and shape[0] == out else None
        if str(self.params or "0").strip() == "0" or len(self.signature) != 1:
            return None
        operand = next(iter(self.signature.values()))
        return (out, str(operand[-1])) if operand else None

    @property
    def is_linear_map(self) -> bool:
        return self.linear_labels is not None

    @property
    def layout_family(self) -> str:
        """Which layout law the op follows, from what it declares: a linear
        map (`is_linear_map`); a recurrent state makes it `recurrent`;
        anything else is `geometry`, whose coupling only its author's
        `axis_use` can state."""
        if self.is_linear_map:
            return "linear"
        if self.carries_recurrent_state:
            return "recurrent"
        return "geometry"

    def layout_axis_use(self) -> Dict[str, AxisUseDecl]:
        """The op's axis tags: derived for the linear family, declared
        otherwise. One query, whichever family."""
        labels = self.linear_labels
        if labels is not None:
            return {labels[1]: AxisUseDecl(category="contract", param="sum")}
        return dict(self.axis_use)

    # Shape kinds an author can reach from YAML. EXPLICIT and SOURCE need a
    # DimMap in `ops.py`, which a YAML declaration cannot supply.
    _YAML_SHAPE_KINDS = (
        "preserve", "broadcast", "axis_replace", "axis_select",
        "axis_concat", "permute", "contract",
    )

    @model_validator(mode="after")
    def _validate_shape_kind(self) -> "CustomOpSpec":
        if self.shape_kind not in self._YAML_SHAPE_KINDS:
            raise ValueError(
                f"CustomOpSpec: `shape_kind={self.shape_kind!r}` is not one of "
                f"{list(self._YAML_SHAPE_KINDS)}. `explicit` / `source` "
                f"need a DimMap in `ops.py::register_op`, not a YAML "
                f"declaration."
            )
        # Both axis rewrites need the named dim that supplies the new extent
        # (directly for axis_replace, as an upper bound for axis_select).
        if self.shape_kind in ("axis_replace", "axis_select") and self.out_key is None:
            raise ValueError(
                f"CustomOpSpec: shape_kind={self.shape_kind} requires "
                "`out_key` (the dim name whose resolved value becomes "
                "the new axis size or selection bound)."
            )
        return self

    @model_validator(mode="after")
    def _validate_signature(self) -> "CustomOpSpec":
        if not self.signature:
            raise ValueError(
                "CustomOpSpec: `signature` is required — an op's arity and "
                "its operands' shapes are facts only the author knows, and "
                "call-site `in:` lists are checked against them."
            )
        if len(self.couples_on) != len(set(self.couples_on)):
            raise ValueError(
                "CustomOpSpec.couples_on must not contain duplicate labels")
        shape_labels = {
            str(label)
            for shape in self.signature.values()
            for label in shape
            if isinstance(label, str)
        }
        unknown = sorted(set(self.couples_on) - shape_labels)
        if unknown:
            raise ValueError(
                f"CustomOpSpec.couples_on references shape label(s) {unknown} "
                f"not present in signature shapes {sorted(shape_labels)}")
        if not self.description.strip():
            raise ValueError(
                "CustomOpSpec.description must explain why the op cannot be "
                "decomposed into registered primitives")
        roles = [r for r in (self.input_roles or ()) if r is not None]
        unknown_roles = sorted(set(roles) - INPUT_ROLE_VOCABULARY)
        if unknown_roles:
            raise ValueError(
                f"CustomOpSpec.input_roles: unknown role(s) {unknown_roles}; "
                f"valid: {sorted(INPUT_ROLE_VOCABULARY)}")
        if len(set(roles)) != len(roles):
            raise ValueError(
                f"CustomOpSpec.input_roles: duplicate role in {self.input_roles!r}")
        if len(self.input_roles or ()) > len(self.signature):
            raise ValueError(
                f"CustomOpSpec.input_roles: {len(self.input_roles)} roles for "
                f"{len(self.signature)} signature operand(s)")
        invalid_ports = sorted(
            name for name in self.secondary_outputs
            if not name.strip() or name == "out")
        if invalid_ports:
            raise ValueError(
                "CustomOpSpec.secondary_outputs uses invalid port name(s) "
                f"{invalid_ports}; names must be non-empty and `out` is the "
                "reserved main-output port")
        return self

    @model_validator(mode="after")
    def _axis_use_matches_coupling(self) -> "CustomOpSpec":
        labels = self.linear_labels
        if labels is not None:
            if self.axis_use:
                raise ValueError(
                    "CustomOpSpec: a custom op applying a `[out, in]` weight is "
                    "a linear map; its axis_use is derived (in contracts by "
                    "sum) and must not be declared")
            expected = {labels[1]}
        else:
            expected = set(self.axis_use)
        if set(self.couples_on) != expected:
            raise ValueError(
                f"CustomOpSpec: couples_on {sorted(self.couples_on)} must name "
                f"exactly the tagged axes {sorted(expected)} — every coupled "
                "axis says HOW it couples (axis_use), and no untagged axis "
                "couples")
        return self

    def arity(self) -> int:
        """Number of inputs this op expects — the signature's length.

        Used by ArchSpec validation to check call-site `in:` length.
        """
        return len(self.signature)
