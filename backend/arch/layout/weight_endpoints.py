"""Occurrence-declared physical layouts for stored weights.

``dims.yaml`` declares capability only: it says which worlds *may* shard a
symbol. Persistent storage is stricter. Each weight-bearing occurrence
declares ``weight_axes`` with its actual
runtime world, part-count expression, fit policy, and any fused segments —
one entry per stored axis, an expert bank's leading ``n_expert_copies`` axis
included (``world: moe_copies``). A missing declaration is rejected by
physical artifact derivation; this module never turns a capable symbol into
placement by inference.

The checkpoint and runtime layouts are the two endpoints of a load-time
transform. They may also differ in axis ORDER: ``stored_axes`` declares that
the checkpoint stores the tensor as a permutation of the op's weight shape,
and ``WeightGeometry`` is the one place that permutation is applied.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import gcd
import operator
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import sympy as sp

from backend.arch import dims
from backend.arch.eval.refusal import SpecRefusal, node_context
from backend.arch.eval.dim import Dim, eval_dim_expr, expr_of, value_of
from backend.arch.eval.op_ref import OpRef

from .layout import (
    AxisLayout,
    TensorLayout,
    block,
    physical_local_extent,
    replicated,
    segmented,
    tensor_layout,
)

if TYPE_CHECKING:
    from .propagation import LayoutPropagation


@dataclass(frozen=True)
class WeightEndpoint:
    """Checkpoint and runtime contracts for one declared stored tensor.

    Two axis orders meet here. ``checkpoint_shape`` / ``checkpoint_layout``
    are in the checkpoint's payload order (after the expert-copy carrier);
    everything else — ``global_shape``, ``runtime_layout``, ``runtime_shape``,
    the segment tuples and ``runtime_symbol_extents`` — is in the op's
    weight-shape order, the order the rank computes with. The two differ only
    where the occurrence declares ``stored_axes`` (``WeightGeometry``).
    """

    node_uid: int
    tensor_index: int
    tensor_name: str
    tensor_logical_name: str
    #: The whole tensor as the checkpoint stores one copy of it.
    checkpoint_shape: Tuple[int, ...]
    #: The whole tensor in the op's weight-shape order
    #: (``PhysicalArtifact.global_shape``).
    global_shape: Tuple[int, ...]
    #: Explicit segment extent expressions, one tuple per runtime axis.
    axis_segments: Tuple[Tuple[str, ...], ...]
    #: Human-readable segment identities (Q/K/V, gate/up, ...).
    axis_segment_names: Tuple[Tuple[str, ...], ...]
    checkpoint_layout: TensorLayout
    runtime_layout: TensorLayout
    #: Local values proven by the same axis contracts as ``runtime_layout``.
    #: The immutable representation prevents callers from substituting an
    #: global node binding when projecting metadata.
    runtime_symbol_extents: Tuple[Tuple[str, int], ...] = ()
    #: Symbols that size two axes of this tensor sharded differently, and so
    #: have no one local value (`_merge_symbol_extents`).
    ambiguous_symbols: Tuple[str, ...] = ()

    @property
    def runtime_shape(self) -> Tuple[int, ...]:
        """One rank's tensor, in the op's weight-shape order."""
        return physical_shape(self.runtime_layout, self.global_shape)


@dataclass(frozen=True)
class WeightGeometry:
    """One stored-weight owner's axes, in checkpoint and in runtime order.

    ``weight_axes`` lists one contract per checkpoint payload axis. The op
    computes with its ``weight_shape`` order, and ``stored_axes`` relates the
    two: checkpoint axis ``i`` is runtime axis ``stored_axes[i]`` (absent: the
    identity). Built from the IR node, the op's weight shape and the node's
    stored tensors; every consumer that needs either order asks this object
    — the walk places the runtime axes, the endpoint and the checkpoint
    oracle read the checkpoint order back through ``to_checkpoint`` — so the
    permutation is applied in exactly one place.

    Sibling tensors (a bias, quantization metadata) project the primary
    weight through ``TensorDecl.axis_refs`` in runtime coordinates, and how a
    quant format packs a transposed payload is not declared anywhere. A
    non-identity order is therefore accepted only on an owner whose one
    stored tensor is saved unquantized; everything else is refused, not
    guessed.
    """

    display_path: str
    #: The op's weight-shape entries (or, lacking one of the right rank, the
    #: stored extents), runtime order.
    slots: Tuple[str, ...]
    #: Dim expression of every runtime axis.
    sources: Tuple[str, ...]
    #: The declared ``weight_axes`` contract of every runtime axis.
    contracts: Tuple[Any, ...]
    #: Checkpoint axis ``i`` is runtime axis ``stored_axes[i]``.
    stored_axes: Tuple[int, ...]

    @classmethod
    def of(
        cls,
        node: Any,
        weight_shape: Sequence[Any],
        tensors: Sequence[Any],
    ) -> "WeightGeometry":
        """Pair ``node.weight_axes`` with the op's axes.

        ``weight_shape`` is the op's declared template; a source op without
        one uses the node's explicit ``shape``, and a template of another
        rank than the declaration (``parameter_tensor``'s ``_shape``) falls
        back to the node's explicit ``shape``, else the stored tensor's own
        extents (a call tied to another's weights stores none).
        """
        declared = tuple(node.weight_axes)
        rank = len(declared)
        template = tuple(weight_shape) or tuple(node.shape_raw or ())
        shape_raw = tuple(node.shape_raw or ())
        if (weight_shape and "_shape" not in tuple(weight_shape)
                and len(shape_raw) == rank and len(weight_shape) < rank):
            # A weight declared with axes ahead of the op's own
            # (`ops.weight_leading_axes`): those lead, named as declared,
            # and the op's own slots (`out`, `in`) follow.
            template = (*shape_raw[:rank - len(weight_shape)],
                        *tuple(weight_shape))
        elif len(template) != rank and len(shape_raw) == rank:
            template = shape_raw
        if template and len(template) == rank:
            slots = tuple(str(entry) for entry in template)
            sources = tuple(_slot_source(node, entry) for entry in template)
        elif tensors and len(tensors[0].shape) == rank:
            slots = tuple(_extent_source(extent) for extent in tensors[0].shape)
            sources = tuple(_slot_source(node, slot) for slot in slots)
        else:
            raise ValueError(
                f"{node.display_path}: weight_axes declares {rank} axes but "
                f"the op's weight shape is {template!r}"
            )
        order = node.stored_axes
        if order is None:
            return cls(node.display_path, slots, sources, declared,
                       tuple(range(rank)))
        order = tuple(order)
        if tuple(sorted(order)) != tuple(range(rank)):
            raise ValueError(
                f"{node.display_path}: stored_axes {list(order)!r} is not a "
                f"permutation of the {rank} declared weight axes"
            )
        if len(template) != rank:
            raise ValueError(
                f"{node.display_path}: stored_axes {list(order)!r} orders "
                f"{rank} axes, but op {node.op!r} declares weight shape "
                f"{tuple(weight_shape)!r}"
            )
        if len(tensors) != 1:
            raise ValueError(
                f"{node.display_path}: stored_axes reorders a weight with "
                f"{len(tensors)} stored tensors; sibling tensors (a bias) "
                "project the primary weight in runtime order, and their "
                "checkpoint order is not declared — only a single-tensor "
                "owner may declare stored_axes"
            )
        runtime = [None] * rank
        for checkpoint_axis, runtime_axis in enumerate(order):
            runtime[runtime_axis] = declared[checkpoint_axis]
        return cls(node.display_path, slots, sources, tuple(runtime), order)

    @property
    def is_identity(self) -> bool:
        return self.stored_axes == tuple(range(len(self.stored_axes)))

    def runtime_axes(self) -> Tuple[Tuple[str, Any], ...]:
        """(source expression, contract) of every runtime axis."""
        return tuple(zip(self.sources, self.contracts, strict=True))

    def to_checkpoint(
        self, per_runtime_axis: Sequence[Any], tensor_index: int = 0
    ) -> Tuple[Any, ...]:
        """Reorder one value per runtime axis into checkpoint axis order.

        Only the primary tensor (index 0) is reordered: ``of`` refuses a
        reordered owner with siblings, so a sibling's two orders agree.
        """
        values = tuple(per_runtime_axis)
        if tensor_index:
            if not self.is_identity:
                raise ValueError(
                    f"{self.display_path}: a reordered weight has no "
                    "sibling tensors"
                )
            return values
        if len(values) != len(self.stored_axes):
            raise ValueError(
                f"{self.display_path}: {len(values)} runtime axes for a "
                f"rank-{len(self.stored_axes)} weight"
            )
        return tuple(values[axis] for axis in self.stored_axes)

    def require_saved_unquantized(self, reps: Any) -> None:
        """Refuse a reordered weight whose saved format packs or scales it.

        Asked of every layer segment of the tensor (`Representations`): a
        weight quantised on some of its node's layers and not others is
        quantised, and a reorder of it is just as undeclared.

        The runtime (resident) side needs no check: a deployment that
        quantizes at load quantizes the runtime-order tensor.
        """
        for segment in reps:
            saved = segment.saved
            if (saved.method is not None or saved.metadata
                    or saved.packing_factor != 1):
                raise ValueError(
                    f"{self.display_path}: stored_axes reorders a weight saved "
                    f"under quant method {saved.method!r}; how a format packs or "
                    "scales a reordered payload is not declared — only an "
                    "unquantized weight may declare stored_axes"
                )

    def tensor_axes(
        self, tensors: Sequence[Any]
    ) -> Tuple[Tuple[Tuple[str, Any], ...], ...]:
        """(source, contract) of every axis of every stored tensor, runtime
        order; siblings project the primary through ``axis_refs``."""
        if not tensors:
            return ()
        stored = tensors[0]
        if len(self.contracts) != len(stored.shape):
            raise ValueError(
                f"{self.display_path}: weight_axes has {len(self.contracts)} "
                f"axes, but stored tensor {stored.name!r} has rank "
                f"{len(stored.shape)}"
            )
        primary = self.runtime_axes()
        out = [primary]
        for tensor in tensors[1:]:
            refs = tensor.axis_refs
            if refs is None:
                raise ValueError(
                    f"{tensor.name}: sibling tensor declares no stored-axis refs"
                )
            if len(refs) != len(tensor.shape):
                raise ValueError(
                    f"{tensor.name}: {len(refs)} stored-axis refs for rank "
                    f"{len(tensor.shape)}"
                )
            if any(index < 0 or index >= len(stored.shape) for index in refs):
                raise ValueError(
                    f"{tensor.name}: stored-axis refs {refs!r} exceed rank "
                    f"{len(stored.shape)}"
                )
            out.append(tuple(primary[index] for index in refs))
        return tuple(out)


def physical_shape(
    layout: TensorLayout, logical_shape: Sequence[int]
) -> Tuple[int, ...]:
    """The equal physical shape one rank allocates under ``layout``."""
    shape = tuple(
        _positive_int(extent, "logical axis extent") for extent in logical_shape
    )
    if len(shape) != len(layout.axes):
        raise ValueError(
            f"layout rank {len(layout.axes)} does not match shape rank {len(shape)}"
        )
    return tuple(
        physical_local_extent(axis, extent)
        for axis, extent in zip(layout.axes, shape, strict=True)
    )


def instantiate(step: Any) -> Tuple[WeightEndpoint, ...]:
    """Instantiate every occurrence that declares ``weight_axes``.

    The rest are not guessed; physical artifact derivation rejects missing
    declarations when it cannot derive a persistent artifact.
    """
    return instantiate_analysis(step.analysis, placement_context(step.analysis))


def placement_context(analysis: Any) -> "PlacementContext":
    """The persistent-placement context of one analysis, read off the layout
    rule that walked it. The walk already holds the deployment, phase,
    process-group families and rank universe; rebuilding them from the step
    and comparing was a second construction of the same facts."""
    try:
        return analysis.rule.placement_context
    except AttributeError:
        raise TypeError(
            "physical artifacts require an analysis walked under "
            "LayoutPropagation") from None


def instantiate_analysis(
    analysis: Any,
    context: "PlacementContext",
) -> Tuple[WeightEndpoint, ...]:
    """Instantiate declared weights from an analysis and placement context.

    This constructor consumes an existing ``Analysis`` and therefore cannot
    trigger another evaluator walk.
    """
    out = []

    for uid in sorted(analysis.storage.values):
        node = analysis.program.by_uid(uid)
        if node.weight_axes is None:
            continue
        op = OpRef.resolve(node.op or "", analysis.program)
        if op is None:
            raise ValueError(
                f"stored tensor owner {node.display_path!r} has unknown op {node.op!r}"
            )
        tensors = tuple(analysis.storage.values[uid])
        binding = analysis.node_dims.values.get(uid)
        if binding is None:
            raise ValueError(
                f"stored tensor owner {node.display_path!r} has no node binding"
            )
        geometry = WeightGeometry.of(node, op.spec.weight_shape or (), tensors)
        if not geometry.is_identity:
            geometry.require_saved_unquantized(
                analysis.formats.values[uid][0])
        axes = geometry.tensor_axes(tensors)
        recorded = _walk_placement(analysis, uid)
        for index, (tensor, axis_contracts) in enumerate(
            zip(tensors, axes, strict=True)
        ):
            shape = tuple(
                _positive_int(value_of(extent), f"{tensor.name} axis extent")
                for extent in tensor.shape
            )
            if len(shape) != len(axis_contracts):
                raise ValueError(
                    f"{tensor.name}: {len(axis_contracts)} declared axes for "
                    f"rank-{len(shape)} storage"
                )
            with node_context(node.display_path):
                assembled = _endpoints_from_walk(
                    recorded, tensor, index, shape,
                )
            if assembled is not None:
                checkpoint_axes, runtime_axes, axis_segments, axis_segment_names, runtime_symbols = assembled
            else:
                checkpoint_axes = []
                runtime_axes = []
                axis_segments = []
                axis_segment_names = []
                runtime_symbols = {}
                # In the view the walk placed this node in.
                placed = replace(context, absent_meshes=analysis.rule.view_of(
                    uid).absent_meshes)
                for extent, (source, contract) in zip(shape, axis_contracts, strict=True):
                    checkpoint, runtime, terms, names, symbols = (
                        placed.axis_endpoints(node, source, contract, extent, binding)
                    )
                    checkpoint_axes.append(checkpoint)
                    runtime_axes.append(runtime)
                    axis_segments.append(terms)
                    axis_segment_names.append(names)
                    _merge_symbol_extents(runtime_symbols, symbols)
            out.append(
                WeightEndpoint(
                    node_uid=uid,
                    tensor_index=index,
                    tensor_name=tensor.name,
                    tensor_logical_name=tensor.logical_name,
                    checkpoint_shape=geometry.to_checkpoint(shape, index),
                    global_shape=shape,
                    axis_segments=tuple(axis_segments),
                    axis_segment_names=tuple(axis_segment_names),
                    checkpoint_layout=tensor_layout(
                        geometry.to_checkpoint(checkpoint_axes, index)),
                    runtime_layout=tensor_layout(runtime_axes),
                    runtime_symbol_extents=_proven(runtime_symbols),
                    ambiguous_symbols=tuple(sorted(
                        name for name, extent in runtime_symbols.items()
                        if extent is None)),
                )
            )
    return tuple(out)


def _walk_placement(analysis: Any, uid: int) -> Any:
    """The walk's recorded weight placement, or None if it kept runtime only."""
    rule: LayoutPropagation = analysis.rule
    recorded = rule.weights
    if not recorded:
        return None
    placement = recorded.get(uid)
    if placement is None or not placement.checkpoint_axes:
        return None
    return placement


def _endpoints_from_walk(
    recorded: Any,
    tensor: Any,
    index: int,
    shape: Tuple[int, ...],
) -> Optional[Tuple[list, list, list, list, dict]]:
    """Assemble one tensor's endpoints from the walk, or None to recompute."""
    if recorded is None:
        return None
    ckpt = recorded.checkpoint_axes
    runtime = recorded.axes
    if index == 0:
        if len(ckpt) != len(shape) or len(runtime) != len(shape):
            return None
        refs = range(len(shape))
    else:
        refs = tensor.axis_refs
        if refs is None or len(refs) != len(shape):
            return None
        if any(i < 0 or i >= len(ckpt) for i in refs):
            return None
    runtime_symbols: dict[str, Optional[int]] = {}
    for i in refs:
        _merge_symbol_extents(runtime_symbols, recorded.axis_symbols[i])
    return (
        [ckpt[i] for i in refs],
        [runtime[i] for i in refs],
        [recorded.axis_segments[i] for i in refs],
        [recorded.axis_segment_names[i] for i in refs],
        runtime_symbols,
    )


def _validated_deployment(step: Any) -> Any:
    pc = getattr(step, "parallel", None)
    if pc is None:
        raise ValueError("weight endpoints require a step with a deployment")
    errors = pc.validate()
    if errors:
        raise ValueError(
            "weight endpoints require a valid deployment: " + "; ".join(errors)
        )
    if step.phase not in dims.PHASES:
        raise ValueError(
            f"unknown phase {step.phase!r}; expected one of {list(dims.PHASES)}"
        )
    return pc


def _slot_source(node: Any, entry: Any) -> str:
    if isinstance(entry, str):
        if entry in (node.dims_raw or {}):
            return str(node.dims_raw[entry])
        if entry == "dim" and node.dim_raw:
            return str(node.dim_raw)
    return str(entry)


def _extent_source(extent: Any) -> str:
    return str(expr_of(extent)) if isinstance(extent, Dim) else str(extent)


# 不可折叠：已验证部署上下文只应有一个；不要再造一份 DeclarationBinding。
@dataclass(frozen=True)
class PlacementContext:
    """One validated deployment's placement inputs for stored-axis contracts."""

    pc: Any
    phase: str
    families: Mapping[str, Any]
    scope: frozenset
    #: Meshes that do not exist for the nodes placed here
    #: (`LayoutPropagation.absent_meshes`): a placement on a world of one is
    #: whole.
    absent_meshes: frozenset = frozenset()

    def axis_endpoints(
        self,
        node: Any,
        source: str,
        contract: Any,
        extent: int,
        binding: Mapping[str, Any],
    ) -> Tuple[
        AxisLayout,
        AxisLayout,
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[Tuple[str, int], ...],
    ]:
        """Instantiate one explicit stored-axis contract.

        Copy endpoints reuse this for the independently factored payload axis
        of an axis-carried expert bank.  Keeping one constructor means segment
        order and placement validation cannot drift between the two stored
        tensor representations.
        """
        if contract.segments is None:
            terms = (source,)
            names = ("",)
            placements = (contract,)
        else:
            terms = tuple(str(child.extent) for child in contract.segments)
            names = tuple(str(child.segment) for child in contract.segments)
            placements = tuple(contract.segments)
        values = tuple(_eval_extent(term, binding) for term in terms)
        if sum(values) != extent:
            raise ValueError(
                f"{node.display_path} axis contract {terms!r} sums to {sum(values)}, "
                f"not stored extent {extent}"
            )

        checkpoint_children = [replicated(self.scope) for _ in terms]
        runtime_children = []
        checkpoint_segments = []
        runtime_segments = []
        runtime_symbols: dict[str, Optional[int]] = {}
        offset = 0
        for term, term_extent, placement, checkpoint_child in zip(
            terms, values, placements, checkpoint_children, strict=True
        ):
            runtime_child, symbols = self._placement_layout(
                node,
                term,
                placement,
                term_extent,
                binding,
            )
            checkpoint_segments.append((offset, term_extent, checkpoint_child))
            runtime_segments.append((offset, term_extent, runtime_child))
            runtime_children.append(runtime_child)
            _merge_symbol_extents(runtime_symbols, symbols)
            offset += term_extent

        if len(terms) == 1:
            checkpoint = checkpoint_children[0]
            runtime = runtime_children[0]
        else:
            checkpoint = segmented(checkpoint_segments)
            runtime = segmented(runtime_segments)
        return (
            checkpoint,
            runtime,
            terms,
            names,
            tuple(sorted(runtime_symbols.items())),
        )

    def _placement_layout(
        self,
        node: Any,
        term: str,
        placement: Any,
        term_extent: int,
        binding: Mapping[str, Any],
    ) -> Tuple[AxisLayout, Tuple[Tuple[str, int], ...]]:
        term_value = _eval_tracked(term, binding)
        term_names = set(getattr(term_value, "names", ()) or ())
        symbol_extents = {
            name: _positive_int(
                value_of(_eval_tracked(name, binding)),
                f"{node.display_path} symbol {name!r}",
            )
            for name in term_names
        }
        if placement.placement_by_symbol:
            matches = sorted(term_names.intersection(placement.placement_by_symbol))
            if len(matches) != 1:
                raise ValueError(
                    f"{node.display_path}: weight endpoint expression {term!r} must "
                    "select exactly one declared placement symbol; got "
                    f"{matches!r} from {sorted(term_names)!r}"
                )
            placement = placement.placement_by_symbol[matches[0]]

        if placement.replicated:
            return (
                replicated(self.scope),
                tuple(sorted(symbol_extents.items())),
            )

        world_slug = placement.world
        world = dims.worlds().get(world_slug)
        if world is None:
            raise ValueError(
                f"{node.display_path}: weight endpoint selects unknown world "
                f"{world_slug!r}"
            )
        if world.bound_per_rank:
            raise ValueError(
                f"{node.display_path}: persistent weight endpoint cannot select "
                f"world {world_slug!r}"
            )
        if world.mesh in self.absent_meshes:
            return (
                replicated(self.scope),
                tuple(sorted(symbol_extents.items())),
            )

        parts_value = _eval_tracked(str(placement.parts), binding)
        parts_extent = _positive_int(
            value_of(parts_value), f"{node.display_path} part count"
        )
        part_names = tuple(sorted(getattr(parts_value, "names", ()) or ()))
        unrelated = sorted(set(part_names) - term_names)
        if unrelated:
            raise ValueError(
                f"{node.display_path}: parts expression {placement.parts!r} names "
                f"{unrelated!r}, which are not contained in axis {term!r}"
            )
        capable = [
            name
            for name in part_names
            if world_slug in (dims.get(name).shardable_by if dims.get(name) else ())
        ]
        if len(capable) != 1:
            raise ValueError(
                f"{node.display_path}: parts expression {placement.parts!r} must "
                f"resolve to exactly one symbol shardable by {world_slug!r}; got "
                f"{capable!r} from {part_names!r}"
            )
        if term_extent % parts_extent:
            raise ValueError(
                f"{node.display_path}: segment {term!r} extent {term_extent} is not "
                f"an integer multiple of its {placement.parts!r} part count "
                f"{parts_extent}"
            )

        degree = self.pc.degree_of_world(world.label, phase=self.phase)
        if degree == 1:
            return (
                replicated(self.scope),
                tuple(sorted(symbol_extents.items())),
            )
        family = self.families.get(world.label)
        if family is None or family.group_size != degree:
            raise ValueError(
                f"{node.display_path}: world {world.label!r} degree {degree} has "
                "no matching process-group family"
            )

        fit = placement.fit
        if fit == "exact":
            if parts_extent % degree:
                raise SpecRefusal(
                    f"{node.display_path}: {world.label} = {degree} cannot split "
                    f"this weight evenly: its `{capable[0]}` axis has "
                    f"{parts_extent} parts, which {degree} does not divide. "
                    f"Pick a {world.label} that divides {parts_extent}",
                    declaration="weight_axes",
                    fix=("`fit: exact` needs the parts to divide every degree "
                         "the product deploys; declare `fit: pad` or "
                         "`fit: replica_cells`, as the serving framework "
                         "splits this weight"),
                    node=node.display_path)
            num_shards = degree
            pad_to = None
        elif fit == "replica_cells":
            num_shards = gcd(parts_extent, degree)
            pad_to = None
        elif fit == "pad":
            num_shards = degree
            # Padding is expressed in whole declared parts.  A 128-wide head
            # therefore aligns each physical shard to 128 elements.
            pad_to = term_extent // parts_extent
        else:  # schema validation should make this unreachable
            raise ValueError(f"unknown weight endpoint fit {fit!r}")

        replication = family.group_size // num_shards
        cells = tuple(
            tuple(range(part * replication, (part + 1) * replication))
            for part in range(num_shards)
        )
        layout = block(family, cells, pad_to=pad_to)
        runtime_local = physical_local_extent(layout, term_extent)
        part_width = term_extent // parts_extent
        if runtime_local % part_width:
            raise ValueError(
                f"{node.display_path}: local segment extent {runtime_local} does "
                f"not contain a whole number of {part_width}-wide declared parts"
            )
        local_parts = runtime_local // part_width
        selected = capable[0]
        selected_numerator = local_parts * symbol_extents[selected]
        if selected_numerator % parts_extent:
            raise ValueError(
                f"{node.display_path}: local part count {local_parts} cannot "
                f"project symbol {selected!r} from {placement.parts!r}"
            )
        symbol_extents[selected] = selected_numerator // parts_extent
        return layout, tuple(sorted(symbol_extents.items()))


def local_symbol_extents(analysis: Any) -> Dict[int, Dict[str, Optional[int]]]:
    """Per-rank symbol values proven by each node's weight-axis contracts.

    The per-node join of ``WeightEndpoint.runtime_symbol_extents`` — the same
    facts quant-metadata projection already consumes. A FLOPs projection or
    kernel export that reads a bare symbol existing only as a FACTOR of a
    merged axis (``n_heads`` inside ``n_heads*d_head``) gets the rank's value
    from here instead of keeping the unsharded binding.

    An endpoint's extents mix the symbols its ``parts:`` placement actually
    divided with symbols the term merely mentions — a replicated or degree-1
    axis hands the global binding straight through. Only the divided ones
    carry per-rank information, and only they join the merge: a passthrough
    global value would otherwise collide with a sibling tensor's proven
    per-rank extent for the same symbol.

    A symbol that sizes two of a node's axes split differently has no one
    per-rank value and maps to None: `local_node_dims` refuses a binding
    that would need it.
    """
    facts: Dict[int, Dict[str, Optional[int]]] = {}
    for endpoint in instantiate_analysis(analysis, placement_context(analysis)):
        binding = analysis.node_dims.values.get(endpoint.node_uid)
        if binding is None:
            raise ValueError(
                f"weight endpoint owner {endpoint.node_uid!r} has no node "
                f"binding"
            )
        divided = tuple(
            (name, extent)
            for name, extent in endpoint.runtime_symbol_extents
            if extent != _positive_int(
                value_of(_eval_tracked(name, _scalar_binding(binding))),
                f"{endpoint.tensor_name} symbol {name!r}",
            )
        )
        _merge_symbol_extents(
            facts.setdefault(endpoint.node_uid, {}),
            (*divided, *((name, None) for name in endpoint.ambiguous_symbols)),
        )
    return facts


def _merge_symbol_extents(
    target: dict[str, Optional[int]],
    incoming: Sequence[Tuple[str, Optional[int]]],
) -> None:
    """Join local symbol facts. A symbol that sizes two axes sharded
    differently (a square projection `d -> d` split on its output only) has
    no one per-rank value: it joins as None. Each axis keeps its own extent
    through its layout; `WeightEndpoint` keeps only the proven symbols
    (`_proven`) and names the others (`ambiguous_symbols`), and
    `local_symbol_extents` hands them on as None for `local_node_dims` to
    refuse."""
    for name, extent in incoming:
        if name in target and target[name] != extent:
            target[name] = None
        else:
            target[name] = extent


def _proven(symbols: Mapping[str, Optional[int]]) -> Tuple[Tuple[str, int], ...]:
    """The symbols with one per-rank value (`_merge_symbol_extents`)."""
    return tuple(sorted((name, extent) for name, extent in symbols.items()
                        if extent is not None))


def _eval_tracked(source: str, binding: Mapping[str, Any],
                  what: str = "weight endpoint") -> Any:
    try:
        return eval_dim_expr(source, _scalar_binding(binding))
    except Exception as exc:
        raise ValueError(
            f"{what} expression {source!r} will not resolve: {exc}"
        ) from exc


def _eval_extent(source: str, binding: Mapping[str, Any]) -> int:
    return _positive_int(
        value_of(_eval_tracked(source, binding)),
        f"weight endpoint axis expression {source!r}",
    )


def _scalar_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Expression environment without parameter_tensor's list-valued `_shape`."""
    return {
        name: value
        for name, value in binding.items()
        if isinstance(value, (Dim, int, sp.Expr)) and not isinstance(value, bool)
    }


def _positive_int(value: Any, what: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{what} must be a positive integer, got {value!r}")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{what} must be a positive integer, got {value!r}") from exc
    if integer <= 0:
        raise ValueError(f"{what} must be a positive integer, got {value!r}")
    return int(integer)
