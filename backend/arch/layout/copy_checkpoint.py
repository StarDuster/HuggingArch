"""Checkpoint-header comparison for expert-bank storage.

An expert bank's weights lead with the expert axis (`shape: [n_expert_copies,
...]`), and the checkpoint stores that axis one of two ways, which the bank
declares (`expert_weight`): one entry per expert (`unrolled`) or one entry
whose leading axis is the experts (`leading_axis`). This module asks whether a
safetensors header encodes it the declared way.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from math import prod
from typing import Any, Optional, Tuple

from backend.arch import dims
from backend.arch.eval.dim import value_of

from backend.arch.eval.storage import leads_with_expert_axis

from . import weight_endpoints
from .persistent import SAVED_METADATA, STORED, share_of
from .weight_endpoints import _positive_int

MATCH = "match"
UNEXPLAINED = "unexplained"
UNVERIFIED = "unverified"
EVIDENCE_HEADER_OK = "header_ok"
EVIDENCE_PATTERN_MISSING = "pattern_missing"
EVIDENCE_COUNT_MISMATCH = "count_mismatch"
EVIDENCE_SINGLETON = "singleton"
COUNT_HEADER = "header"
COUNT_MISSING = "missing"


@dataclass(frozen=True)
class CheckpointEncodingVariant:
    """One physical checkpoint artifact variant in copy-domain terms.

    ``physical_elements`` and ``copy_axis_extent`` deliberately stop at the
    copy boundary. A quant format may factor its payload into blocks, scales,
    and packing axes; those are format contracts, while this oracle asks only
    whether the expert-copy domain is encoded as entries or as a leading axis.
    """

    physical_elements: int
    dtype: str
    entries: Optional[int]
    copy_axis_extent: Optional[int]
    copy_indexed: bool
    shape: Optional[Tuple[int, ...]] = None

    @staticmethod
    def canonical_dtype(dtype: str) -> str:
        normalized = str(dtype).upper()
        if normalized == "F8_E8M0FNU":
            return "F8_E8M0"
        return normalized

    def _core(self) -> tuple[int, str, Optional[int], bool]:
        return (
            self.physical_elements,
            self.canonical_dtype(self.dtype),
            self.copy_axis_extent,
            self.copy_indexed,
        )

    def _singleton_key(self) -> tuple[int, str, Optional[Tuple[int, ...]]]:
        shape = self.shape
        if shape is not None and self.copy_axis_extent == 1:
            shape = shape[1:]
        return (
            self.physical_elements,
            self.canonical_dtype(self.dtype),
            shape,
        )

    def expected_key(
        self, *, singleton: bool
    ) -> tuple[tuple[Any, ...], Optional[Tuple[int, ...]]]:
        if singleton:
            physical, dtype, shape = self._singleton_key()
            return (physical, dtype), shape
        return self._core(), self.shape

    def header_key(
        self, *, singleton: bool
    ) -> tuple[tuple[Any, ...], frozenset[Tuple[int, ...]]]:
        if self.shape is None:
            raise ValueError("checkpoint header variant has no physical shape")
        if not singleton:
            return self._core(), frozenset((self.shape,))
        core = (
            self.physical_elements,
            self.canonical_dtype(self.dtype),
        )
        shapes = {self.shape}
        if not self.copy_indexed and self.shape[:1] == (1,):
            shapes.add(self.shape[1:])
        return core, frozenset(shapes)


@dataclass(frozen=True)
class CheckpointCopyComparison:
    """One declared artifact family against one safetensors header pattern.

    ``verdict`` states what the comparison concluded (MATCH, UNEXPLAINED,
    or UNVERIFIED); ``evidence`` states why — what the header evidence
    showed (one of the ``EVIDENCE_*`` values above).
    """

    checkpoint_prefix: str
    artifact_key: str
    artifact_names: Tuple[str, ...]
    node_uids: Tuple[int, ...]
    pattern_names: Tuple[str, ...]
    checkpoint: str
    global_parts: int
    expected_entries: int
    header_entries: Optional[int]
    header_count_provenance: str
    expected_variants: Tuple[CheckpointEncodingVariant, ...]
    header_variants: Tuple[CheckpointEncodingVariant, ...]
    verdict: str
    evidence: str

@dataclass(frozen=True)
class ExpertArtifact:
    """One checkpoint artifact of an expert bank: a stored tensor whose axis
    0 is the expert axis (`storage.leads_with_expert_axis`), or one of its
    saved metadata siblings, which lead with the same axis."""

    node_uid: int
    tensor_index: int
    #: Which layer segment of that tensor (`formats.Representations`) — 0
    #: unless a layer-scoped override stores some of the node's layers
    #: differently, in which case each segment is its own set of checkpoint
    #: entries at its own dtype.
    segment_index: int
    surface: str
    metadata_index: Optional[int]
    artifact_name: str
    #: The bank's `expert_weight`: `unrolled` or `leading_axis`.
    checkpoint: str
    #: The experts the axis holds — its extent, as the walk resolved it.
    global_parts: int
    #: Entries the checkpoint holds: one per expert per instance when
    #: unrolled, one per instance otherwise.
    declared_checkpoint_entries: int
    #: The whole artifact in checkpoint axis order, the expert axis first.
    checkpoint_shape: Tuple[int, ...]


def expert_artifacts(analysis: Any) -> Tuple[ExpertArtifact, ...]:
    """Every expert-bank artifact the walk stored, with the encoding its
    bank declares. A tensor led by the expert axis with no `expert_weight` on
    itself or an ancestor is refused: the checkpoint names it one way or the
    other, and leaving it unsaid hides which."""
    context = weight_endpoints.placement_context(analysis)
    checkpoint_shapes = {
        (row.node_uid, row.tensor_index): row.checkpoint_shape
        for row in weight_endpoints.instantiate_analysis(analysis, context)
    }
    out = []
    for uid in sorted(analysis.storage.values):
        node = analysis.program.by_uid(uid)
        tensors = tuple(analysis.storage.values[uid])
        reps = tuple(analysis.formats.values.get(uid) or ())
        instances = int(value_of(analysis.instances.values.get(uid, 1)))
        for index, (tensor, reps_of_tensor) in enumerate(
                zip(tensors, reps, strict=True)):
            if not leads_with_expert_axis(tensor):
                continue
            encoding = _declared_encoding(node)
            parts = _positive_int(value_of(tensor.shape[0]),
                                  f"{tensor.name} expert axis")
            stored_shape = checkpoint_shapes.get((uid, index))
            if stored_shape is None:
                raise ValueError(
                    f"{tensor.name}: an expert-bank tensor with no "
                    "weight-axis endpoint")
            # One set of entries per layer segment: the copies it covers,
            # each at its own saved format, so a bank stored FP8 on most of
            # its layers and BF16 on the rest is two groups of entries and
            # not one average.
            for segment_index, (rep, share) in enumerate(
                    reps_of_tensor.with_shares()):
                entries = share_of(instances, share, tensor.name) * (
                    parts if encoding == "unrolled" else 1)
                out.append(ExpertArtifact(
                    uid, index, segment_index, STORED, None, tensor.name,
                    encoding, parts, entries, tuple(stored_shape)))
                for metadata_index, metadata in enumerate(rep.saved.metadata):
                    out.append(ExpertArtifact(
                        uid, index, segment_index, SAVED_METADATA,
                        metadata_index, metadata.name, encoding, parts,
                        entries, _shape(metadata, metadata.name)))
    return tuple(out)


def _declared_encoding(node: Any) -> str:
    """`expert_weight` on the node or the nearest ancestor declaring it."""
    cursor = node
    while cursor is not None:
        if cursor.expert_weight is not None:
            return cursor.expert_weight
        cursor = cursor.parent
    raise ValueError(
        f"{node.display_path}: stores a weight led by the expert axis and "
        "neither it nor an enclosing call declares `expert_weight:` "
        "(`unrolled` or `leading_axis`), how the checkpoint stores that axis")


def _shape(tensor: Any, what: str) -> Tuple[int, ...]:
    return tuple(
        _positive_int(value_of(extent), f"{what} axis extent")
        for extent in tensor.shape
    )


@dataclass(frozen=True)
class _ExpectedCheckpointArtifact:
    endpoint: ExpertArtifact
    prefix: str
    key: str
    dtype: str
    payload_elements: int
    payload_shape: Optional[Tuple[int, ...]]


@dataclass(frozen=True)
class _ExpectedArtifactGroup:
    """Every declared artifact that folds into one checkpoint artifact key."""

    prefix: str
    key: str
    items: Tuple[_ExpectedCheckpointArtifact, ...]
    checkpoint: str
    global_parts: int

    @classmethod
    def build(
        cls,
        prefix: str,
        key: str,
        items: Tuple[_ExpectedCheckpointArtifact, ...],
    ) -> "_ExpectedArtifactGroup":
        checkpoints = {item.endpoint.checkpoint for item in items}
        global_parts_set = {item.endpoint.global_parts for item in items}
        if len(checkpoints) != 1 or len(global_parts_set) != 1:
            raise ValueError(
                f"checkpoint artifact {key!r} merges incompatible copy contracts: "
                f"checkpoint={sorted(checkpoints)!r}, parts={sorted(global_parts_set)!r}"
            )
        return cls(
            prefix=prefix,
            key=key,
            items=tuple(items),
            checkpoint=next(iter(checkpoints)),
            global_parts=next(iter(global_parts_set)),
        )

    @property
    def expected_entries(self) -> int:
        return sum(item.endpoint.declared_checkpoint_entries for item in self.items)

    def artifact_names(self) -> Tuple[str, ...]:
        return tuple(sorted({item.endpoint.artifact_name for item in self.items}))

    def node_uids(self) -> Tuple[int, ...]:
        return tuple(sorted({item.endpoint.node_uid for item in self.items}))

    def expected_variants(self) -> Tuple[CheckpointEncodingVariant, ...]:
        counts: Counter[
            tuple[int, str, Optional[int], bool, Optional[Tuple[int, ...]]]
        ] = Counter()
        for item in self.items:
            entries = item.endpoint.declared_checkpoint_entries
            if entries < 0:
                raise ValueError(
                    f"{item.endpoint.artifact_name}: negative checkpoint entries {entries}"
                )
            if not entries:
                continue
            leading = self.checkpoint == "leading_axis"
            physical_elements = item.payload_elements * (
                self.global_parts if leading else 1
            )
            shape = None
            if item.payload_shape is not None:
                shape = (
                    (self.global_parts, *item.payload_shape)
                    if leading
                    else item.payload_shape
                )
            counts[
                (
                    physical_elements,
                    CheckpointEncodingVariant.canonical_dtype(item.dtype),
                    self.global_parts if leading else None,
                    not leading,
                    shape,
                )
            ] += entries
        return tuple(
            CheckpointEncodingVariant(
                physical_elements=physical_elements,
                dtype=dtype,
                entries=entries,
                copy_axis_extent=copy_axis_extent,
                copy_indexed=copy_indexed,
                shape=shape,
            )
            for (
                physical_elements,
                dtype,
                copy_axis_extent,
                copy_indexed,
                shape,
            ), entries in sorted(
                counts.items(),
                key=lambda item: (
                    item[0][0],
                    item[0][1],
                    item[0][2] is not None,
                    item[0][2] or 0,
                    item[0][3],
                    item[0][4] is None,
                    item[0][4] or (),
                ),
            )
        )


@dataclass(frozen=True)
class _EncodingMatch:
    """Declared-versus-header encoding evidence for one artifact family."""

    expected: Tuple[CheckpointEncodingVariant, ...]
    header: Tuple[CheckpointEncodingVariant, ...]
    singleton: bool

    def geometry_matches(self) -> bool:
        exact_shapes: dict[tuple[Any, ...], set[Tuple[int, ...]]] = {}
        opaque_cores: set[tuple[Any, ...]] = set()
        header_shape_candidates: dict[
            tuple[Any, ...], list[frozenset[Tuple[int, ...]]]
        ] = {}
        for variant in self.expected:
            core, shape = variant.expected_key(singleton=self.singleton)
            if shape is None:
                opaque_cores.add(core)
            else:
                exact_shapes.setdefault(core, set()).add(shape)
        for variant in self.header:
            core, shapes = variant.header_key(singleton=self.singleton)
            header_shape_candidates.setdefault(core, []).append(shapes)

        expected_cores = set(exact_shapes) | opaque_cores
        if expected_cores != set(header_shape_candidates):
            return False
        for core in expected_cores:
            expected = exact_shapes.get(core, set())
            candidates = header_shape_candidates[core]
            if any(
                not any(shape in choices for choices in candidates)
                for shape in expected
            ):
                return False
            if core not in opaque_cores and any(
                choices.isdisjoint(expected) for choices in candidates
            ):
                return False
        return True

    def counts_match(self) -> bool:
        exact_counts: Counter[tuple[tuple[Any, ...], Tuple[int, ...]]] = Counter()
        opaque_counts: Counter[tuple[Any, ...]] = Counter()
        header_supplies: dict[
            tuple[Any, ...], list[tuple[frozenset[Tuple[int, ...]], int]]
        ] = {}
        for variant in self.expected:
            if variant.entries is None:
                raise ValueError("expected checkpoint variant has unknown count")
            core, shape = variant.expected_key(singleton=self.singleton)
            if shape is None:
                opaque_counts[core] += variant.entries
            else:
                exact_counts[(core, shape)] += variant.entries
        for variant in self.header:
            if variant.entries is None:
                raise ValueError("checkpoint header variant has unknown count")
            core, shapes = variant.header_key(singleton=self.singleton)
            header_supplies.setdefault(core, []).append((shapes, variant.entries))

        cores = set(opaque_counts) | {core for core, _shape in exact_counts}
        if cores != set(header_supplies):
            return False
        for core in cores:
            demands = Counter(
                {
                    shape: entries
                    for (demand_core, shape), entries in exact_counts.items()
                    if demand_core == core
                }
            )
            supplies = header_supplies[core]
            if sum(entries for _shapes, entries in supplies) != (
                sum(demands.values()) + opaque_counts[core]
            ):
                return False
            if not self._shape_demands_can_be_satisfied(supplies, demands):
                return False
        return True

    @staticmethod
    def _shape_demands_can_be_satisfied(
        supplies: list[tuple[frozenset[Tuple[int, ...]], int]],
        demands: Counter[Tuple[int, ...]],
    ) -> bool:
        if not demands:
            return True
        shapes = tuple(demands)
        source = 0
        supply_offset = 1
        demand_offset = supply_offset + len(supplies)
        sink = demand_offset + len(shapes)
        residual: dict[int, dict[int, int]] = defaultdict(dict)

        def add_edge(left: int, right: int, capacity: int) -> None:
            residual[left][right] = capacity
            residual[right].setdefault(left, 0)

        for index, (choices, entries) in enumerate(supplies):
            supply_node = supply_offset + index
            add_edge(source, supply_node, entries)
            for shape_index, shape in enumerate(shapes):
                if shape in choices:
                    add_edge(supply_node, demand_offset + shape_index, entries)
        for shape_index, shape in enumerate(shapes):
            add_edge(demand_offset + shape_index, sink, demands[shape])

        flow = 0
        target = sum(demands.values())
        while flow < target:
            parent = {source: -1}
            queue = deque((source,))
            while queue and sink not in parent:
                left = queue.popleft()
                for right, capacity in residual[left].items():
                    if capacity > 0 and right not in parent:
                        parent[right] = left
                        queue.append(right)
            if sink not in parent:
                return False
            increment = target - flow
            cursor = sink
            while cursor != source:
                previous = parent[cursor]
                increment = min(increment, residual[previous][cursor])
                cursor = previous
            cursor = sink
            while cursor != source:
                previous = parent[cursor]
                residual[previous][cursor] -= increment
                residual[cursor][previous] += increment
                cursor = previous
            flow += increment
        return True


@dataclass(frozen=True)
class _CheckpointNamespace:
    """Artifact keys under one stack prefix.

    Both sides carry the stack's full declared address (`model.layers.X.`),
    so a key is a name with that prefix taken off — the same on both."""

    prefix: str

    def tail(self, name: str) -> Optional[str]:
        name = str(name)
        return name[len(self.prefix):] if name.startswith(self.prefix) else None

    def declared_key(self, name: str) -> str:
        tail = self.tail(name)
        if tail is None:
            raise ValueError(
                f"declared checkpoint artifact {name!r} does not start in stack "
                f"prefix {self.prefix!r}"
            )
        return tail


@dataclass(frozen=True)
class _CheckpointHeader:
    """Header-side evidence: the tensors a declared artifact name selects."""

    header: Any

    def matching_pattern(
        self, key: str, namespace: _CheckpointNamespace
    ) -> Optional[str]:
        """The artifact's full name under the namespace, when it selects a
        header tensor."""
        name = namespace.prefix + key
        return name if self.header.select(name) else None

    def variants(
        self,
        pattern_name: str,
        *,
        namespace: _CheckpointNamespace,
        checkpoint: str,
    ) -> Tuple[CheckpointEncodingVariant, ...]:
        tail = namespace.tail(pattern_name)
        if tail is None:
            raise ValueError(
                f"checkpoint pattern {pattern_name!r} does not contain prefix "
                f"{namespace.prefix!r}"
            )
        copy_indexed = "X" in tail.split(".")
        return tuple(
            CheckpointEncodingVariant(
                physical_elements=prod(group.shape),
                dtype=CheckpointEncodingVariant.canonical_dtype(group.dtype),
                entries=count,
                copy_axis_extent=(
                    group.shape[0] if checkpoint == "leading_axis" else None
                ),
                copy_indexed=copy_indexed,
                shape=tuple(group.shape),
            )
            for group, count in self.header.select(pattern_name).parts
        )


def compare_checkpoint(
    step: Any, header: Any
) -> Tuple[CheckpointCopyComparison, ...]:
    """Compare declared copy encodings with safetensors header artifacts.

    ``header`` (`hf.header.SafetensorsHeader`) is explicit input because
    neither the spec nor the evaluated step owns a checkpoint; replaying
    materialization here would only compare ``Analysis`` with itself.

    A declared name selects every header tensor of every block variant that
    stores it: Step-3.7's 11- and 31-layer MoE templates therefore meet one
    42-entry selection, as do GPT-OSS's two 12-layer templates. The comparison
    groups declarations before checking counts and physical slot conservation.
    """
    _validated_deployment(step)
    header = _CheckpointHeader(header)

    analysis = step.analysis
    grouped: dict[tuple[str, str], list[_ExpectedCheckpointArtifact]] = (
        defaultdict(list)
    )
    for endpoint in expert_artifacts(analysis):
        node = analysis.program.by_uid(endpoint.node_uid)
        prefix = analysis.program.checkpoint_prefix(node)
        key = _CheckpointNamespace(prefix).declared_key(endpoint.artifact_name)
        dtype, payload_elements, payload_shape = _checkpoint_payload_fact(
            analysis, endpoint
        )
        grouped[(prefix, key)].append(
            _ExpectedCheckpointArtifact(
                endpoint=endpoint,
                prefix=prefix,
                key=key,
                dtype=dtype,
                payload_elements=payload_elements,
                payload_shape=payload_shape,
            )
        )

    out = []
    for (prefix, key), items in sorted(grouped.items()):
        group = _ExpectedArtifactGroup.build(prefix, key, tuple(items))
        namespace = _CheckpointNamespace(prefix)
        expected_variants = group.expected_variants()
        expected_entries = group.expected_entries
        singleton = group.global_parts == 1
        pattern_name = header.matching_pattern(key, namespace)
        common = dict(
            checkpoint_prefix=prefix,
            artifact_key=key,
            artifact_names=group.artifact_names(),
            node_uids=group.node_uids(),
            pattern_names=() if pattern_name is None else (pattern_name,),
            checkpoint=group.checkpoint,
            global_parts=group.global_parts,
            expected_entries=expected_entries,
            expected_variants=expected_variants,
        )

        if pattern_name is None:
            out.append(
                CheckpointCopyComparison(
                    **common,
                    header_entries=0,
                    header_count_provenance=COUNT_MISSING,
                    header_variants=(),
                    verdict=UNVERIFIED,
                    evidence=EVIDENCE_PATTERN_MISSING,
                )
            )
            continue

        header_variants = header.variants(
            pattern_name,
            namespace=namespace,
            checkpoint=group.checkpoint,
        )
        header_entries = sum(variant.entries for variant in header_variants)
        match = _EncodingMatch(
            expected_variants, header_variants, singleton=singleton
        )
        geometry_matches = match.geometry_matches()
        counts_match = match.counts_match()
        if not geometry_matches:
            verdict, evidence = UNEXPLAINED, EVIDENCE_HEADER_OK
        elif not counts_match:
            verdict, evidence = UNEXPLAINED, EVIDENCE_COUNT_MISMATCH
        elif group.global_parts == 1:
            verdict, evidence = UNVERIFIED, EVIDENCE_SINGLETON
        else:
            verdict, evidence = MATCH, EVIDENCE_HEADER_OK
        out.append(
            CheckpointCopyComparison(
                **common,
                header_entries=header_entries,
                header_count_provenance=COUNT_HEADER,
                header_variants=header_variants,
                verdict=verdict,
                evidence=evidence,
            )
        )
    return tuple(out)


def _validated_deployment(step: Any) -> Any:
    pc = getattr(step, "parallel", None)
    if pc is None:
        raise ValueError(
            "weight-copy endpoints require a step with a deployment"
        )
    errors = pc.validate()
    if errors:
        raise ValueError(
            "weight-copy endpoints require a valid deployment: "
            + "; ".join(errors)
        )
    if step.phase not in dims.PHASES:
        raise ValueError(
            f"unknown phase {step.phase!r}; expected one of {list(dims.PHASES)}"
        )
    return pc


def _checkpoint_payload_fact(
    analysis: Any, endpoint: ExpertArtifact
) -> Tuple[str, int, Optional[Tuple[int, ...]]]:
    """One artifact's saved dtype, per-expert slot count, and per-expert
    payload shape (checkpoint order, the expert axis taken off).

    Deliberately module-level: tests substitute this oracle to exercise
    mixed exact/opaque payload evidence.
    """
    tensor = analysis.storage.values[endpoint.node_uid][endpoint.tensor_index]
    rep = analysis.formats.values[endpoint.node_uid][
        endpoint.tensor_index].segments[endpoint.segment_index]
    if endpoint.surface == STORED:
        logical_elements = _positive_int(
            value_of(tensor.elements), f"{tensor.name} logical element count"
        )
        packing = _positive_int(
            rep.saved.packing_factor, f"{tensor.name} saved packing factor"
        )
        if logical_elements % packing:
            raise ValueError(
                f"{tensor.name}: logical elements {logical_elements} are not "
                f"divisible by saved packing factor {packing}"
            )
        physical_elements = logical_elements // packing
        raw_dtype = rep.saved.storage_dtype
        payload_shape = _saved_payload_shape(
            endpoint.artifact_name, endpoint.checkpoint_shape[1:], rep.saved
        )
    elif endpoint.surface == SAVED_METADATA:
        if endpoint.metadata_index is None:
            raise ValueError(f"{endpoint.artifact_name}: metadata index is missing")
        metadata = rep.saved.metadata[endpoint.metadata_index]
        physical_elements = _positive_int(
            value_of(metadata.elements),
            f"{endpoint.artifact_name} saved metadata elements",
        )
        raw_dtype = metadata.dtype
        whole = _saved_metadata_payload_shape(
            endpoint.artifact_name,
            metadata,
            rep.saved,
        )
        payload_shape = None if whole is None else whole[1:]
    else:
        raise ValueError(
            f"{endpoint.artifact_name}: {endpoint.surface!r} is not a "
            "checkpoint artifact"
        )
    if not isinstance(raw_dtype, str) or not raw_dtype:
        raise ValueError(f"{endpoint.artifact_name}: saved dtype is empty")
    dtype = raw_dtype
    if physical_elements % endpoint.global_parts:
        raise ValueError(
            f"{endpoint.artifact_name}: physical elements {physical_elements} "
            f"are not divisible by its {endpoint.global_parts} experts"
        )
    physical_elements //= endpoint.global_parts
    if payload_shape is not None and prod(payload_shape) != physical_elements:
        raise ValueError(
            f"{endpoint.artifact_name}: checkpoint payload shape "
            f"{payload_shape!r} has {prod(payload_shape)} slots, expected "
            f"{physical_elements}"
        )
    return dtype, physical_elements, payload_shape


def _saved_payload_shape(
    artifact_name: str, logical_shape: Tuple[int, ...], saved: Any
) -> Optional[Tuple[int, ...]]:
    packing = _positive_int(
        saved.packing_factor, f"{artifact_name} saved packing factor"
    )
    if packing == 1:
        return logical_shape
    # packing_factor declares slot cardinality, not a physical packing axis.
    # Exact packed shape needs a format-level physical-shape declaration.
    return None


def _saved_metadata_payload_shape(
    artifact_name: str,
    metadata: Any,
    saved: Any,
) -> Optional[Tuple[int, ...]]:
    siblings = tuple(getattr(saved.scheme, "metadata_siblings", ()) or ())
    declarations = tuple(
        sibling
        for sibling in siblings
        if sibling.suffix == metadata.logical_suffix
    )
    if len(declarations) != 1:
        raise ValueError(
            f"{artifact_name}: saved metadata has {len(declarations)} format "
            "shape declarations"
        )
    declaration = declarations[0]
    if declaration.shape is None and not declaration.shape_per_strategy:
        # A cardinality expression determines slot count but not axis order.
        return None
    return _shape(metadata, "checkpoint saved metadata")
