"""The layout algebra — one atom, sum, product, bottom.

    AxisLayout ::= Block(group, replica_cells)   the atom: an owner partition
                 | Segmented(segments)   sum  (⊕): per-segment concatenation
                 | Product(cuts)         product (⊗): nested subdivision
                 | Flattened(factors)    row-major flatten of factored axes
                 | Unknown(reason)       bottom: an error at any consumer

No runtime ingredient lives here: routing is a pair of axis-swap ops plus
an edge-side `ExpectedRoutedLoad`, and partial sums live on `Value` — both
answer "what is this memory right now" and neither pollutes geometry.

Canonical form consists of exactly the unit laws: a single-segment
Segmented is its child, a single-cell Block with no padding is Replicated
(sugar), a single-cut Product is that cut. The factories canonicalize;
the raw dataclasses REJECT non-canonical and ill-formed construction in
`__post_init__`, so there is no bypass — well-formedness is a property of
the type, not of one entry path. Equality is structural comparison of
canonical trees — groups compare by explicit membership, replica_cells by the
partition data itself (a partition has no second spelling), Product cut
ORDER is semantic (the nesting is the index map) and never reordered.
Isomorphic-segment Segmenteds do NOT reduce: per-segment blocks locally
concatenated and one block over the whole axis are two different owner
maps — the concat counterexample.

Non-divisible extents are LOUD everywhere: every division is exact or
raises — a named non-divisible policy must have chosen before layout
construction. `Unknown` is bottom: every consumer entry point rejects it.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union

import sympy as sp

from backend.arch.eval.dim import (apply, ceil_div, require,
                                   require_multiple, same_law,
                                   settled_non_negative, varies)
from backend.arch.eval.refusal import LayoutRefusal

from .groups import ProcessGroupFamily, joint_components, orthogonal, same

Cells = Tuple[Tuple[int, ...], ...]


class UniformFiberError(ValueError):
    """Cuts of one tensor do not tile its ranks."""


class IndivisibleSequence(LayoutRefusal):
    """A step's tokens do not split evenly over the sequence world that cuts
    the residual stream (prefill context parallelism, or tensor-parallel
    sequence parallelism).

    The deployment is fine; the length is not. A caller that chose the length
    can say so — `reason` is the stable code it reads, `extent` / `world` /
    `degree` the facts it names."""

    reason = "sequence_not_divisible"

    def __init__(self, where: str, extent: int, world: str, degree: int):
        self.extent = extent
        self.world = world
        self.degree = degree
        super().__init__(
            f"{where}: token extent {extent} does not divide the sequence "
            f"world degree {degree}")


def _is_int(value) -> bool:
    """A whole extent: an int, or an expression in the workload's symbols
    (whose integrality the walk's gates assumed — `dim.require_multiple`)."""
    return (isinstance(value, int) and not isinstance(value, bool)) \
        or varies(value)


def whole_interval(start: Any, stop: Any, extent: Any) -> bool:
    """Whether `[start, stop)` is the whole axis by its declaration: `start`
    the law 0 and `stop` the axis's own law.

    Asked of the laws, not of one point's numbers. A slice whose bounds
    come to the whole axis at some points only (a pool of complete windows,
    `floor(s/m)*m`, that tiles `s` exactly when `m` divides it) is a proper
    subinterval: its topology cannot depend on the workload, and a cut axis
    is gathered before it, as the frameworks run it (sglang's DeepSeek-V4
    compressor gathers every rank's tokens before it pools windows,
    `layers/attention/dsv4/compressor.py:425-433`)."""
    return same_law(start, 0) and same_law(stop, extent)


def _exact_div(extent: int, num_shards: int, what: str) -> int:
    # Zero is an extent: an empty axis (DSv4's complete windows in a step
    # shorter than one window) divides into empty shards.
    _assume_non_negative(extent)
    require_multiple(
        extent, num_shards,
        lambda value: ValueError(
            f"{what}: extent {value} not divisible by {num_shards} and no "
            "padding declared — a named non-divisible policy must choose "
            "before construction"))
    return _divided(extent, num_shards)


def _divided(extent: int, num_shards: int) -> int:
    """A division already known exact, in both domains: `//` on a number,
    `/` on an expression (where `//` would leave a `floor` nobody needs)."""
    return apply(operator.truediv, operator.floordiv, extent, num_shards)


def _check_segment_total(layout: "Segmented", extent: int) -> None:
    total = sum(ext for _, ext, _ in layout.segments)
    if total != extent:
        raise ValueError(f"Segmented tiles {total} elements but was "
                         f"consumed with extent {extent}")


def _check_index(index: int, extent: int) -> None:
    if extent <= 0:
        raise ValueError(f"extent must be positive, got {extent}")
    if not (0 <= index < extent):
        raise IndexError(index)


def _factor_indices(layout: "Flattened", index: int,
                    extent: int) -> Tuple[int, ...]:
    """Decode one row-major flattened index into factor coordinates."""
    _check_index(index, extent)
    expected = 1
    for factor_extent in layout.factor_extents:
        expected *= factor_extent
    if extent != expected:
        raise ValueError(
            f"Flattened factors multiply to {expected} but were consumed "
            f"with extent {extent}")
    remaining = index
    reversed_indices = []
    for factor_extent in reversed(layout.factor_extents):
        reversed_indices.append(remaining % factor_extent)
        remaining //= factor_extent
    return tuple(reversed(reversed_indices))


@dataclass(frozen=True)
class Replicated:
    """The single-cell Block, canonicalized. Carries its SCOPE: replication
    is always relative to a definite rank set (v2's lesson — a tensor
    replicated over one process-group family says nothing about ranks
    outside it). Scope takes part in structural equality."""

    scope: frozenset

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", frozenset(self.scope))
        if not self.scope:
            raise ValueError("replication over an empty scope is "
                             "meaningless")

    def logical_local_extent(self, extent: int) -> int:
        return extent

    def physical_local_extent(self, extent: int) -> int:
        return extent

    def owner_ranks(self, index: int, extent: int,
                    universe: frozenset = frozenset()) -> frozenset:
        del index, extent, universe
        return self.scope

    def local_index(self, rank: int, index: int, extent: int,
                    universe: frozenset) -> Optional[int]:
        del extent, universe
        return index if rank in self.scope else None

    def slice_axis(self, start: int, stop: int, extent: int) -> "AxisLayout":
        del start, stop, extent
        return self


@dataclass(frozen=True)
class Block:
    """The atom: an equal-celled, ordered, covering partition of in-group
    coordinates, one pattern governing every group of the family. Cell j
    co-owns index block j; a cell larger than one IS replication."""

    group: ProcessGroupFamily
    replica_cells: Cells
    pad_to: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "replica_cells",
                           tuple(tuple(c) for c in self.replica_cells))
        if not self.replica_cells or not self.replica_cells[0]:
            raise ValueError("replica_cells must be a non-empty partition")
        size = len(self.replica_cells[0])
        if any(len(c) != size for c in self.replica_cells):
            raise ValueError("replica_cells must be equal-sized")
        flat = sorted(c for cell in self.replica_cells for c in cell)
        if flat != list(range(self.group.group_size)):
            raise ValueError("replica_cells must cover the group's coordinates "
                             "exactly")
        if self.pad_to is not None and (
            not _is_int(self.pad_to) or self.pad_to <= 0
        ):
            raise ValueError("pad_to must be a positive integer")
        if len(self.replica_cells) == 1 and self.pad_to is None:
            raise ValueError("a single-cell unpadded Block is Replicated — "
                             "use the block() factory")

    @property
    def num_shards(self) -> int:
        return len(self.replica_cells)

    @property
    def replication_degree(self) -> int:
        return len(self.replica_cells[0])

    def is_sharded_on(self, family: ProcessGroupFamily) -> bool:
        """True when this block is a one-rank-per-cell cut of `family`."""
        return self.replication_degree == 1 and same(self.group, family)

    def shard_of_rank(self, rank: int) -> int:
        coord = self.group.coordinate(rank)
        for j, cell in enumerate(self.replica_cells):
            if coord in cell:
                return j
        raise AssertionError("replica_cells cover all coordinates by construction")

    def padded_global_extent(self, extent: int) -> int:
        if self.pad_to is None:
            _exact_div(extent, self.num_shards, "Block")
            return extent
        quantum = self.num_shards * self.pad_to
        return ceil_div(extent, quantum) * quantum

    def physical_local_extent(self, extent: int) -> int:
        # Exact by construction: the padded extent is whole quanta.
        return _divided(self.padded_global_extent(extent), self.num_shards)

    def gathers_padding(self, gathered: int, extent: int) -> bool:
        """Whether a gather of this cut that produced `gathered` elements
        over a logical `extent` brought back exactly its padding: a padded
        cut gathers `padded_global_extent`, and the surplus is a local trim,
        not a geometry mismatch."""
        return (self.pad_to is not None and gathered != extent
                and gathered == self.padded_global_extent(extent))

    def valid_counts_per_shard(self, extent: int) -> Tuple[int, ...]:
        """Valid (non-padding) elements per part — tail num_shards run short.
        Never consumed per rank: SOL pricing reads physical (the uneven≈pad
        assumption); this enters only the per-part-deduplicated global
        conservation Σ = extent."""
        phys = self.physical_local_extent(extent)
        return tuple(max(0, min(phys, extent - j * phys))
                     for j in range(self.num_shards))

    def logical_local_extent(self, extent: int) -> int:
        if self.pad_to is not None:
            raise ValueError("padded Block has per-part effective counts, "
                             "not one per-rank logical extent")
        return _exact_div(extent, self.num_shards, "Block")

    def owner_ranks(self, index: int, extent: int,
                    universe: frozenset = frozenset()) -> frozenset:
        del universe
        _check_index(index, extent)
        phys = self.physical_local_extent(extent)
        j = index // phys
        out = set()
        for g in self.group.groups:
            out.update(g[c] for c in self.replica_cells[j])
        return frozenset(out)

    def local_index(self, rank: int, index: int, extent: int,
                    universe: frozenset) -> Optional[int]:
        del universe
        if rank not in self.owner_ranks(index, extent):
            return None
        phys = self.physical_local_extent(extent)
        return index - (index // phys) * phys

    def slice_axis(self, start: int, stop: int, extent: int) -> "AxisLayout":
        if not whole_interval(start, stop, extent):
            raise ValueError(
                "slicing a proper subinterval of a rank-cut axis would "
                "leave some ranks empty — rejected; segment-internal "
                "slices go through Segmented")
        return self


@dataclass(frozen=True)
class Segmented:
    """Sum: per-segment layouts, locally concatenated in segment order."""

    segments: Tuple[Tuple[int, int, "AxisLayout"], ...]  # (offset, extent, child)

    def __post_init__(self) -> None:
        if len(self.segments) < 2:
            raise ValueError("a single-segment Segmented is its child — "
                             "use the segmented() factory")
        offset = 0
        for off, ext, child in self.segments:
            if (
                not _is_int(off)
                or not _is_int(ext)
                or off != offset
                or ext <= 0
            ):
                raise ValueError("segments must tile the axis contiguously "
                                 "from 0 with positive extents")
            if isinstance(child, (Unknown, Segmented)):
                raise ValueError("Unknown/nested Segmented may not be a "
                                 "segment child")
            physical_local_extent(child, ext)
            offset += ext

    def logical_local_extent(self, extent: int) -> int:
        _check_segment_total(self, extent)
        return sum(logical_local_extent(child, ext)
                   for _, ext, child in self.segments)

    def physical_local_extent(self, extent: int) -> int:
        _check_segment_total(self, extent)
        return sum(physical_local_extent(child, ext)
                   for _, ext, child in self.segments)

    def owner_ranks(self, index: int, extent: int,
                    universe: frozenset) -> frozenset:
        _check_segment_total(self, extent)
        for off, ext, child in self.segments:
            if off <= index < off + ext:
                return owner_ranks(child, index - off, ext, universe)
        raise IndexError(index)

    def local_index(self, rank: int, index: int, extent: int,
                    universe: frozenset) -> Optional[int]:
        _check_segment_total(self, extent)
        base = 0
        for off, ext, child in self.segments:
            if off <= index < off + ext:
                sub = local_index(child, rank, index - off, ext, universe)
                return None if sub is None else base + sub
            base += physical_local_extent(child, ext)
        raise IndexError(index)

    def slice_axis(self, start: int, stop: int, extent: int) -> "AxisLayout":
        _check_segment_total(self, extent)
        pieces = []
        rebased = 0
        for off, ext, child in self.segments:
            lo, hi = max(start, off), min(stop, off + ext)
            if lo >= hi:
                continue
            sub = slice_axis(child, lo - off, hi - off, ext)
            pieces.append((rebased, hi - lo, sub))
            rebased += hi - lo
        return segmented(pieces)


@dataclass(frozen=True)
class Product:
    """Product: ordered outer→inner nested subdivision. The order IS the
    index map and takes part in equality. Static Blocks only, no padding
    inside (physical extent would lose uniqueness), pairwise-orthogonal
    groups required — sufficiency comes from the tensor-level uniform-fiber
    check, which lives with TensorLayout."""

    cuts: Tuple["Block", ...]

    def __post_init__(self) -> None:
        if len(self.cuts) < 2:
            raise ValueError("a single-cut Product is that cut — use the "
                             "product() factory")
        for c in self.cuts:
            if not isinstance(c, Block):
                raise ValueError("Product cuts must be static Blocks "
                                 "(Segmented/nested Product reserved)")
            if c.pad_to is not None:
                raise ValueError("a padded Block owns its axis alone — "
                                 "physical extent inside a Product has no "
                                 "unique meaning")
        for i in range(len(self.cuts)):
            for j in range(i + 1, len(self.cuts)):
                if not orthogonal(self.cuts[i].group, self.cuts[j].group):
                    raise ValueError(
                        f"Product cuts {i} and {j} are not orthogonal")

    def logical_local_extent(self, extent: int) -> int:
        for cut in self.cuts:
            extent = _exact_div(extent, cut.num_shards, "Product")
        return extent

    def physical_local_extent(self, extent: int) -> int:
        return self.logical_local_extent(extent)

    @property
    def num_shards(self) -> int:
        """The product of the ordered cuts' shard counts."""
        value = 1
        for cut in self.cuts:
            value *= cut.num_shards
        return value

    def owner_ranks(self, index: int, extent: int,
                    universe: frozenset) -> frozenset:
        del universe
        out = self.cuts[0].group.ranks()
        span = extent
        rel = index
        for cut in self.cuts:
            span = _exact_div(span, cut.num_shards, "Product")
            j = rel // span
            rel -= j * span
            cell_owned = set()
            for g in cut.group.groups:
                cell_owned.update(g[c] for c in cut.replica_cells[j])
            out = out & frozenset(cell_owned)
        return out

    def local_index(self, rank: int, index: int, extent: int,
                    universe: frozenset) -> Optional[int]:
        if rank not in self.owner_ranks(index, extent, universe):
            return None
        span = extent
        rel = index
        for cut in self.cuts:
            span = _exact_div(span, cut.num_shards, "Product")
            rel -= (rel // span) * span
        return rel

    def slice_axis(self, start: int, stop: int, extent: int) -> "AxisLayout":
        if not whole_interval(start, stop, extent):
            raise ValueError(
                "slicing a proper subinterval of a rank-cut axis would "
                "leave some ranks empty — rejected; segment-internal "
                "slices go through Segmented")
        return self


@dataclass(frozen=True)
class Flattened:
    """One logical axis retaining a row-major factored ownership map.

    Axis-carried storage can flatten independent semantic dimensions into a
    rank-1 checkpoint tensor.  An inner cut then owns repeated intervals, not
    one contiguous Block, so erasing the factors would change the index map.
    This constructor keeps that map explicit without pretending it is a
    normal reshape accepted by the transfer table.
    """

    factors: Tuple["AxisLayout", ...]
    factor_extents: Tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "factors", tuple(self.factors))
        object.__setattr__(self, "factor_extents", tuple(self.factor_extents))
        if len(self.factors) < 2 or len(self.factors) != len(
                self.factor_extents):
            raise ValueError(
                "Flattened requires at least two equally counted factors "
                "and extents")
        if any(not _is_int(extent) or extent <= 0
               for extent in self.factor_extents):
            raise ValueError("Flattened factor extents must be positive ints")
        if any(isinstance(factor, (Unknown, Flattened))
               for factor in self.factors):
            raise ValueError("Unknown/nested Flattened factors are invalid")
        # Reuse the tensor-level joint-coverage theorem for the factors that
        # this logical axis folds together.
        tensor_layout(self.factors)

    def logical_local_extent(self, extent: int) -> int:
        expected = 1
        local = 1
        for factor, factor_extent in zip(
                self.factors, self.factor_extents, strict=True):
            expected *= factor_extent
            local *= logical_local_extent(factor, factor_extent)
        if extent != expected:
            raise ValueError(
                f"Flattened factors multiply to {expected} but were "
                f"consumed with extent {extent}")
        return local

    def physical_local_extent(self, extent: int) -> int:
        expected = 1
        local = 1
        for factor, factor_extent in zip(
                self.factors, self.factor_extents, strict=True):
            expected *= factor_extent
            local *= physical_local_extent(factor, factor_extent)
        if extent != expected:
            raise ValueError(
                f"Flattened factors multiply to {expected} but were "
                f"consumed with extent {extent}")
        return local

    def owner_ranks(self, index: int, extent: int,
                    universe: frozenset) -> frozenset:
        indices = _factor_indices(self, index, extent)
        owners = universe
        for factor, factor_index, factor_extent in zip(
                self.factors, indices, self.factor_extents, strict=True):
            owners = owners & owner_ranks(
                factor, factor_index, factor_extent, universe)
        return owners

    def local_index(self, rank: int, index: int, extent: int,
                    universe: frozenset) -> Optional[int]:
        indices = _factor_indices(self, index, extent)
        local = 0
        for factor, factor_index, factor_extent in zip(
                self.factors, indices, self.factor_extents, strict=True):
            sub = local_index(
                factor, rank, factor_index, factor_extent, universe)
            if sub is None:
                return None
            local = (
                local * physical_local_extent(factor, factor_extent) + sub)
        return local

    def slice_axis(self, start: int, stop: int, extent: int) -> "AxisLayout":
        if not whole_interval(start, stop, extent):
            raise ValueError(
                "slicing a flattened factored axis requires an explicit "
                "factor-aware index map")
        return self


@dataclass(frozen=True)
class Unknown:
    reason: str = ""


AxisLayout = Union[
    Replicated, Block, Segmented, Product, Flattened, Unknown]

def replicated(scope) -> Replicated:
    return Replicated(scope=frozenset(scope))


def block(group: ProcessGroupFamily, replica_cells: Cells,
          pad_to: Optional[int] = None) -> AxisLayout:
    """Canonicalizing constructor for the atom (unit law: single cell,
    no padding → Replicated)."""
    replica_cells = tuple(tuple(c) for c in replica_cells)
    if len(replica_cells) == 1 and pad_to is None:
        flat = sorted(c for cell in replica_cells for c in cell)
        if flat != list(range(group.group_size)):
            raise ValueError("replica_cells must cover the group's coordinates "
                             "exactly")
        return Replicated(scope=group.ranks())
    return Block(group=group, replica_cells=replica_cells, pad_to=pad_to)


def sharded(group: ProcessGroupFamily, pad_to: Optional[int] = None) -> AxisLayout:
    """Plain sharding: one rank per cell."""
    return block(group, tuple((i,) for i in range(group.group_size)), pad_to)


def segmented(segments) -> AxisLayout:
    """Canonicalizing constructor for the sum (unit law: one segment →
    its child)."""
    segs = tuple(segments)
    if len(segs) == 1:
        off, ext, child = segs[0]
        if (
            not _is_int(off)
            or not _is_int(ext)
            or off != 0
            or ext <= 0
            or isinstance(child, Unknown)
        ):
            raise ValueError("ill-formed single segment")
        physical_local_extent(child, ext)
        return child
    return Segmented(segments=segs)


def product(cuts) -> AxisLayout:
    """Canonicalizing constructor for the product (unit law: one cut →
    that cut)."""
    cuts = tuple(cuts)
    if len(cuts) == 1:
        if not isinstance(cuts[0], Block) or cuts[0].pad_to is not None:
            raise ValueError("a Product cut must be an unpadded Block")
        return cuts[0]
    return Product(cuts=cuts)


def flattened(factors, factor_extents) -> AxisLayout:
    """Construct a rank-1 row-major view without erasing factor ownership."""
    return Flattened(tuple(factors), tuple(factor_extents))


def _scopes_of(layout: AxisLayout):
    if isinstance(layout, Replicated):
        yield layout.scope
    elif isinstance(layout, Block):
        yield layout.group.ranks()
    elif isinstance(layout, Segmented):
        for _, _, child in layout.segments:
            yield from _scopes_of(child)
    elif isinstance(layout, Product):
        for cut in layout.cuts:
            yield cut.group.ranks()
    elif isinstance(layout, Flattened):
        for factor in layout.factors:
            yield from _scopes_of(factor)


def _reject_unknown(layout: AxisLayout) -> None:
    # 不可折叠：Unknown 是 bottom，五个查询的守卫留在门面一处；
    # 不给 Unknown 造五个只会 raise 的方法。
    if isinstance(layout, Unknown):
        raise ValueError(f"Unknown layout reached a consumer: "
                         f"{layout.reason or 'no reason recorded'}")


def logical_local_extent(layout: AxisLayout, extent: int) -> int:
    """The closed-form per-rank logical extent of a STATIC axis. Padded
    Blocks have no single per-rank logical number (tail num_shards run short) —
    callers price physical and check conservation instead."""
    _reject_unknown(layout)
    _assume_non_negative(extent)
    return layout.logical_local_extent(extent)


def physical_local_extent(layout: AxisLayout, extent: int) -> int:
    _reject_unknown(layout)
    _assume_non_negative(extent)
    return layout.physical_local_extent(extent)


def _assume_non_negative(extent: Any) -> None:
    """An extent is a count of elements. A number is checked here; an
    expression in the workload's symbols that SymPy cannot settle — the
    decode `s_kv = s_cache + n_verify - 1`, non-negative because both are at
    least one, which the assumption system does not infer across a sum — is
    an assumption of the walk, refused at a point where it goes negative.

    Asked of the assumption system (`dim.settled_non_negative`), not of a
    relation built to carry the question to it: an extent the declarations
    already settle — every literal extent, every count declared positive —
    states nothing, and `require` would return without recording it. One
    the declarations do not settle, and one that is negative, go the road
    they went before: the relation is built, and it is recorded or
    refused."""
    if settled_non_negative(extent):
        return
    require(sp.Ge(extent, 0),
           lambda at: ValueError(
               f"extent must be non-negative, got {at(extent)}"))


def owner_ranks(layout: AxisLayout, index: int, extent: int,
           universe: frozenset) -> frozenset:
    """Closed-form owner set of one global index. `universe` is the scope's
    full rank set (what Replicated means relative to)."""
    _reject_unknown(layout)
    _check_index(index, extent)
    return layout.owner_ranks(index, extent, universe)


def local_index(layout: AxisLayout, rank: int, index: int,
                 extent: int, universe: frozenset) -> Optional[int]:
    """The semantic appendix's `I(r, i)`: this rank's local offset for a
    global index, or None when the rank does not own it. Closed-form; used
    by the oracle to discharge injectivity/boundedness, never by pricing."""
    _reject_unknown(layout)
    _check_index(index, extent)
    return layout.local_index(rank, index, extent, universe)


def slice_axis(layout: AxisLayout, start: int, stop: int,
               extent: int) -> AxisLayout:
    """The transfer-matrix slice column. Replicated: any interval. Block:
    proper subintervals are REJECTED — they would empty some ranks and
    break the symmetry theorem; segment-internal slicing goes through
    Segmented. Segmented: per-segment intersection, recursing into hit
    segments. Product: whole axis only (phase-1 conservative stance)."""
    _reject_unknown(layout)
    require(
        sp.And(sp.Le(0, start), sp.Lt(start, stop), sp.Le(stop, extent)),
        lambda at: ValueError(
            f"bad interval [{at(start)}, {at(stop)}) of {at(extent)}"))
    return layout.slice_axis(start, stop, extent)


def axis_fully_replicated(axis: AxisLayout) -> bool:
    """Whether an axis has no owner cut, including a formal Segmented
    sum whose every segment is replicated.

    Segmented and Replicated stay distinct layouts: concat records the
    segment injection and no global canonicalization erases it. Kernel
    pairing may nevertheless consume either without communication.
    """
    if isinstance(axis, Replicated):
        return True
    if isinstance(axis, Segmented):
        return all(axis_fully_replicated(child)
                   for _offset, _extent, child in axis.segments)
    if isinstance(axis, Flattened):
        return all(axis_fully_replicated(factor) for factor in axis.factors)
    return False


def aligned(left: AxisLayout, right: AxisLayout, *,
            where: str, what: str) -> None:
    """The alignment law's non-broadcast branch: structural equality.

    二元法则：没有不任意的 receiver，不要写成 left.aligned(right)。
    """
    if left != right and not (
            axis_fully_replicated(left) and axis_fully_replicated(right)):
        raise ValueError(
            f"{where}: alignment law violated on {what}: "
            f"{left!r} != {right!r}")


def paired_heads(fine: AxisLayout, coarse: AxisLayout,
                 fine_extent: int, coarse_extent: int, *,
                 where: str) -> AxisLayout:
    """GQA / SSM-group head pairing: global fan-out plus per-rank locality.

    Layout equality is neither necessary nor sufficient. Not
    sufficient: 24 fine heads and 16 coarse heads both cut 8-way are
    structurally EQUAL layouts with an illegal grouping. Not
    necessary: GLM's 32 query heads 8-way beside 2 KV heads in two
    4-rank replica cells are UNEQUAL layouts and a legal grouping. The
    real condition is the one the replica cells encode: fine head `h`
    reads coarse head `h // (Hf/Hc)`, so every rank's fine-head interval
    must map into the coarse interval that rank co-owns.

    二元法则：fine/coarse 对称约束，不要挂成任一侧的方法。
    """
    if coarse_extent <= 0 or fine_extent % coarse_extent:
        raise ValueError(
            f"{where}: head fan-out {fine_extent} fine "
            f"heads over {coarse_extent} coarse heads does not divide — not a "
            "GQA grouping")
    if isinstance(coarse, Replicated):
        if isinstance(fine, (Replicated, Block)):
            return fine
        raise ValueError(
            f"{where}: unsupported fine head layout {fine!r}")
    if isinstance(fine, Replicated):
        raise ValueError(
            f"{where}: replicated fine heads over a "
            f"sharded coarse axis {coarse!r} — every rank would read coarse "
            "shards it does not own")
    if not (isinstance(fine, Block) and isinstance(coarse, Block)):
        raise ValueError(
            f"{where}: head axes must be Replicated or "
            f"Block, got {fine!r} / {coarse!r}")
    if fine.pad_to is not None or coarse.pad_to is not None:
        raise ValueError(f"{where}: padded head axes are not modeled")
    if not same(fine.group, coarse.group):
        raise ValueError(
            f"{where}: fine and coarse heads are cut over "
            f"different rank groups ({fine.group.id!r} vs "
            f"{coarse.group.id!r})")
    ratio = fine_extent // coarse_extent
    local_fine = physical_local_extent(fine, fine_extent)
    local_coarse = physical_local_extent(coarse, coarse_extent)
    for rank in sorted(fine.group.ranks()):
        fine_part = fine.shard_of_rank(rank)
        coarse_part = coarse.shard_of_rank(rank)
        first = (fine_part * local_fine) // ratio
        last = ((fine_part + 1) * local_fine - 1) // ratio
        if first < coarse_part * local_coarse \
                or last >= (coarse_part + 1) * local_coarse:
            raise ValueError(
                f"{where}: rank {rank} owns fine heads "
                f"[{fine_part * local_fine}, "
                f"{(fine_part + 1) * local_fine}) which read coarse heads "
                f"[{first}, {last}] outside its coarse shard "
                f"[{coarse_part * local_coarse}, "
                f"{(coarse_part + 1) * local_coarse}) — "
                "the head grouping and the replica cells disagree")
    return fine


@dataclass(frozen=True)
class TensorLayout:
    """Per-axis layouts plus buffer order. Element owner_ranks are the
    intersection of axis owner_ranks; coverage is guaranteed by the tensor-level
    uniform-fiber check, run in __post_init__ — raw construction is no
    bypass."""

    axes: Tuple[AxisLayout, ...]
    buffer_order: Tuple[int, ...] = field(default=None)  # None = row major

    def __post_init__(self) -> None:
        if self.buffer_order is not None and (
                sorted(self.buffer_order) != list(range(len(self.axes)))):
            raise ValueError("buffer_order must be a permutation of the "
                             "axis indices")
        scopes = {s for a in self.axes for s in _scopes_of(a)}
        if len(scopes) > 1:
            raise ValueError("all families of one tensor must cover the "
                             "same rank universe — a sub-scope Replicated "
                             "or cut says nothing about the other ranks")
        from itertools import product as cartesian
        for combo in cartesian(*self.cut_alternatives()):
            cuts = [c for group in combo for c in group]
            if cuts and not is_uniform_surjection(cuts):
                raise UniformFiberError(
                    "the cuts of this tensor do not tile its ranks "
                    "(diagonal or non-orthogonal joint coverage)")

    def cut_alternatives(self):
        """Per axis: the list of rank-cut combinations an element can see.
        A Product contributes all its cuts jointly; a Segmented contributes
        one ALTERNATIVE per segment (an element lives in exactly one segment
        — cuts of different segments never co-own an element, so they are
        checked separately, never jointly)."""
        def alts(a):
            if isinstance(a, Block):
                # A single-cell (padding-only) Block cuts no ranks; letting
                # its group into the check would bridge unrelated joint_components.
                return [[a]] if a.num_shards > 1 else [[]]
            if isinstance(a, Product):
                return [list(a.cuts)]
            if isinstance(a, Segmented):
                out = []
                for _, _, child in a.segments:
                    out.extend(alts(child))
                return out
            if isinstance(a, Flattened):
                combinations = [[]]
                for factor in a.factors:
                    combinations = [
                        left + right
                        for left in combinations
                        for right in alts(factor)
                    ]
                return combinations
            return [[]]
        return [alts(a) for a in self.axes]


def is_uniform_surjection(cuts) -> bool:
    """The tensor-level coverage check, per joint component: rank ->
    (part_1, .., part_k) must be SURJECTIVE onto the product of the
    coordinate sets present in the component, with equal fibers. The fiber
    VALUE follows (= |component| / the product of num_shards present); it equals
    the product of cell sizes only when the cut groups are independent —
    replication is exactly why injectivity is not required. Components
    are the connected scopes of "shares a group in any cut" — counting over
    the flat rank union would merge replica groups of one family into a
    spurious fiber."""
    cuts = list(cuts)
    if not cuts:
        return True
    scope = cuts[0].group.ranks()
    if any(c.group.ranks() != scope for c in cuts[1:]):
        return False
    for comp in joint_components([c.group for c in cuts], scope):
        labels: dict = {}
        for r in comp:
            labels.setdefault(
                tuple(c.shard_of_rank(r) for c in cuts), []).append(r)
        expected = 1
        for k in range(len(cuts)):
            expected *= len({key[k] for key in labels})
        if len(labels) != expected:
            return False
        fibers = {len(v) for v in labels.values()}
        if len(fibers) != 1:
            return False
        # The fiber value is |component| / expected by the two conditions
        # above; it equals the product of cell sizes only when the cut
        # groups are independent — same-group different-cell-partition cuts (two
        # axes cut by xor classes of one group) are legal with a smaller
        # fiber, so the cell-size product is NOT a check.
    return True


def tensor_layout(axes, buffer_order=None) -> TensorLayout:
    """Thin constructor; all validation lives in TensorLayout itself."""
    return TensorLayout(
        axes=tuple(axes),
        buffer_order=None if buffer_order is None else tuple(buffer_order))
