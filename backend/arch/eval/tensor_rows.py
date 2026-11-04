"""The storage ⋈ formats row both the walk and traffic read.

The row and its one join live outside `analysis` because `Analysis` is a
producer of the joined rows while `traffic` consumes them; owning the join in
either made the other import its consumer/producer and forced `weight_promotions`
to import traffic inside a method. Here both read the same edge join.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterator, NamedTuple, Optional, Sequence, Tuple

from .storage import StorageKind


class TensorRow(NamedTuple):
    """One stored tensor as `Analysis.tensors()` hands it out.

    A named row rather than a bare 4-tuple: the fields were parallel by
    position, which is the shape `Layer` exists to kill, on the one surface a
    production caller reads. Still a tuple, so positional unpacking at existing
    call sites keeps working.
    """

    #: node the tensor belongs to, `Program.by_uid`-resolvable
    uid: int
    #: the stored tensor itself (`storage.StoredTensor`)
    tensor: Any
    #: how it is represented (`formats.Representation`: the format the
    #: checkpoint saved it at and the one the deployment runs it at), `None`
    #: when no format applies. A row from a CHECKPOINT projection
    #: (`checkpoint_tensors`) carries the saved format alone.
    rep: Optional[Any]
    #: physical copies in the whole model — per-node values are per-COPY
    count: Any
    #: which of the node's stored tensors this is (`Analysis.storage`'s
    #: index). Carried rather than left to the reader's enumeration, because
    #: a tensor stored at several formats across its node's layers is several
    #: rows at one index, and a per-rank projection joins on the index
    #: (`layout.persistent` keys its artifacts `(uid, tensor_index)`).
    tensor_index: int = 0
    #: which layer segment of that tensor this is — 0 for a tensor whose
    #: layers all store it alike, and the segment's position in
    #: `formats.Representations` otherwise. A per-rank projection joins its
    #: artifacts on it (`layout.persistent` keys its artifacts
    #: `(uid, tensor_index, segment_index)`), so a segment takes its own
    #: placement and its own copies rather than the last segment's.
    segment_index: int = 0

    @property
    def name(self) -> str:
        return self.tensor.name

    @property
    def logical_name(self) -> str:
        return self.tensor.logical_name

    @property
    def shape(self) -> Tuple[Any, ...]:
        return tuple(self.tensor.shape)

    @property
    def elements(self) -> Any:
        return self.tensor.elements

    @property
    def format(self) -> Any:
        return None if self.rep is None else self.rep.resident

    @property
    def storage_dtype(self) -> Optional[str]:
        fmt = self.format
        if fmt is not None:
            return fmt.storage_dtype
        return self.tensor.dtype

    @property
    def logical_dtype(self) -> str:
        if self.tensor.logical_dtype:
            return self.tensor.logical_dtype
        fmt = self.format
        if fmt is not None:
            return fmt.logical_dtype
        return self.storage_dtype or ""

    @property
    def storage_bytes(self) -> Any:
        fmt = self.format
        if fmt is None:
            return self.tensor.bytes() if self.tensor.dtype else 0
        return fmt.weight_bytes_for(self.elements)

    @property
    def is_metadata(self) -> bool:
        return self.tensor.kind == StorageKind.METADATA

    @property
    def is_buffer(self) -> bool:
        return self.tensor.kind == StorageKind.BUFFER

    @property
    def method(self) -> Optional[str]:
        fmt = self.format
        return None if fmt is None else fmt.method

    @property
    def selected_by(self) -> str:
        fmt = self.format
        return "" if fmt is None else fmt.selected_by

    @property
    def metadata(self) -> Tuple[Any, ...]:
        fmt = self.format
        return () if fmt is None else fmt.metadata


def _shared(count: Any, share: Fraction) -> Any:
    """`count` restricted to one layer segment: its share of the copies.

    A whole number of copies when `count` is the node's copies — a node's
    layer count is a factor of them, so a segment's share divides exactly.
    A caller that bills ONE copy and multiplies by its own count gets the
    share itself, which is the weight its multiplication needs: summed over
    the segments it is the per-copy average, and times the copies it is the
    exact total.
    """
    if share == 1:
        return count
    scaled = count * share
    if isinstance(scaled, Fraction) and scaled.denominator == 1:
        return int(scaled)
    return scaled


def pair_stored(
    uid: int,
    stored: Optional[Sequence] = None,
    reps: Optional[Sequence] = None,
    count: Any = 1,
) -> Iterator[TensorRow]:
    """Index-parallel zip of stored tensors with their formats.

    The two tuples are aligned by index; a short formats tuple leaves
    `rep is None` rather than inventing a width. This is the join
    `tensors()` documents — traffic and the per-rank projection call it
    too, so a bias cannot pair with its weight's format in one reader
    and not another. Callers that bill a single copy (a fused kernel
    reading an origin) pass `count=1`.

    One row per (stored tensor × layer segment). A tensor is one segment
    unless a layer-scoped `quant_context.overrides` row gives some of its
    node's layers another saved format (`formats.Representations`); it is
    then one row per format, each carrying that segment's copies, which is
    what `count` has always meant. Step-5's expert bank is 85 copies of FP8
    and 3 of BF16 out of one node, and no caller needs a new idea for it.
    """
    stored = stored or ()
    reps = reps or ()
    for index, tensor in enumerate(stored):
        reps_of_tensor = reps[index] if index < len(reps) else None
        if reps_of_tensor is None:
            yield TensorRow(uid, tensor, None, count, index)
            continue
        for segment_index, (segment, share) in enumerate(
                reps_of_tensor.with_shares()):
            yield TensorRow(uid, tensor, segment, _shared(count, share),
                            index, segment_index)
