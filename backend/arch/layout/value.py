"""Value — is this memory the final answer, or a mergeable accumulator.

The second edge fact (the first being `ExpectedRoutedLoad`): shape-complete
but value-pending. A `Partial` never survives past an endpoint, and the
only ops it may pass through are those with a registered `ReductionHomomorphism` —
the default is `requires_complete`, which is always CORRECT (finalizing
immediately never computes a wrong number; it can only place the
reduction earlier than the framework's real schedule).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .groups import ProcessGroupFamily, joint_components, orthogonal

SUM = "sum"
MAX = "max"
ONLINE_ATTENTION = "online_attention"
PENDING_MERGE_KINDS = (SUM, MAX, ONLINE_ATTENTION)

#: Payload names per pending-merge kind — communication is priced on the
#: schema, not on the visible output shape. The (running max, normalizer)
#: pair is Milakov & Gimelshein's online softmax state; the weighted-value
#: payload is the online-attention extension FlashAttention productized.
#: These are mergeable accumulator schemas with associative merges, not
#: scalar reducers of one type.
PAYLOAD_SCHEMAS = {
    SUM: ("value",),
    MAX: ("value",),
    ONLINE_ATTENTION: ("running_max", "normalizer", "weighted_value"),
}


@dataclass(frozen=True)
class Complete:
    """The local buffer holds final values."""


@dataclass(frozen=True)
class ValueProvenance:
    """Where a semantic value was produced and which operands define it.

    Provenance is evidence for a later lowering, not part of the distributed
    value algebra: two otherwise-identical partials do not become different
    layouts merely because their graph uids differ. Named operands keep the
    lowering contract structural instead of making it rediscover roles by
    scanning op names or component paths.
    """

    producer: int
    operands: Tuple[Tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.producer, int) or isinstance(self.producer, bool):
            raise ValueError(
                f"provenance producer must be an integer uid, got "
                f"{self.producer!r}")
        roles = set()
        for role, uid in self.operands:
            if not isinstance(role, str) or not role.strip():
                raise ValueError(
                    f"provenance operand role must be non-empty, got "
                    f"{role!r}")
            if role in roles:
                raise ValueError(
                    f"duplicate provenance operand role {role!r}")
            if not isinstance(uid, int) or isinstance(uid, bool):
                raise ValueError(
                    f"provenance operand {role!r} must name an integer "
                    f"uid, got {uid!r}")
            roles.add(role)

    def operand(self, role: str) -> int:
        """The uid bound to `role`, failing closed when it was not declared."""
        for name, uid in self.operands:
            if name == role:
                return uid
        raise ValueError(
            f"provenance from uid {self.producer} has no operand {role!r}; "
            f"declared roles are {[name for name, _uid in self.operands]}")


@dataclass(frozen=True)
class Partial:
    """The local buffer holds one addend of a pending reduction over
    `group`. For contractions over a replicated-cell Block, `group` is the
    cross-replica derived family (see groups.cross_replica_family) — the derived family itself encodes which rank speaks for which
    replica cell, so no cell data rides along."""

    reducer: str
    group: ProcessGroupFamily
    schema: Tuple[str, ...] = ()
    provenance: Optional[ValueProvenance] = field(
        default=None, compare=False)

    def __post_init__(self) -> None:
        if self.reducer not in PENDING_MERGE_KINDS:
            raise ValueError(f"unknown reducer {self.reducer!r} — the "
                             "vocabulary is closed")
        if not self.schema:
            object.__setattr__(self, "schema", PAYLOAD_SCHEMAS[self.reducer])
        elif self.schema != PAYLOAD_SCHEMAS[self.reducer]:
            raise ValueError(f"schema {self.schema!r} does not belong to "
                             f"reducer {self.reducer!r} — schemas are part "
                             "of the closed vocabulary")
        if self.provenance is not None and not isinstance(
                self.provenance, ValueProvenance):
            raise TypeError(
                f"Partial.provenance must be ValueProvenance or None, got "
                f"{self.provenance!r}")

    def with_provenance(self, producer: int, **operands: int) -> "Partial":
        """Attach graph evidence without changing this Partial's algebra."""
        return Partial(
            reducer=self.reducer,
            group=self.group,
            schema=self.schema,
            provenance=ValueProvenance(
                producer=producer, operands=tuple(operands.items())),
        )


@dataclass(frozen=True)
class UnknownValue:
    reason: str = ""


def join_orthogonal_partials(a: Partial, b: Partial) -> Partial:
    """Two pending reductions merge into one join-group Partial iff same
    reducer/schema and ORTHOGONAL groups; anything else finalizes first.
    The join family's groups are the connected joint_components of the union
    linkage — the coarsest family both reductions refine."""
    if a.reducer != b.reducer or a.schema != b.schema:
        raise ValueError("mixed reducers/schemas never merge — finalize "
                         "the earlier Partial first")
    if not orthogonal(a.group, b.group):
        raise ValueError("non-orthogonal pending groups never merge — "
                         "finalize the earlier Partial first")
    # Order each joined group by the COORDINATE pair, not by physical
    # rank id: rank-sorted groups lose the product structure and fail
    # orthogonality against a third independent family (the join must be
    # associatively composable).
    joined = ProcessGroupFamily(
        id=f"join({a.group.id},{b.group.id})",
        groups=tuple(
            tuple(sorted(comp, key=lambda r: (a.group.coordinate(r),
                                              b.group.coordinate(r))))
            for comp in joint_components((a.group, b.group),
                                         a.group.ranks())))
    return Partial(reducer=a.reducer, group=joined, schema=a.schema)


COMPLETE = Complete()
