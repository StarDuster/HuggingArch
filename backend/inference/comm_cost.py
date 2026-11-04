"""Per-rank traffic of the parallelism comm primitives.

Layout materializes the collectives and owns the flat-ring volume
(`layout.comm_volume.collective_per_rank_bytes`). This module turns
that volume into time: hierarchical phases, then bytes over the
per-direction fabric bandwidth (not HBM), plus each phase's fixed latency
(the α of the α-β model): the link latency (the request's, the same on
every link at this granularity) times the link traversals the phase's
schedule waits on in sequence (`comm_volume.latency_hops`: 2 log2 N for an
all-reduce, log2 N for a reduce-scatter, all-gather or all-to-all, one for
a point-to-point send). The latency is what a
one-token decode pays per collective whatever its bytes; a cost linear in
the payload alone priced a 64-way tensor-parallel decode at a rate no
deployment reaches.

`collective_phases` is the hierarchical schedule. When a collective's
world spans multiple nodes, NCCL does not run one flat global ring —
intra-node reduce-scatter, then a smaller inter-node all-reduce, then
intra-node all-gather. A flat `2(N-1)/N · T / rdma_bw` under-counts
because the RDMA hop only moves `T/N_intra` per rank.

Inside one NVLink domain a framework does not run NCCL's schedule on every
all-reduce: it dispatches by payload to its own one-shot and two-shot
kernels and, above them, to NVLS where the switch reduces
(`comm_volume.ALGORITHMS`). Which payload takes which is the framework's
binding (`framework_bindings`, `collectives:`, read here as
`CollectiveRung`s); whether the topology admits an algorithm is the card's
(`gpu_specs`: an NVLink domain, NVLS). `collective_ladder` is the one
place both are put together: the payload ranges and, for each, the bytes
and hops it is priced at — the walk's row, the row's law and every re-read
of it on another card or at another payload read that one answer.

`collective_per_rank_bytes` is re-exported so older callers keep the
same import. At N=1 every formula collapses to zero.
"""
from __future__ import annotations

import math
import re
from typing import Tuple, Any, Dict, List, Mapping, NamedTuple, Optional, Sequence

from backend.arch.ops import collective_schedule, sp_axis_effect
from backend.arch.layout.comm_volume import (ALGORITHMS, COMM_OPS, NCCL,
                                             ring_factor_symbolic)
from backend.arch.layout.comm_volume import (
    collective_per_rank_bytes as collective_per_rank_bytes,
)
from . import gpu_specs
from .phase_formulas import _fmt_si

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def sp_axis_index(axes: Optional[List[str]]) -> Optional[int]:
    """Position whose symbolic axis string carries a free `s_ctx`."""
    if not axes:
        return None
    for i, axis_str in enumerate(axes):
        if "s_ctx" in _IDENT_RE.findall(axis_str or ""):
            return i
    return None


def comm_display_shape(op: str, base_shape: Optional[List[int]],
                       axes: Optional[List[str]], world: int
                       ) -> Optional[List[int]]:
    """The tensor shape a collective presents on each rank after it runs.

    Shape-preserving except `reduce_scatter` (SP axis /world) and
    `all_gather` (×world). `base_shape` is the producer's per-rank
    output. Display only — interconnect bytes use the unadjusted
    producer shape.
    """
    if not base_shape:
        return None
    out = [int(d) for d in base_shape]
    effect = sp_axis_effect(op)
    if world > 1 and effect:
        i = sp_axis_index(axes)
        if i is not None and 0 <= i < len(out):
            v = int(out[i])
            if effect == "divide" and v % world == 0:
                out[i] = v // world
            elif effect == "multiply":
                out[i] = v * world
    return out


class Fabric(NamedTuple):
    """One card's interconnect, as `price_collective` reads it: its node
    size, each fabric's per-direction bandwidth (bytes/s) and label, one
    link traversal's latency (s, the request's `comm_latency_us`), and
    what its topology admits of a framework's own all-reduces
    (`gpu_specs.has_nvlink_domain`, `has_nvls`) beside the compute
    capability a dispatch tests. A fabric built from bandwidths alone
    admits none of them: NCCL's schedule prices every collective."""
    gpus_per_node: Optional[int]
    intra_bw: Optional[float]
    inter_bw: Optional[float]
    intra_label: str = "NVLink"
    inter_label: str = "RDMA"
    step_latency: float = 0.0
    nvlink: bool = False
    nvls: bool = False
    compute_capability: Optional[str] = None

    @classmethod
    def of(cls, gpu: "gpu_specs.GPUSpec", step_latency: float = 0.0
           ) -> "Fabric":
        """A card's fabric off its sheet row, at `step_latency`."""
        return cls(gpu_specs.gpus_per_node(gpu),
                   gpu_specs.peak_intra_bw_bytes_per_sec(gpu),
                   gpu_specs.peak_inter_bw_bytes_per_sec(gpu),
                   gpu_specs.intra_bw_label(gpu), gpu_specs.inter_bw_label(gpu),
                   step_latency)._replace(**topology(gpu))

    def without_latency(self) -> "Fabric":
        """This fabric with free steps: what its bandwidth alone charges."""
        return self._replace(step_latency=0.0)


def topology(gpu: Optional["gpu_specs.GPUSpec"]) -> Dict[str, Any]:
    """What a card's sheet row says its NVLink domain admits, as `Fabric`'s
    fields: none of it without a card."""
    if gpu is None:
        return {}
    return {"nvlink": gpu_specs.has_nvlink_domain(gpu),
            "nvls": gpu_specs.has_nvls(gpu),
            "compute_capability": gpu.compute_capability or None}


def _capability(text: str) -> Tuple[int, int]:
    major, _dot, minor = str(text).partition(".")
    return int(major), int(minor)


class CollectiveRung(NamedTuple):
    """One entry of a framework's all-reduce dispatch (`framework_bindings`,
    `collectives:`): the algorithm it runs (`comm_volume.ALGORITHMS`) on a
    payload of at most `max_bytes` (inclusive) and at most
    `max_bytes_times_world` summed over the group, in a group whose size
    is one of `worlds` (the sizes the framework's kernel accepts; a group of
    any other size runs NCCL), on a card whose compute capability is at
    least `min_compute_capability` and below `below_compute_capability`. A
    row carries its collective's rungs (`as_dict`), so a re-read of it on
    another card needs no binding."""
    algorithm: str
    max_bytes: Optional[int] = None
    max_bytes_times_world: Optional[int] = None
    min_compute_capability: Optional[str] = None
    below_compute_capability: Optional[str] = None
    worlds: Tuple[int, ...] = ()

    def cap(self, world: int) -> float:
        """The largest payload (bytes, inclusive) this rung takes in a
        group of `world`."""
        caps = [math.inf]
        if self.max_bytes is not None:
            caps.append(float(self.max_bytes))
        if self.max_bytes_times_world is not None:
            caps.append(float(self.max_bytes_times_world) / int(world))
        return min(caps)

    def runs_on(self, fabric: Fabric, *, world: int, base_stride: int) -> bool:
        """Whether the card's topology and capability admit this rung for a
        group of `world` ranks `base_stride` apart: every member in one
        NVLink domain, the topology the algorithm needs, and the capability
        the framework tests."""
        n = int(world)
        g = max(int(fabric.gpus_per_node or 1), 1)
        if n <= 1 or n > min(n, max(1, g // max(int(base_stride or 1), 1))):
            return False
        if n not in self.worlds:
            return False
        needs = ALGORITHMS[self.algorithm].needs
        if (needs == "nvlink" and not fabric.nvlink) or (
                needs == "nvls" and not fabric.nvls):
            return False
        if self.min_compute_capability or self.below_compute_capability:
            if not fabric.compute_capability:
                return False
            cc = _capability(fabric.compute_capability)
            if (self.min_compute_capability
                    and cc < _capability(self.min_compute_capability)):
                return False
            if (self.below_compute_capability
                    and cc >= _capability(self.below_compute_capability)):
                return False
        return True

    def as_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if k == "worlds" else v)
                for k, v in self._asdict().items() if v is not None}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CollectiveRung":
        d = dict(d)
        return cls(**{**d, "worlds": tuple(int(w) for w in d.get("worlds", ()))})


class Segment(NamedTuple):
    """One payload range of a collective on a card: payloads up to
    `max_bytes` (inclusive, past the ranges before it) run `algorithm`, at
    `s_per_byte` seconds per payload byte and `latency_s` fixed over the
    step's instances."""
    algorithm: str
    max_bytes: float
    s_per_byte: float
    latency_s: float


def collective_ladder(*, op: str, world: int, count_per_step: int,
                      base_stride: int, fabric: Fabric,
                      rungs: Sequence[CollectiveRung] = ()
                      ) -> Tuple[Segment, ...]:
    """A collective's payload ranges on `fabric`: one per rung of the
    framework's dispatch, in its order, then NCCL's schedule for every
    payload past them. A rung the card does not admit is an empty range
    (`max_bytes` 0) priced as NCCL's, so the ladder has one shape on every
    card and a zero payload costs what it does without one. The one answer
    to which algorithm a payload takes (`Segment.of`) and what it costs."""
    def priced(algorithm: str) -> Tuple[float, float]:
        per_byte, _rows = price_collective(
            op=op, world=world, tensor_bytes=1.0, count_per_step=count_per_step,
            fabric=fabric.without_latency(), base_stride=base_stride,
            algorithm=algorithm)
        return per_byte, collective_latency(
            op=op, world=world, count_per_step=count_per_step,
            gpus_per_node=fabric.gpus_per_node, base_stride=base_stride,
            step_latency=fabric.step_latency, algorithm=algorithm)

    nccl = priced(NCCL)
    out = []
    for rung in rungs:
        if rung.runs_on(fabric, world=world, base_stride=base_stride):
            out.append(Segment(rung.algorithm, rung.cap(world),
                               *priced(rung.algorithm)))
        else:
            out.append(Segment(NCCL, 0.0, *nccl))
    out.append(Segment(NCCL, math.inf, *nccl))
    return tuple(out)


def ladder_segment(ladder: Sequence[Segment], tensor_bytes: float) -> Segment:
    """The range a payload of `tensor_bytes` falls in: the first whose
    bound holds it, as the row's law chooses (`row_roofline`)."""
    for seg in ladder:
        if tensor_bytes <= seg.max_bytes:
            return seg
    return ladder[-1]


def collective_latency(*, op: str, world: int, count_per_step: int,
                       gpus_per_node: Optional[int], base_stride: int,
                       step_latency: float, algorithm: str = NCCL) -> float:
    """The fixed part of a collective's time over a step: each phase of its
    schedule (`collective_phases`, run by `algorithm`) pays `step_latency`
    per link traversal it waits on, for every instance in the step.
    Independent of the payload once the algorithm is chosen."""
    phases = collective_phases(
        op=op, world=int(world), tensor_bytes=1.0,
        gpus_per_node=int(gpus_per_node or 1),
        base_stride=int(base_stride or 1), algorithm=algorithm)
    return sum(step_latency * ph["hops"] for ph in phases) * int(count_per_step)


def price_collective(
    *,
    op: str,
    world: int,
    tensor_bytes: float,
    count_per_step: int,
    fabric: Fabric,
    base_stride: int,
    algorithm: str = NCCL,
    t_sym: Optional[str] = None,
    t_sub: Optional[str] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """One collective's time and per-phase rows on `fabric`, run by
    `algorithm` (`collective_ladder` chooses it for a payload).

    The anchor row in `phases` and the Gallery's re-stamp of the same
    layout on another card both call this, so a second card is priced by
    the same loop as the first rather than by a scalar ratio of link
    bandwidths applied to the first card's total.
    """
    phases_list = collective_phases(
        op=op, world=int(world), tensor_bytes=float(tensor_bytes),
        gpus_per_node=int(fabric.gpus_per_node or 1),
        base_stride=int(base_stride or 1), algorithm=algorithm,
        t_sym=t_sym, t_sub=t_sub,
    )
    t_comm = 0.0
    rows: List[Dict[str, Any]] = []
    for ph in phases_list:
        where = ph["fabric"]
        fabric_bw = fabric.intra_bw if where == "intra" else fabric.inter_bw
        fabric_label = (fabric.intra_label if where == "intra"
                        else fabric.inter_label)
        phase_bytes = float(ph["bytes_per_rank"]) * int(count_per_step)
        phase_latency = (fabric.step_latency * ph["hops"]
                         * int(count_per_step))
        phase_t = (phase_bytes / fabric_bw if fabric_bw and fabric_bw > 0
                   else 0.0) + phase_latency
        t_comm += phase_t
        rows.append({
            "name": ph["name"],
            "algorithm": ph["algorithm"],
            "fabric": where,
            "fabric_label": fabric_label,
            "world": ph["world"],
            "bytes_per_rank": phase_bytes,
            "bw_bytes_per_sec": float(fabric_bw or 0.0),
            # The link traversals one instance of the phase waits on, and
            # their time over the row's instances.
            "hops": ph["hops"],
            "latency_s": phase_latency,
            "t_phase_s": phase_t,
            "symbolic": ph["symbolic"],
            "substituted": ph["substituted"],
        })
    return t_comm, rows


def collective_phases(
    op: str,
    world: int,
    tensor_bytes: float,
    *,
    gpus_per_node: int,
    base_stride: int = 1,
    algorithm: str = NCCL,
    t_sym: Optional[str] = None,
    t_sub: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Per-phase wire bytes for one collective, hierarchy-aware.

    `algorithm` (`comm_volume.ALGORITHMS`) runs a group that fits one
    node as one phase at its bytes and hops; anything but NCCL's schedule is
    refused for a group that does not (`collective_ladder` never asks).
    Every phase states its `algorithm` and `hops`.

    Returns a list of `{name, fabric, world, tensor_bytes, bytes_per_rank,
    symbolic, substituted}` dicts. `fabric` is `"intra"` (fits inside one
    NVLink/PCIe domain) or `"inter"` (RDMA / cross-node). Sum of
    `bytes_per_rank` across the returned phases is the total wire bytes
    a rank actually transmits when the cost is summed `bytes / fabric_bw`.

    `base_stride` is the number of ranks BETWEEN consecutive members of
    this collective's group — i.e. the size of the parallelism inner_group
    nested INSIDE it on the device mesh. The standard layout nests TP
    innermost, so a TP collective has `base_stride = 1` (members are
    adjacent ranks), but an OUTER axis composed over TP is spaced out:
    a CP collective over `tp_attention` has `base_stride = tp_attention`,
    an EP collective over `tp_moe_expert` has `base_stride =
    tp_moe_expert`. Members co-located in one node:

        members_per_node = min(world, max(1, gpus_per_node // base_stride))

    With `base_stride = 1` this is `min(world, gpus_per_node)` and the
    whole function reduces EXACTLY to the previous size-only behavior —
    so single-TP costs are byte-identical. It only changes the composed
    case: a small outer-axis collective (e.g. cp=4) whose stride pushes
    every member onto a separate node (`members_per_node == 1`) correctly
    takes the inter fabric instead of being mis-priced as intra.

    Fabrics:
      * Point-to-point (`send_recv`): one hop of the whole tensor, on the
        intra fabric when the group fits in one node and on the inter fabric
        otherwise — the hop's place in the group is not known here, so a
        group that straddles nodes prices every hop at its slowest.
      * Single node (`world <= members_per_node`): one intra ring.
      * Fully cross-node (`members_per_node == 1`): one inter ring.
      * Straddle, AR, divisible: 3-phase Megatron/NCCL hierarchical AR —
          1. intra reduce-scatter on T     → (N_intra-1)/N_intra · T
          2. inter all-reduce on the shard → 2(N_inter-1)/N_inter · T/N_intra
          3. intra all-gather on T         → (N_intra-1)/N_intra · T
        where `N_intra = members_per_node`, `N_inter = world / N_intra`.
      * Straddle, non-AR or non-divisible: single inter ring (RS/AG/A2A
        have no widely-deployed hierarchical decomposition that wins, and
        the hierarchical AR schedule isn't well-defined when world isn't a
        multiple of the per-node group).
    """
    n = int(world or 1)
    if n <= 1 or op not in COMM_OPS or tensor_bytes <= 0:
        return []
    g = max(int(gpus_per_node or 1), 1)
    s = max(int(base_stride or 1), 1)
    # Group members that land inside one node, given the inner-inner_group
    # stride. base_stride=1 → min(n, g) (the legacy size-only rule).
    m = min(n, max(1, g // s))
    T = float(tensor_bytes)
    # `T` token in the formula. When the caller passes the tensor's
    # axis·dtype expansion (`t_sym` = "(B·s_ctx·d_model·dt_a)", `t_sub` =
    # "(237·1·7168·2)"), it replaces the opaque `T` / SI-byte literal so the
    # popover shows where the bytes come from. Defaults keep the prior
    # behavior byte-for-byte ("T" symbolic, SI-byte substituted).
    Tsym = t_sym or "T"
    Tsub_full = t_sub or _fmt_si(int(T), " B")

    def _phase(name: str, fabric: str, w: int, tb: float, sym: str, sub: str,
               alg: str = NCCL) -> Dict[str, Any]:
        if w <= 1 or tb <= 0:
            return {}
        # The coefficients are declared once, in `comm_volume.ALGORITHMS`.
        run = ALGORITHMS[alg]
        return {
            "name": name,
            "algorithm": alg,
            "fabric": fabric,
            "world": int(w),
            "tensor_bytes": float(tb),
            "bytes_per_rank": run.factor(name, w) * tb,
            "hops": run.hops(name, w),
            "symbolic": sym,
            "substituted": sub,
        }

    # Intra-domain — the whole group fits inside one NVLink/PCIe inner_group
    # (every member co-located). One phase, by `algorithm`.
    if n <= m:
        run = ALGORITHMS[algorithm]
        ph = _phase(
            op, "intra", n, T,
            f"{run.factor_symbolic(op)} * {Tsym}",
            f"{run.factor_symbolic(op, n)} * {Tsub_full}",
            algorithm,
        )
        return [ph] if ph else []
    if algorithm != NCCL:
        raise ValueError(
            f"{op} of {n} ranks spans more than one node of {g}: only NCCL's "
            f"schedule runs it, not {algorithm}")

    # Cross-domain. AR straddling node boundaries (more than one member
    # per node) uses a 3-phase hierarchy; everything else — fully
    # cross-node (one member per node), non-AR, or non-divisible — falls
    # back to a single inter-fabric ring.
    if collective_schedule(op) != "hierarchical" or m <= 1 or (n % m) != 0:
        coeff_sym = ring_factor_symbolic(op)
        coeff_sub = ring_factor_symbolic(op, n)
        ph = _phase(
            op, "inter", n, T,
            f"{coeff_sym} * {Tsym}",
            f"{coeff_sub} * {Tsub_full}",
        )
        return [ph] if ph else []

    n_intra = m
    n_inter = n // m
    shard_T = T / n_intra
    # Inter phase runs on the post-RS shard T/N_intra. When T is expanded
    # the shard reads `(B·s_ctx·d_model·dt_a)/N_intra` (division explicit); the
    # default path keeps the pre-divided SI-byte literal byte-for-byte.
    shard_sub = f"{Tsub_full}/{n_intra}" if t_sub else _fmt_si(int(shard_T), " B")
    out: List[Dict[str, Any]] = []
    p1 = _phase(
        "reduce_scatter", "intra", n_intra, T,
        f"{ring_factor_symbolic('reduce_scatter', symbol='N_intra')} * {Tsym}",
        f"{ring_factor_symbolic('reduce_scatter', n_intra)} * {Tsub_full}",
    )
    if p1:
        out.append(p1)
    p2 = _phase(
        "all_reduce", "inter", n_inter, shard_T,
        f"{ring_factor_symbolic('all_reduce', symbol='N_inter')} * {Tsym}/N_intra",
        f"{ring_factor_symbolic('all_reduce', n_inter)} * {shard_sub}",
    )
    if p2:
        out.append(p2)
    p3 = _phase(
        "all_gather", "intra", n_intra, T,
        f"{ring_factor_symbolic('all_gather', symbol='N_intra')} * {Tsym}",
        f"{ring_factor_symbolic('all_gather', n_intra)} * {Tsub_full}",
    )
    if p3:
        out.append(p3)
    return out

