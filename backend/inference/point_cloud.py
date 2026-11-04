"""The one filling producer for decode point clouds.

Every cloud states the same coordinates and keeps the same absence records.
Context×batch, parallel-layout and Gallery code enumerate their own questions
and yield `CloudRecord`s; this module is the one place that decides whether a
record becomes a served point or an absent one, and it alone calls the decode
and absence constructors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from .frontier import POINT_DIMS, PointRefusal, absent_point, decode_point


def _held_coordinates(coordinates: Mapping[str, Any]) -> None:
    """Refuse a served point that is not fully identified."""
    missing = [name for name in POINT_DIMS if name not in coordinates]
    unknown = [name for name in coordinates if name not in POINT_DIMS]
    if missing or unknown:
        raise ValueError(
            f"a point states every cloud coordinate; missing={missing}, "
            f"unknown={unknown}")


@dataclass
class PointCloud:
    """A filled cloud, with the caller's token beside each point."""

    points: list[dict] = field(default_factory=list)
    tokens: dict[int, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CloudRecord:
    """One question a consumer asks of the cloud, before it is filled.

    A source enumerates records: it states the coordinates, the estimate to
    read and any source-side provenance. It may also state an existing refusal
    (layout, error, card) or the capacity verdict. The producer alone turns
    that record into a point.
    """
    coordinates: Mapping[str, Any]
    estimate: Any = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    token: Any = None
    refusal: Optional[PointRefusal] = None
    reason: str = ""
    served: bool = True
    beyond_max_context: bool = False
    capacity_reason: str = ""
    no_step_refusal: PointRefusal = "card"
    no_step_reason: str = "this card publishes no decode step"

    @classmethod
    def unanswered(cls, coordinates: Mapping[str, Any],
                   refusal: PointRefusal, reason: str, *,
                   token: Any = None) -> "CloudRecord":
        return cls(coordinates=coordinates, refusal=refusal, reason=reason,
                   token=token)


def point_coordinates(
    *, mode: str, num_gpus: int, gpu_sku: str, context: int,
    batch_key: Any, draft_tokens: int, acceptance: float,
    ep_load_factor: Any,
) -> dict[str, Any]:
    """The one coordinate mapping, including a plan point's ρ."""
    try:
        rho = float(ep_load_factor)
    except (TypeError, ValueError):
        rho = str(ep_load_factor)
    return {
        "mode": mode,
        "num_gpus": int(num_gpus),
        "gpu_sku": gpu_sku,
        "context": context,
        "batch_key": batch_key,
        "draft_tokens": max(0, int(draft_tokens)),
        "acceptance": float(acceptance),
        "ep_load_factor": rho,
    }


def produce_cloud(records: Sequence[CloudRecord]) -> PointCloud:
    """Fill one cloud from records, served or absent.

    This is the only point-cloud filling path. A record that already knows it
    is refused stays absent. A readable record is decoded here; a point past
    the model's published context keeps its values and warning, an unserved
    point is capacity-refused, and no decode time is the source's stated card
    or capacity refusal.
    """
    cloud = PointCloud()
    for record in records:
        coordinates = dict(record.coordinates)
        _held_coordinates(coordinates)
        if record.refusal is not None:
            point = absent_point(coordinates, record.refusal, record.reason)
        else:
            drawn = decode_point(record.estimate)
            provenance = dict(record.provenance)
            if record.beyond_max_context and drawn is not None:
                point = {**drawn, **coordinates, **provenance,
                         "beyond_max_context": True}
            elif not record.served:
                point = absent_point(coordinates, "capacity",
                                     record.capacity_reason)
            elif drawn is None:
                point = absent_point(coordinates, record.no_step_refusal,
                                     record.no_step_reason)
            else:
                point = {**drawn, **coordinates, **provenance}
        cloud.points.append(point)
        cloud.tokens[id(point)] = record.token
    return cloud


def estimate_records(
    *,
    seq_lens: Sequence[int],
    verify_widths: Sequence[int],
    batch_points: Sequence[tuple[Optional[str], Optional[int]]],
    estimates: Sequence[Sequence[Any]],
    gpu_sku: str,
    parallel: Any,
    num_gpus: int,
    acceptance_rate: float,
    ep_load_factor: Any,
    served: Callable[..., bool],
) -> Iterable[CloudRecord]:
    """The context×batch cloud from a `CapacitySweepPlan`'s finished points.

    `served` is capacity's `point_served`, passed by the caller to keep this
    module below capacity's preparation layer.
    """
    from .frontier import parallel_mode

    mode = parallel_mode(parallel)
    for seq, verify, points in zip(seq_lens, verify_widths, estimates,
                                   strict=True):
        for (key, _prefill, decode), est in zip(batch_points, points,
                                                strict=True):
            coordinates = point_coordinates(
                mode=mode, num_gpus=num_gpus, gpu_sku=gpu_sku, context=seq,
                batch_key=key if key is not None else "max",
                draft_tokens=int(verify) - 1,
                acceptance=acceptance_rate, ep_load_factor=ep_load_factor)
            if not est:
                yield CloudRecord.unanswered(
                    coordinates, "error", "the sweep returned no point")
                continue
            is_served = served(
                max_batch_per_gpu=est.get("max_batch_per_gpu"),
                beyond_max_context=bool(est.get("beyond_max_context")),
                requests=decode or 1)
            beyond = bool(est.get("beyond_max_context"))
            if not is_served and not beyond:
                ceiling = int(est.get("max_batch_per_gpu") or 0)
                yield CloudRecord(
                    coordinates=coordinates, estimate=est, served=False,
                    beyond_max_context=beyond,
                    capacity_reason=(
                        f"this GPU serves {ceiling} request"
                        f"{'s' if ceiling != 1 else ''} at this context"))
                continue
            yield CloudRecord(coordinates=coordinates, estimate=est,
                              beyond_max_context=beyond)


def estimate_cloud(**kwargs: Any) -> PointCloud:
    """The context×batch cloud, filled by the one producer."""
    return produce_cloud(estimate_records(**kwargs))


def parallel_records(plan: Any, results: Sequence[Any],
                     layout_label: Callable[[Any], str]
                     ) -> Iterable[CloudRecord]:
    """The Parallelism sweep's cloud from every layout/card/batch answer.

    `plan` is the caller-side `ParallelSweepPlan`; only its published coordinate
    and absence interface is read here, so this module owns the cloud without
    owning layout enumeration.
    """
    for cell, result in zip(plan.cells, results, strict=True):
        if "refused" in result:
            for gpu in plan.gpu_skus:
                yield CloudRecord.unanswered(
                    plan.absent_coordinates(
                        cell.mode, cell.parallel.cluster_size, gpu,
                        batch_key=plan.requested_batch_key),
                    "layout", str(result["refused"]))
            continue
        if result.get("error"):
            for gpu in plan.gpu_skus:
                yield CloudRecord.unanswered(
                    plan.absent_coordinates(
                        cell.mode, cell.parallel.cluster_size, gpu,
                        batch_key=plan.requested_batch_key),
                    "error", str(result["error"]))
            continue
        for gpu, card in (result.get("cards") or {}).items():
            if "error" in card:
                yield CloudRecord.unanswered(
                    plan.absent_coordinates(
                        cell.mode, cell.parallel.cluster_size, gpu,
                        batch_key=plan.requested_batch_key),
                    "card", str(card["error"]))
                continue
            for point in card.get("ladder") or ():
                coordinates = plan.coordinates(
                    cell.mode, cell.parallel.cluster_size, gpu,
                    batch_key=point.get("batch_key", "max"))
                beyond = ({"beyond_max_context": True}
                          if point.get("beyond_max_context") else {})
                yield CloudRecord(
                    coordinates=coordinates, estimate=point,
                    provenance={
                        "layout": layout_label(cell.parallel), "x": cell.x,
                        "parallel": cell.parallel.to_dict(), **beyond},
                    no_step_refusal="capacity",
                    no_step_reason="this card serves no decode step")


def parallel_cloud(plan: Any, results: Sequence[Any],
                   layout_label: Callable[[Any], str]) -> PointCloud:
    """The Parallelism cloud, filled by the one producer."""
    return produce_cloud(parallel_records(plan, results, layout_label))
