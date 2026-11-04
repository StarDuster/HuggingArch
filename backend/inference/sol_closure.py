"""The SOL closure: what an `estimate_capacity` result must satisfy before its
numbers are published.

An estimate that returns is not yet a valid one. Its result can carry a
`calibration_error`, and its per-op rows must close to the phase totals they
are summed into: each row's traffic partitions into weight read + metadata +
activation in + activation out, and per phase the rows' FLOPs, access and
write bytes equal the phase's. The SOL surface (`scripts/dump_sol_surface.py`
`measure_case`) publishes only a result that closes, and the validate replay
(`backend.inference.analysable`) accepts only one, so both ask this module.

A failure raises `SolClosureError`, a `ValueError` like every other analysis
refusal; a row's failure names the row's node (`node_context`).
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from backend.arch.eval.refusal import EngineGap, node_context

#: Each phase of a result, and the key of its per-op rows.
PHASE_ROWS = (("prefill", "per_op_timing"), ("decode", "per_op_timing_decode"))


class SolClosureError(EngineGap):
    """An estimate whose numbers do not close: the engine's own accounting
    disagrees with itself, which no spec edit resolves."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            capability=("an estimate whose numbers close (each per-op row's "
                        "traffic partitions its access, phase totals sum "
                        "the rows)"))


def row_partition(row: Mapping[str, Any]) -> Dict[str, int]:
    """One per-op row's traffic columns; raises unless
    `weight_read + metadata + act_in + act_out + convert == access`."""
    cols = {
        "access": int(row.get("access_bytes_per_rank") or 0),
        "metadata": int(row.get("metadata_bytes_per_rank") or 0),
        "weight_read": int(row.get("weight_read_bytes_per_rank") or 0),
        "act_in": int(row.get("act_in_bytes_per_rank") or 0),
        "act_out": int(row.get("act_out_bytes_per_rank") or 0),
        # The quantised copy of its own operand a kernel needs because the
        # edge does not carry it: a term of the row's access like the rest.
        "convert": int(row.get("convert_bytes_per_rank") or 0),
        "flops": int(row.get("flops_per_rank") or 0),
    }
    if (cols["weight_read"] + cols["metadata"] + cols["act_in"]
            + cols["act_out"] + cols["convert"] != cols["access"]):
        raise SolClosureError(
            f"row traffic does not partition for {row.get('aggregation_id')}: "
            f"{cols['weight_read']} + {cols['metadata']} + {cols['act_in']} + "
            f"{cols['act_out']} + {cols['convert']} != {cols['access']}")
    return cols


def check_phase(phase_name: str, phase: Mapping[str, Any],
                rows: Sequence[Mapping[str, Any]]) -> None:
    """One phase: every row partitions, and the rows' FLOPs, access and
    write bytes equal the phase's."""
    sums = {"access": 0, "metadata": 0, "weight_read": 0, "act_in": 0,
            "act_out": 0, "convert": 0, "flops": 0}
    for row in rows:
        with node_context(str(row.get("aggregation_id") or "")):
            cols = row_partition(row)
        for key in sums:
            sums[key] += cols[key]
    access = int(phase.get("total_bytes") or 0)
    flops = int(phase.get("flops_per_step") or 0)
    write = int(phase.get("mem_write_bytes") or 0)
    if sums["act_out"] != write:
        raise SolClosureError(
            f"{phase_name}: phase/per-op write closure failed: "
            f"phase={write}, rows={sums['act_out']}")
    if sums["access"] != access:
        raise SolClosureError(
            f"{phase_name}: phase/per-op access closure failed: "
            f"phase={access}, rows={sums['access']}")
    if sums["flops"] != flops:
        raise SolClosureError(
            f"{phase_name}: phase/per-op FLOPs closure failed: "
            f"phase={flops}, rows={sums['flops']}")


def check_closure(result: Mapping[str, Any]) -> None:
    """Raise `SolClosureError` unless `result` (an `estimate_capacity`
    return) closes: no `calibration_error`, and both phases close."""
    error = result.get("calibration_error")
    if error:
        raise SolClosureError(f"calibration: {error}")
    for phase_name, rows_key in PHASE_ROWS:
        check_phase(phase_name, result["phases"][phase_name],
                    result.get(rows_key) or [])
