"""Structured placement accounting beside a Kernel Bench plan.

The three leaf outcomes are kept apart even when their old textual notes were
easy to conflate: a zero-cost launch is measured@0, an unknown op family has no
measurement contract, and a known signature whose projection cannot be stated
is a projection repair rather than a provider gap.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kernel_bench.plan import (build_jobs_with_unplaced, read_jobs_jsonl,
                               unplaced_by_reason, unplaced_path, write_plan)


MODEL = "meta-llama/Llama-3.2-1B"


def _plan(monkeypatch=None, *, without_signature=None, without_projection=None):
    from backend.inference import benchmark_identity
    from backend.inference import calibration_overlay

    if without_signature is not None:
        assert monkeypatch is not None
        real_bindings = calibration_overlay.op_bindings

        def bindings():
            return {op: signature for op, signature in real_bindings().items()
                    if op != without_signature}
        monkeypatch.setattr(calibration_overlay, "op_bindings", bindings)

    if without_projection is not None:
        assert monkeypatch is not None
        real_identity = benchmark_identity.identity_shape_from_shape_in

        def identity(op, shape_in, **kwargs):
            return None if op == without_projection else real_identity(
                op, shape_in, **kwargs)
        monkeypatch.setattr(
            benchmark_identity, "identity_shape_from_shape_in", identity)

    return build_jobs_with_unplaced(
        model_id=MODEL, batch_size=1, seq_len=2048, phase="decode",
        avg_output_tokens=0)


def test_a_nonzero_op_without_a_signature_is_missing_signature(monkeypatch):
    _jobs, _notes, records = _plan(
        monkeypatch, without_signature="rmsnorm")
    missing = [record for record in records
               if record["reason"] == "missing_signature"]
    assert missing
    assert {record["op"] for record in missing} == {"rmsnorm"}
    assert all(record["aggregation_id"] for record in missing)


def test_a_signature_without_a_projection_names_the_refusal(monkeypatch):
    _jobs, _notes, records = _plan(
        monkeypatch, without_projection="linear")
    missing = [record for record in records
               if record["reason"] == "missing_projection"
               and record["op"] == "linear"]
    assert missing
    assert all("no shape projection" in record["detail"] for record in missing)
    assert all(record["aggregation_id"] for record in missing)


def test_a_zero_cost_view_row_is_measured_at_zero_not_unplaced():
    jobs, _notes, records = _plan()
    zero = [record for record in records
            if record["reason"] == "zero_launch"]
    assert zero
    assert {"view_split", "transpose", "view_merge"} <= {
        record["op"] for record in zero}
    assert not {"view_split", "transpose", "view_merge"} & {
        job["op"] for job in jobs}


def test_zero_launch_wins_over_a_concurrent_decline(monkeypatch):
    from backend.inference.benchmark_identity import RowIdentity

    real = RowIdentity.leaf_dtype_env

    def declining(identity, row):
        env, why = real(identity, row)
        if row.get("op") in {"view_split", "transpose", "view_merge"}:
            return env, "the leaf's dtype environment has no corpus token"
        return env, why

    monkeypatch.setattr(RowIdentity, "leaf_dtype_env", declining)
    _jobs, _notes, records = _plan()
    reasons = {record["reason"] for record in records
               if record["op"] in {"view_split", "transpose", "view_merge"}}
    assert reasons == {"zero_launch"}


def test_the_plan_consumes_point_rows_classifier_and_its_order(monkeypatch):
    """Pin the shared order at the producer and consumer in one path.

    `point_keys` classifies every row; this plan run consumes that
    `placement_reason` instead of rebuilding it from the underlying facts.  The
    selected rows deliberately carry competing facts, including a zero-cost
    decline without signature or projection and a nonzero decline with both
    identity gaps.
    """
    from backend.inference import benchmark_identity, calibration_overlay
    from backend.inference.benchmark_identity import RowIdentity

    competing = {"view_split", "linear", "rmsnorm", "add"}
    real_bindings = calibration_overlay.op_bindings

    def bindings():
        return {op: signature for op, signature in real_bindings().items()
                if op not in {"view_split", "linear", "rmsnorm"}}

    real_identity = benchmark_identity.identity_shape_from_shape_in

    def identity(op, shape_in, **kwargs):
        return None if op in competing else real_identity(
            op, shape_in, **kwargs)

    real_leaf = RowIdentity.leaf_dtype_env

    def declining(identity, row):
        env, why = real_leaf(identity, row)
        if row.get("op") in {"view_split", "linear"}:
            return env, "the leaf's dtype environment has no corpus token"
        return env, why

    monkeypatch.setattr(calibration_overlay, "op_bindings", bindings)
    monkeypatch.setattr(
        benchmark_identity, "identity_shape_from_shape_in", identity)
    monkeypatch.setattr(RowIdentity, "leaf_dtype_env", declining)
    _jobs, _notes, records = _plan()
    by_op = {record["op"]: record["reason"] for record in records}

    assert by_op["view_split"] == "zero_launch"
    assert by_op["linear"] == "declined"
    assert by_op["rmsnorm"] == "missing_signature"
    assert by_op["add"] == "missing_projection"


def test_one_snapshot_writes_all_three_structural_records(monkeypatch, tmp_path):
    jobs, _notes, records = _plan(
        monkeypatch, without_signature="rmsnorm", without_projection="linear")
    reasons = {record["reason"] for record in records}
    assert {"zero_launch", "missing_signature", "missing_projection"} <= reasons
    assert not {"linear", "rmsnorm"} & {job["op"] for job in jobs}

    plan = tmp_path / "plan.jsonl"
    _queued, _gaps = write_plan(jobs, str(plan), unplaced=records)
    assert not unplaced_path(str(plan)).endswith(".jsonl")
    on_disk = read_jobs_jsonl(unplaced_path(str(plan)))
    assert {record["reason"] for record in on_disk} >= reasons
    assert set(on_disk[0]) >= {"op", "aggregation_id", "reason"}
    lines = unplaced_by_reason(on_disk)
    assert lines[0].startswith("zero_launch")
    assert [line.split(":", 1)[0] for line in lines] == [
        "zero_launch", "missing_signature", "missing_projection"]

    write_plan([], str(plan), unplaced=[])
    assert not Path(unplaced_path(str(plan))).exists()


def test_report_counts_structural_rows_in_the_plan_denominator(
        tmp_path, capsys):
    from kernel_bench.cli import main

    records = [
        {"op": "view", "aggregation_id": "view", "reason": "zero_launch"},
        {"op": "custom_op", "aggregation_id": "custom_op",
         "reason": "missing_signature"},
        {"op": "linear", "aggregation_id": "linear",
         "reason": "missing_projection"},
    ]
    plan = tmp_path / "plan.jsonl"
    write_plan([], str(plan), unplaced=records)
    assert main(["report", "--jobs", str(plan), "--gpu", "H100-SXM"]) == 0
    out = capsys.readouterr().out
    assert "report: 3 cells" in out
    assert "structural 3" in out
    assert "zero_launch: 1 row(s)" in out


def test_the_queue_carries_structural_records_without_reading_them(tmp_path):
    from kernel_bench.queue import Workspace, hand_over, pending

    records = [{"op": "custom_op", "aggregation_id": "custom_op",
                "reason": "missing_signature"}]
    staged = tmp_path / "staged.jsonl"
    write_plan([], str(staged), unplaced=records)
    space = Workspace(tmp_path / "box").prepare()
    hand_over(space, "org/model", staged)

    assert pending(space) == ["org/model"]
    moved = read_jobs_jsonl(unplaced_path(str(space.plan_of("org/model"))))
    assert moved == records


def test_coverage_keeps_zero_in_measured_and_details_with_unplaced():
    from backend.measure.auto_sweep import (MEASURED, MISSING_PROJECTION,
                                             MISSING_SIGNATURE, PointMeasure,
                                             MeasuredFormula, ZERO_LAUNCH)
    from backend.measure.lookup import Key

    key = Key("gemm", '{"m":1}', '{"activation":"bf16"}', "H100-SXM", "decode")
    measure = PointMeasure(
        keys=(None, None, None, key),
        launches=(0.0, 0.0, 0.0, 1.0),
        unkeyed=(MEASURED, "unplaced", "unplaced", None),
        labels=("view", "custom_op", "linear", "linear"),
        reasons=(ZERO_LAUNCH, MISSING_SIGNATURE, MISSING_PROJECTION, ""),
        formula=MeasuredFormula("T_3*n_3", (True, False, False, False)),
        aggregation_ids=("view", "custom_op", "linear", "linear"))
    folded = measure.fold(
        lambda _key, _context, _label, _read: (1.0, MEASURED, None))
    coverage = folded.track.coverage()
    assert coverage["measured"] == 2
    assert coverage["total"] == 4
    assert coverage["states"]["unplaced"] == ["custom_op", "linear"]
    assert coverage["unplaced_details"] == [
        {"op": "custom_op", "aggregation_id": "custom_op",
         "reason": MISSING_SIGNATURE},
        {"op": "linear", "aggregation_id": "linear",
         "reason": MISSING_PROJECTION},
    ]
    assert folded.rows[0][0] == MEASURED and folded.rows[0][1] == 0.0


def test_structured_reasons_are_closed_vocabulary():
    from backend.measure.auto_sweep import MEASURED, MISSING_SIGNATURE, PointMeasure
    from backend.measure.auto_sweep import MeasuredFormula

    with pytest.raises(ValueError, match="unknown unplaced reason"):
        PointMeasure(
            keys=(None,), launches=(0.0,), unkeyed=("unplaced",),
            labels=("custom_op",), reasons=("unknown",),
            formula=MeasuredFormula("0", (False,)),
            aggregation_ids=("custom_op",))
    assert PointMeasure(
        keys=(None,), launches=(0.0,), unkeyed=("unplaced",),
            labels=("custom_op",), reasons=(MISSING_SIGNATURE,),
            formula=MeasuredFormula("0", (False,)),
            aggregation_ids=("custom_op",)).unkeyed[0] == "unplaced"
    assert MEASURED == "measured"
