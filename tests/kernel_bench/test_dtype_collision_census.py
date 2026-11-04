"""Read-only census coverage for the legacy dtype-blind restamp key."""
from __future__ import annotations

import json

from tools.dtype_collision_census import census_dtype_collisions, main


def _job(op: str, shape: str, dtype: str, bytes_: float, *, node="n",
         phase="decode", parallel="tp=1", model="org/model"):
    return {
        "model": model,
        "op": op,
        "node_spec_id": node,
        "parallel": parallel,
        "phase": phase,
        "physical_key": [op, shape, dtype, phase],
        "flops_per_rank": 0.0,
        "access_bytes_per_rank": bytes_,
    }


def test_census_reports_cross_dtype_same_geometry_price_collisions():
    shape = '{"B":1}'
    jobs = [
        _job("topk", shape, '{"activation":"bf16"}', 2.0),
        _job("topk", shape, '{"activation":"fp32"}', 4.0),
        # Repeated jobs at the same complete cell do not add exposure.
        _job("topk", shape, '{"activation":"bf16"}', 2.0),
        # A dtype change with the same price cannot miswrite a numerator.
        _job("mul", shape, '{"activation":"bf16"}', 2.0),
        _job("mul", shape, '{"activation":"fp32"}', 2.0),
        # A dtype-aware physical cell is not a legacy collision.
        _job("linear", shape, '{"activation":"bf16"}', 2.0),
        _job("linear", '{"B":2}', '{"activation":"fp32"}', 4.0),
    ]
    report = census_dtype_collisions([("org__model", jobs)])
    assert report["legacy_keys"] == 1
    assert report["dtype_cells"] == 2
    assert report["collisions"] == [{
        "model": "org/model", "op": "topk",
        "legacy_keys": 1, "dtype_cells": 2,
    }]
    assert report["models"][0]["ops"][0]["op"] == "topk"


def test_census_cli_writes_the_model_op_and_cell_counts(tmp_path, capsys):
    plans = tmp_path / "plans"
    plans.mkdir()
    jobs = [
        _job("topk", '{"B":1}', '{"activation":"bf16"}', 2.0),
        _job("topk", '{"B":1}', '{"activation":"fp32"}', 4.0),
        _job("scale_const", '{"B":1}', '{"activation":"bf16"}', 2.0),
        _job("scale_const", '{"B":1}', '{"activation":"fp32"}', 4.0),
    ]
    (plans / "org__model.jsonl").write_text(
        "".join(json.dumps(job) + "\n" for job in jobs), encoding="utf-8")
    out = tmp_path / "census.json"
    assert main(["--plans", str(plans), "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["legacy_keys"] == 2
    assert report["dtype_cells"] == 4
    assert [(row["model"], row["op"]) for row in report["collisions"]] == [
        ("org/model", "scale_const"), ("org/model", "topk")]
    assert "legacy_keys dtype_cells" in capsys.readouterr().out
