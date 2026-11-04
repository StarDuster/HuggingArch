"""Module bench latency sweep — the bench's own timing, at more lengths.

A sweep point at the bench's length must be the number the bench's Summary
row shows (same `_timing_view`, not a second pricing), each point says what
its phase bound, and an entry the bench refuses is refused here the same way.
"""

from __future__ import annotations

import pytest

from backend.arch.eval.precision import AS_SHIPPED
from backend.arch.eval.transforms import DECODE, PREFILL
from backend.arch.loader import get_default_loader
from backend.playground import latency_sweep as ls
from backend.playground import module_bench as mb
from tests.snapshots import resolve_snapshot

MODEL = "deepseek-ai/DeepSeek-V3.2"
ENTRY = {"name": "mla_dsa_attention", "model_id": MODEL}
L = 2048


def _run(sweep: ls.LatencySweep):
    return sweep.assemble([ls.cell(item) for item in sweep.cells()])


def test_sweep_point_at_the_bench_length_is_the_bench_time(monkeypatch):
    resolve_snapshot(MODEL, get_default_loader())  # SkipTest if absent
    monkeypatch.setattr(mb, "_measurement_scope", lambda: None)
    bench = mb.module_bench(entries=[dict(ENTRY)], batch=2, input_tokens=L,
                            precision=AS_SHIPPED, gpu_sku="B200")
    (row,) = bench["results"]
    out = _run(ls.LatencySweep(
        entries=(dict(ENTRY), {"name": "not_a_component", "model_id": MODEL}),
        batch=2, lens=(1024, L), gpu_sku="B200", precision=AS_SHIPPED))

    assert [e["entry"] for e in out["errors"]] == [f"not_a_component@{MODEL}"]
    (swept,) = out["results"]
    assert swept["entry"] == row["entry"]
    for phase in (PREFILL, DECODE):
        points = swept["phases"][phase]
        assert [p["len"] for p in points] == [1024, L]
        at_l = points[1]
        assert at_l["t_sol_s"] == pytest.approx(row["timings"][phase]["t_sol_s"])
        # The binding each point ran under is the bench's phase contract.
        assert {k: at_l[k] for k in ("batch", "s_query", "s_kv")} == \
            bench["contract"]["phases"][phase]
        # A longer context never makes either phase cheaper.
        assert points[0]["t_sol_s"] <= points[1]["t_sol_s"]
    # Intensity is the same kernels' FLOPs over their HBM access, and a
    # prefill of L tokens reuses each weight / KV byte far more than one
    # decode token against L cached positions.
    for phase in (PREFILL, DECODE):
        for p in swept["phases"][phase]:
            assert p["arithmetic_intensity"] == pytest.approx(p["flops"] / p["access_bytes"])
    assert swept["phases"][PREFILL][1]["arithmetic_intensity"] > \
        swept["phases"][DECODE][1]["arithmetic_intensity"]
