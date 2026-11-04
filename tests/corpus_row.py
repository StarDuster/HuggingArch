"""One complete, schema-current corpus row for tests to start from.

Five test modules each kept their own hand-written copy of "a valid row" —
`test_benchmarks`, `test_inference_api`, `test_calibration_overlay_contracts`,
`test_op_calibration`, `tests/service/test_bench_routes` — and every schema
version had to find all five. v5 (`kernel_symbol`) found none of them: the
copies simply stopped listing every column, and 88 tests failed on a `KeyError`
naming a column the schema had had for an hour.

The copies also drifted in a quieter way. Each said `schema_version: "2"` while
carrying v4's columns, so the fixtures described a row shape that has never
existed — harmless for what those tests assert, and exactly the kind of thing
that is true nowhere and believed everywhere.

So: one row here, `ALL_COLUMNS`-complete by construction, and a `row(**over)`
that takes the fields a test actually cares about. A new column added to the
schema without a value here fails ONE assertion in this module rather than
stranding five fixtures.
"""
from __future__ import annotations

import csv as _csv
import io
from typing import Dict, List

from backend.inference import benchmark_identity as _benchmark_identity
from backend.inference import benchmarks as _benchmarks

# A plausible measured attention cell. The values are only required to be
# schema-legal — a test that depends on a particular number should override it,
# not read one out of here and hope.
_BASE: Dict[str, str] = {
    "model": "deepseek-ai/DeepSeek-V3",
    "spec_hash": "a" * 16,
    "phase": "prefill",
    "parallel": "tp=8",
    "framework": "vllm",
    "framework_version": "0.21.0",
    "gpu": "H100-SXM",
    "op": "flash_attention",
    "node_spec_id": "self_attn",
    # Both JSON identity columns come from the canonical producers, not from a
    # hand-typed literal. They are matched BYTE-for-byte, so key order decides
    # whether a lookup hits — and the hand-written copies were not in the
    # producer's order, so a fixture row could be indexed under a key nothing
    # that goes through `merge_validated_rows` will ever build.
    "shape_json": _benchmark_identity.compute_per_row_shape_json(
        batch_size=1, seq_len=2048,
        dim_values={"n_heads": 24, "d_head": 128}),
    # An attention kernel: no weight of its own, a BF16 cache.
    "dtype_env_json": _benchmark_identity.compute_per_row_dtype_env_json(
        weight_dtype=None, activation_dtype="bf16", kv_dtype="bf16",
        metadata_dtype=None),
    "kernel_impl": "flash_attention_v3",
    "latency_us": "387.5",
    "p50_us": "385.0",
    "p95_us": "410.0",
    "samples": "1000",
    "count_per_step": "28",
    "achieved_tflops": "690.0",
    "achieved_bw_gbps": "920.0",
    "peak_tflops_at_test": "989.0",
    "peak_bw_at_test": "3350.0",
    "schema_version": str(_benchmarks.KNOWN_SCHEMA_VERSION),
    "contributor": "@huggingarch-seed",
    "date": "2026-05-16",
    "sm": "sm_90",
    "cross_check": "passed",
    # The evidence behind that verdict: a relative error large enough to look
    # alarming on its own, next to the cosine that shows the kernel is
    # computing the right thing. Reading either alone is what the pair exists
    # to prevent.
    "cross_check_rel_err": "0.24",
    "cross_check_cosine": "0.999300",
    # And the two the verdict was ACTUALLY made on, inside the fp8 band
    # `Band(1.1e-1, 3.1)` this row's dtype carries. Note they sit well
    # inside while `rel_err` above reads 0.24: that gap is the whole
    # reason these columns exist, and this row is the canonical example
    # of a pass that looks alarming through the diagnostic and is
    # comfortable through the gate.
    "xcheck_typical": "0.0162",
    "xcheck_worst": "0.0914",
    # What the device ran. A real one, not a placeholder, so a test that
    # asserts on the column's shape (`<digest>:<name>`) has something to assert.
    "kernel_symbol": ("38ea398b357d:void pytorch_flash::flash_fwd_kernel"
                      "<Flash_fwd_kernel_traits<128, 64, 64, 4, false, false"),
    # v7 provenance: which sweep, which box, what routing load. A fixture row
    # carries plausible values so a test can assert on the columns' shape.
    "run_id": "20260516T000000Z-testhost-1",
    "box": "testbox",
    "measured_load_factor": "",
    "moe_block_member": "",
    # v9: the seed this row's inputs were drawn at, in the column's own form
    # (decimal unsigned 64-bit, as a string).
    "input_seed": "17236409675464283217",
    # Empty: this row's kernel computed at the cell's own precision. A
    # dequantising provider fills it (`Provider.computes_at`).
    "measured_dtype_env_json": "",
    # Empty: this row's provider returns a bare closure, so no call was
    # rendered. A `Launch` fills it with the call and its operands.
    "launch_signature": "",
    # Empty reads as `"measured"`, which this row is — it has numbers. A
    # refusal (`outcome="oom"`) carries none.
    "outcome": "",
    # Empty: a measurement has nothing to explain. A row whose outcome is a
    # failed attempt (`error`, `timeout`, …) says why here.
    "outcome_detail": "",
    # Empty: timed before the clock recorded itself (the old memset flush) —
    # what every kept row says. A row timed now carries `l2_drain`.
    "timing_method": "",
    # v14: which driver ran the row. The SAME string twice, which is the
    # ordinary case and what `is_driver_substituted` reads as `False`: this
    # process loaded the driver its kernel module shipped. The value is the
    # real kernel driver of the H100 box this row's `gpu` names.
    #
    # Native on purpose. A fixture that silently described a SUBSTITUTED
    # driver would make every test that starts here quietly assert the
    # exceptional case; a test about the substitution overrides
    # `umd_version` and says so.
    "kmd_version": "575.57.08",
    "umd_version": "575.57.08",
}

# The point of the module: a column the schema gained and this row did not is
# one failure here, not five elsewhere. At import, so it cannot be skipped.
_missing = [c for c in _benchmarks.ALL_COLUMNS if c not in _BASE]
_extra = [c for c in _BASE if c not in _benchmarks.ALL_COLUMNS]
if _missing or _extra:
    raise AssertionError(
        f"tests/corpus_row.py is out of step with the schema: "
        f"missing {_missing}, unknown {_extra}")


def row(**over: str) -> Dict[str, str]:
    """A complete row, with `over` applied. Fresh dict per call."""
    unknown = [k for k in over if k not in _benchmarks.ALL_COLUMNS]
    if unknown:
        raise AssertionError(f"not corpus columns: {unknown}")
    out = dict(_BASE)
    out.update(over)
    return out


def to_csv(rows: List[Dict[str, str]]) -> str:
    """`rows` as CSV text in the canonical column order."""
    buf = io.StringIO()
    w = _csv.writer(buf, lineterminator="\n")
    w.writerow(_benchmarks.ALL_COLUMNS)
    for r in rows:
        w.writerow([r[c] for c in _benchmarks.ALL_COLUMNS])
    return buf.getvalue()
