"""A card that admits no request publishes no step (`capacity.without_step`):
no batch, no per-op rows, no module table, no encode step, no fusion delta."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from backend.inference.capacity import without_step

ROWS = ("per_op_timing", "per_op_timing_decode", "per_op_timing_draft")


def _result(reason):
    phase = {"batch": 1, "t_sol_s": None, "flops_by_module": [{"uid": 1}],
             "time_unpublished_reason": reason or ""}
    return {
        "infeasibility_reason": reason, "max_batch_per_gpu": 0 if reason else 4,
        "phases": {"prefill": {"phase": "prefill", **phase},
                   "decode": {"phase": "decode", **phase}},
        "per_op_timing": [{"uid": 1}], "per_op_timing_decode": [{"uid": 1}],
        "vs_naive_delta": {"prefill": {}, "decode": {}},
        "encode": {"batch": 1, "stacks": [{"stack": "vision", "per_op_timing": [{"uid": 2}],
                                           "flops_by_module": [{"uid": 2}]}]},
        "weight_breakdown": {"total": 1},
    }


def test_a_card_that_admits_no_request_publishes_no_step():
    out = without_step(_result("weights do not fit"), ROWS)
    assert out["phases"] == {
        name: {"phase": name, "batch": None, "time_unpublished_reason": "weights do not fit"}
        for name in ("prefill", "decode")}
    for key in (*ROWS, "vs_naive_delta", "encode"):
        assert key not in out, key
    # What is not a step stays: the card's verdict and its weights.
    assert out["max_batch_per_gpu"] == 0 and out["weight_breakdown"] == {"total": 1}


def test_a_card_that_serves_is_published_whole():
    result = _result(None)
    assert without_step(result, ROWS) is result
