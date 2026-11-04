"""A quantization scale is read exactly as far as the weight it scales.

A routed expert bank's weights are read only as far as the step's tokens
reach (`RowRoofline.stored_read`'s `min`); the scales stored beside those
weights are the same experts' scales, so they are read in the same
proportion. Billing the whole bank's scales on every decode step made an
INT4 group-quantized model (a BF16 scale per 32 weights) read more per MoE
layer than its FP8 sibling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.loader import SpecLoader
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from tests.snapshots import resolve_snapshot, shared_loader

pytestmark = pytest.mark.regression

#: INT4 group scales, block-FP8 scales, MXFP4 scales, and a BF16 bank.
_QUANTIZED_MOE = ("moonshotai/Kimi-K2-Thinking", "moonshotai/Kimi-K2-Instruct",
                  "openai/gpt-oss-20b")
_BF16_MOE = "Qwen/Qwen3-30B-A3B"


@pytest.fixture(scope="module")
def loader() -> SpecLoader:
    return shared_loader()


def _rows(model_id, loader, phase):
    return analyze(
        resolve_snapshot(model_id, loader), parallel=ParallelConfig(),
        batch_size=1,
        avg_input_tokens=2048, avg_output_tokens=0,
        phase=phase, breakdown=False).per_op_timing


def _routed(row) -> bool:
    return (row["weight_read_bytes_per_rank"]
            < row["weight_bytes_total_per_rank"])


@pytest.mark.parametrize("model_id", _QUANTIZED_MOE)
def test_routed_scales_are_read_with_their_weights(model_id, loader):
    """Batch-1 decode: every routed expert row reads its scales in the
    proportion it reads its weights."""
    rows = _rows(model_id, loader, "decode")
    routed = [r for r in rows if _routed(r)]
    assert routed, f"{model_id}: a batch-1 decode step routes to few experts"
    scaled = 0
    for r in routed:
        total_w = r["weight_bytes_total_per_rank"]
        total_m = r["metadata_bytes_total_per_rank"]
        want = r["weight_read_bytes_per_rank"] * total_m / total_w
        # Both sides are truncated to whole bytes from the same fraction.
        assert abs(r["metadata_bytes_per_rank"] - want) <= 1 + total_m / total_w, \
            r["aggregation_id"]
        scaled += total_m > 0
    assert scaled, f"{model_id}: its routed experts carry scales"


@pytest.mark.parametrize("model_id", (*_QUANTIZED_MOE, _BF16_MOE))
def test_full_coverage_reads_every_scale(model_id, loader):
    """A dense weight, and every row of a prefill, is read whole: its scales
    too, to the byte — except a stored-table lookup, which reads the rows it
    picks (the `Lookup` law: min(input, output) elements, one per element
    written; `embed_tokens` at prefill gathers the step's token rows, not
    the whole vocabulary)."""
    for phase in ("prefill", "decode"):
        for r in _rows(model_id, loader, phase):
            if r.get("op") == "embedding":
                continue
            if _routed(r):
                assert phase == "decode", r["aggregation_id"]
                continue
            assert (r["metadata_bytes_per_rank"]
                    == r["metadata_bytes_total_per_rank"]), \
                (phase, r["aggregation_id"])


def test_int4_experts_read_less_than_fp8_at_equal_coverage(loader):
    """Kimi-K2-Thinking (INT4 experts, a BF16 scale per 32 weights) and
    Kimi-K2-Instruct (block-FP8 experts) route the same way; at batch-1
    decode the INT4 experts read fewer bytes, scales included."""
    def expert_read(model_id):
        return sum(r["weight_read_bytes_per_rank"] + r["metadata_bytes_per_rank"]
                   for r in _rows(model_id, loader, "decode") if _routed(r))

    int4 = expert_read("moonshotai/Kimi-K2-Thinking")
    fp8 = expert_read("moonshotai/Kimi-K2-Instruct")
    assert 0 < int4 < fp8, (int4, fp8)
