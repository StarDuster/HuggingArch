"""CP prefill completes on the shapes that used to refuse it.

One case per fixed defect, each a model
whose CP=2 prefill raised before the fix:

* D1 — a recurrence checked its laws on arriving layouts before gathering
  its whole sequence (Nemotron's mamba2_scan);
* D2 — an elementwise op refused one sharded and one whole operand (Ling's
  GLA output gate);
* D3 — the routed-rows mean was cut by index arithmetic (Kimi-K2: 128/3;
  DeepSeek-V3 at s=96: an odd integer mean, 3);
* D4 — a gather on a stage's sequential inbound named no producer
  (MiMo-V2.5's vision merger);
* D5 — an elementwise op on a per-head parameter read the global B·s_ctx
  (Qwen3-Coder-Next's and Kimi-K3's `neg_exp` on `A_log`).
"""
from __future__ import annotations

import pytest

from tests.snapshots import resolve_snapshot

CASES = [
    ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", 2, 1024),
    ("inclusionAI/Ling-2.6-1T", 2, 1024),
    ("moonshotai/Kimi-K2-Base", 2, 1024),
    ("deepseek-ai/DeepSeek-V3", 1, 96),
    ("XiaomiMiMo/MiMo-V2.5", 2, 1024),
    ("Qwen/Qwen3-Coder-Next", 2, 1024),
    ("moonshotai/Kimi-K3", 2, 1024),
]


@pytest.mark.parametrize("model_id,batch,seq", CASES)
def test_cp_prefill_completes(model_id, batch, seq):
    from tests.snapshots import shared_loader
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.pipeline import analyze
    resolved = resolve_snapshot(model_id, shared_loader())
    analyze(resolved, parallel=ParallelConfig(cp=2), batch_size=batch,
            avg_input_tokens=seq, avg_output_tokens=0, phase="prefill", breakdown=False)


def test_routed_rows_split_as_a_mean_under_cp():
    """Kimi-K2: 384 experts, top-8, B=2, s=1024 → M = 128/3 rows per expert.
    Under CP=2 each rank routes half the tokens, so every routed expert op
    costs exactly half per rank — the mean splits, it is not refused."""
    from fractions import Fraction
    from tests.snapshots import shared_loader
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.pipeline import analyze
    resolved = resolve_snapshot("moonshotai/Kimi-K2-Base", shared_loader())
    one, two = (
        {r["aggregation_id"]: r for r in analyze(
            resolved, parallel=ParallelConfig(cp=cp), batch_size=2,
            avg_input_tokens=1024, avg_output_tokens=0, phase="prefill", breakdown=False).per_op_timing}
        for cp in (1, 2))
    routed = [k for k in one if ".experts." in k and one[k]["flops_per_rank"]]
    assert routed
    for k in routed:
        assert Fraction(two[k]["flops_per_rank"]) / Fraction(one[k]["flops_per_rank"]) == Fraction(1, 2), k
