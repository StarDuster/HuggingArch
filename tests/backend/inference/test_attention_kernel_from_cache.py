"""A fused attention row is measured under the kernel that reads the cache
its keys come from (`KVAccess.key_slots`, slot kinds' `kernel:` in
cache_families.yaml): the absorbed MLA decode and the dense MLA prefill both
read a latent, GQA reads K/V, and DSA's sparse span is its own op."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.benchmark_identity import RowIdentity, resolve_op
from backend.inference.framework_bindings import DEFAULT_FRAMEWORK, get_binding
from backend.inference.op_signatures import op_signature
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from tests.snapshots import resolve_snapshot, shared_loader

CASES = (
    ("deepseek-ai/DeepSeek-V2-Lite", "mla_attention"),
    # Kimi-K3's gated MLA pair: ~20 s a phase, so with the regression set.
    pytest.param("moonshotai/Kimi-K3", "mla_attention",
                 marks=pytest.mark.regression),
    ("Qwen/Qwen3-8B", "flash_attention"),
    ("deepseek-ai/DeepSeek-V3.2", "flash_dsa"),
)


@pytest.mark.parametrize("phase", ("prefill", "decode"))
@pytest.mark.parametrize("model_id,kernel", CASES)
def test_attention_rows_key_to_the_kernel_reading_their_cache(
        model_id, kernel, phase):
    result = analyze(resolve_snapshot(model_id, shared_loader()),
                     parallel=ParallelConfig(), avg_input_tokens=64,
                     avg_output_tokens=0, phase=phase,
                     framework_binding=get_binding(DEFAULT_FRAMEWORK), breakdown=True)
    identity = RowIdentity(result)
    ops = {identity.of(row)[0] for row in result.per_op_timing}
    attention = {"gqa_attention", "mla_attention", "mla_sparse_attention"}
    assert {op for op in ops if op_signature(op) in attention} == {kernel}


def test_slots_naming_no_one_kernel_are_refused():
    for slots in ({"index_key"}, {"kv", "latent"}):
        with pytest.raises(ValueError, match="no one kernel"):
            resolve_op("flash_attention", key_slots=frozenset(slots))
    assert resolve_op("rmsnorm", key_slots=frozenset({"latent"})) == "rmsnorm"
