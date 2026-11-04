"""The leaf that writes a cached tensor writes it in the cache's format.

`traffic`'s contract has always said a cached tensor's width is its own
format; only the read side honoured it (`kv_reads`). The write — and every
same-step read of the edge the writer produces — was priced at the
activation width, so DeepSeek-V4-Flash's latent KV row (FP8 E4M3 with one
UE8M0 scale per 64 channels and its last 64 channels BF16, 583 B a token)
was billed as 1024 B of BF16. These pin the write against a figure derived
by hand from the format, not from the code that prices it.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval import cache as cache_mod
from backend.arch.eval import traffic as traffic_mod
from backend.arch.eval.precision import PRESETS
from backend.arch.eval.transforms import Deployment
from tests.snapshots import shared_program

TOKENS = 64


def _n(value) -> float:
    """A walked extent as the number it is at this point."""
    from backend.arch.eval.dim import value_of
    return float(value_of(value))


def _walk(name, profile=None):
    return Deployment(shared_program(name), precision=profile).at(
        batch=1, input_tokens=TOKENS).analyze()


def _writers(analysis):
    """`{writer leaf: cached tensor}` and the step's traffic."""
    return (cache_mod.landed_by(analysis, cache_mod.derive(analysis)),
            traffic_mod.derive(analysis))


def test_v4_latent_kv_write_is_the_583_byte_record():
    analysis = _walk("deepseek-ai__DeepSeek-V4-Flash")
    writers, flows = _writers(analysis)
    latent = {uid: t for uid, t in writers.items() if t.name == "kv_concat"}
    assert latent
    # By hand, from the format alone: 512 channels, the last 64 (d_qk_rope)
    # kept BF16, the other 448 FP8 E4M3 with one E8M0 byte per 64 of them.
    by_hand = 448 * 1 + 64 * 2 + (448 // 64) * 1
    assert by_hand == 583
    for uid, tensor in latent.items():
        assert _n(tensor.position_bytes) == by_hand
        written = flows.values[uid].act_writes
        assert _n(written) == TOKENS * by_hand, \
            analysis.program.by_uid(uid).display_path
        main = next(t for t in flows.values[uid].out_terms if t.port == "")
        assert main.dtype == "F8_E4M3" and main.cached == tensor


def test_fp8_kv_dense_model_writes_k_and_v_at_fp8():
    analysis = _walk("Qwen__Qwen3-8B", PRESETS["kv_fp8"])
    writers, flows = _writers(analysis)
    assert writers
    for uid, tensor in writers.items():
        # fp8_per_tensor: one byte an element, its one scale static.
        assert _n(tensor.written_width) == 1.0
        node = analysis.program.by_uid(uid)
        shape = analysis.shapes.values[uid]
        elements = _n(shape.written_elements())
        assert _n(flows.values[uid].act_writes) == elements, \
            node.display_path


def test_a_bf16_cache_model_writes_at_its_activation_width():
    analysis = _walk("Qwen__Qwen3-8B")
    writers, flows = _writers(analysis)
    precision = analysis.resolved_precision
    assert writers
    for uid in writers:
        node = analysis.program.by_uid(uid)
        act = precision.edge_bytes(node.role, analysis.program.stack_of(node))
        elements = _n(analysis.shapes.values[uid].written_elements())
        assert _n(flows.values[uid].act_writes) == elements * act
