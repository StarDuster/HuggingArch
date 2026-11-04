"""A recurrent state is read where the step continues a sequence and written
back every step.

Serving keeps a linear-attention / SSM layer's state in a slot per request:
a decode step reads it and stores the updated state back (vLLM v0.32.0
`linear_attention_decode`, `fused_recurrent_gated_delta_rule` with
`inplace_final_state`; sglang v0.5.22 `seg_la_fwd`), and a prefill reads
it only for a sequence that already has positions — sglang's
`has_initial_states` is `extend_prefix_lens > 0`; vLLM gathers a new
sequence's slot and then zeroes it, Mamba2 not even gathering — while every
prefill stores the state its last token
leaves. A decode cell therefore moves twice the state, and a fresh prefill
the state once, as a write.

The sizes here are first principles off the checkpoint's config (heads x
d_k x d_v x fp32, the forward's `torch.zeros(batch, heads, d_k, d_v)`), not
the cache module's own arithmetic; a per-position KV cache in the same model
is the control, whose read is the cache length and whose write is the one
token the step lands.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval import cache
from backend.arch.eval.dim import value_of
from backend.arch.eval.traffic import derive
from backend.arch.eval.transforms import DECODE, PREFILL, Deployment
from tests.snapshots import shared_program, shared_walk, snapshot_config

MODEL = "Qwen/Qwen3.5-35B-A3B"
BATCH = 4


def _text():
    config = snapshot_config(MODEL)
    return config.get("text_config", config)


def _state_bytes() -> int:
    """One request's recurrent matrix, from the config: the HF forward's
    `torch.zeros(batch, num_v_heads, k_head_dim, v_head_dim)` at the
    `mamba_ssm_dtype` (float32)."""
    text = _text()
    assert text["mamba_ssm_dtype"] == "float32"
    return (text["linear_num_value_heads"] * text["linear_key_head_dim"]
            * text["linear_value_head_dim"] * 4)


def _walk(phase, **workload):
    deployment = Deployment(shared_program(MODEL)).at(batch=BATCH, **workload)
    return shared_walk(deployment.during(DECODE, cache_len=4096)
                       if phase == DECODE else deployment.during(PREFILL))


def _state_rows(analysis):
    """`(state tensor, its carrier's traffic)` for every recurrent matrix."""
    layer = cache.derive(analysis)
    traffic = derive(analysis)
    out = []
    for entry in layer.values.values():
        out.extend(
            (tensor, traffic.values[tensor.node])
            for tensor in entry.tensors
            if tensor.recurrent and tensor.kind == "recurrent_state"
        )
    assert out, f"{MODEL} keeps no recurrent state"
    return out


def _state_write(row, tensor) -> float:
    return sum(float(value_of(term.elements * term.width))
               for term in row.out_terms if term.cached == tensor)


def test_decode_reads_and_writes_back_the_whole_state():
    state = BATCH * _state_bytes()
    for tensor, row in _state_rows(_walk(DECODE, input_tokens=1024,
                                         output_tokens=2)):
        assert float(value_of(row.kv_reads)) == state
        assert _state_write(row, tensor) == state


def test_a_fresh_prefill_writes_the_state_and_reads_none():
    state = BATCH * _state_bytes()
    for tensor, row in _state_rows(_walk(PREFILL, input_tokens=1024,
                                         output_tokens=1)):
        assert float(value_of(row.kv_reads)) == 0
        assert _state_write(row, tensor) == state


def test_a_prefill_continuing_a_cached_prefix_reads_the_state():
    state = BATCH * _state_bytes()
    for tensor, row in _state_rows(_walk(PREFILL, input_tokens=1024,
                                         output_tokens=1,
                                         s_prefix_cached=512)):
        assert float(value_of(row.kv_reads)) == state
        assert _state_write(row, tensor) == state


def test_a_per_position_cache_is_read_at_its_length_and_written_one_token():
    """The control: the full-attention layers' K and V. A decode step lands
    the one token it adds (bf16, `num_key_value_heads x head_dim` per
    request) as its writer's only output, and the score and the apply each
    read every cached position — no state term on either side."""
    text = _text()
    s_kv = 4096
    row_bytes = text["num_key_value_heads"] * text["head_dim"] * 2
    analysis = _walk(DECODE, input_tokens=1024, output_tokens=2)
    layer = cache.derive(analysis)
    traffic = derive(analysis)
    writers = cache.landed_by(analysis, layer)
    assert writers and {t.kind for t in writers.values()} == {"kv"}
    for uid in writers:
        terms = traffic.values[uid].out_terms
        assert [float(value_of(t.elements * t.width)) for t in terms] \
            == [BATCH * row_bytes]
    readers = [uid for uid, access in cache.kv_access(analysis).values.items()
               if access.reads and uid not in
               {t.node for e in layer.values.values() for t in e.tensors}]
    assert readers              # each attention's score and apply
    for uid in readers:
        assert float(value_of(traffic.values[uid].kv_reads)) \
            == BATCH * s_kv * row_bytes
