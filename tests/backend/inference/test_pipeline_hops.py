"""What each pipeline stage sends, in theory (no framework): the worked
derivations of the PP placement design (PP_LAYOUT_PROPOSAL §3).

Prefill is B = 1, S = 8192; decode is B = 64 over an 8192-token cache.
Bytes are per rank, summed over the stage's `send_recv` rows: each hop is
the producer's local tensor, sent once per cut it crosses.
"""
from __future__ import annotations

from functools import lru_cache

import pytest

from backend.inference import pipeline
from backend.inference.parallel_config import ParallelConfig
from tests.snapshots import resolve_snapshot, shared_loader

S = 8192


@lru_cache(maxsize=None)
def _sent(model_id, pp, tp, stage, phase="prefill", timed=None):
    parallel = ParallelConfig(pp=pp, tp_attention=tp, tp_dense_ffn=tp,
                              tp_moe_expert=tp)
    rows = pipeline.analyze(
        resolve_snapshot(model_id, shared_loader()), parallel=parallel,
        batch_size=1 if phase == "prefill" else 64,
        avg_input_tokens=S, avg_output_tokens=0,
        phase=phase, stage=stage, timed_stacks=timed, breakdown=False).per_op_timing
    return round(sum(float(r.get("comm_bytes_per_rank") or 0)
                     for r in rows if r.get("comm_op") == "send_recv"))


@pytest.mark.parametrize("model_id, pp, tp, prefill, decode", [
    # (a) the carry layers[17] -> layers[18], [1, 8192, 4096] bf16
    ("Qwen/Qwen3-8B", 2, 4, 67_108_864, 524_288),
    # (b) [1, 8192, 7168]; the MTP head reads the norm on stage 1: no hop
    ("deepseek-ai/DeepSeek-V3", 2, 4, 117_440_512, 917_504),
    # (f) the hyper-connection copies are an axis: [1, 8192, 4, 4096]
    ("deepseek-ai/DeepSeek-V4-Flash", 2, 4, 268_435_456, 2_097_152),
    # (g) [1, 8192, 3072]
    ("meta-llama/Llama-3.2-3B", 2, 4, 50_331_648, 393_216),
])
def test_one_cut_sends_the_carry(model_id, pp, tp, prefill, decode):
    assert _sent(model_id, pp, tp, 0) == prefill
    assert _sent(model_id, pp, tp, 0, "decode") == decode


@pytest.mark.regression  # minutes-long
def test_the_attnres_bank_relays_each_element_to_its_reader():
    """(c) Kimi-K3 pp4 x tp8 (layers 0-22 | 23-45 | 46-69 | 70-92): the
    `block_residual` bank's elements are read up to `output_attn_res` on
    stage 3, so each crosses every cut from its writer, once per cut beside
    the hidden state: stage 0 sends hidden, e0, e1; stage 1 hidden, e0-e3;
    stage 2 hidden, e0-e5; stage 3 nothing. Each is [1, 8192, 7168] bf16,
    117,440,512 B."""
    assert [_sent("moonshotai/Kimi-K3", 4, 8, s) for s in range(4)] == [
        352_321_536, 587_202_560, 822_083_584, 0]


def test_a_shared_indexer_reads_its_top_k_across_the_cut():
    """(d) GLM-5.2: at pp2 layer 39 is shared, so the int32 top-k crosses
    beside the hidden state; at pp3 layer 26 is a full layer (no top-k
    crosses cut 0) and layer 52 shared (it crosses cut 1)."""
    assert _sent("zai-org/GLM-5.2", 2, 4, 0) == 100_663_296 + 67_108_864
    assert [_sent("zai-org/GLM-5.2", 3, 4, s) for s in (0, 1)] == [
        100_663_296, 167_772_160]


def test_an_encoder_element_is_handed_over_not_sent():
    """(e) Qwen3-VL pp32 x tp1 (stage 0 = layers 0-1): the deepstack element
    layer 2 adds is an encoder's output, handed to the language model like
    an input (`Program.handoffs`), so no cut carries it; only the hidden
    state crosses cut 0."""
    assert _sent("Qwen/Qwen3-VL-235B-A22B-Instruct", 32, 1, 0) == 67_108_864


def test_an_mtp_head_reads_the_embedding_across_the_cut_when_timed():
    """(f) DeepSeek-V4-Flash: the MTP head reads `('layers', 'embed')` and
    runs on stage 1, so the embedding crosses cut 0 for it — priced only
    when the MTP is timed."""
    timed = ("", "mtp")
    assert _sent("deepseek-ai/DeepSeek-V4-Flash", 2, 4, 0,
                 timed=timed) == 268_435_456 + 67_108_864
    assert _sent("deepseek-ai/DeepSeek-V4-Flash", 2, 4, 0, "decode",
                 timed=timed) == 2_097_152 + 524_288


def test_empty_stages_send_nothing():
    """Llama-3.2-1B, 16 layers at pp32: layers 0-15 sit on stages 15-30.
    Stages 0-14 and 31 run nothing and send nothing; stage 30 runs the last
    layer and the head, and has no one to send to."""
    model = "meta-llama/Llama-3.2-1B"
    assert [_sent(model, 32, 1, s) for s in (0, 14, 30, 31)] == [0] * 4
    assert _sent(model, 32, 1, 15) == 33_554_432


def _hops(model_id, pp, stage, phase):
    """The stage's `send_recv` rows as `(source path, payload shape, width,
    bytes)`, tp=1."""
    rows = pipeline.analyze(
        resolve_snapshot(model_id, shared_loader()),
        batch_size=1 if phase == "prefill" else 64,
        avg_input_tokens=S, avg_output_tokens=0,
        parallel=ParallelConfig(pp=pp), phase=phase, stage=stage,
        breakdown=False,
        ).per_op_timing
    return [(r["source_occurrence"]["node"], r["comm_payload_shape"],
             r["comm_payload_dtype_bytes"], int(r["comm_bytes_per_rank"]))
            for r in rows if r.get("comm_op") == "send_recv"]


def test_a_shared_cache_read_past_a_cut_sends_what_its_reader_reads():
    """(h) DeepSeek-V4.1-Flash at pp4: layer 8 writes the compressed KV that
    layers 10-13 read (`reads_from`), and layer 9 ends stage 0. Each step
    the cut carries what those readers read of that cache — its read law,
    top-512 of the pooled rows at decode — not the one step's output, which
    at decode is half a pooled row. Prefill reads what it just wrote: the
    4096 pooled rows. At the cache's own width, NVFP4 (half a byte), per
    request."""
    compressed = "attn.compressed.compress_read"
    decode = [h for h in _hops("deepseek-ai/DeepSeek-V4.1-Flash", 4, 0,
                               "decode") if h[0].startswith(compressed)]
    assert [h[1:] for h in decode] == [([64, 512, 512], 0.5, 8_388_608)]
    prefill = [h for h in _hops("deepseek-ai/DeepSeek-V4.1-Flash", 4, 0,
                                "prefill") if h[0].startswith(compressed)]
    assert [h[1:] for h in prefill] == [([1, 4096, 512], 0.5, 1_048_576)]
