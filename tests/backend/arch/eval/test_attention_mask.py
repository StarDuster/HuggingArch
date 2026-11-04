"""Which keys an attention sees: the mask it declares, and the keys one query
attends against the keys one step reads.

* `is_causal` is the attention's own declaration (HF `self.is_causal`; vLLM
  encoder attention `causal=False`). A causal attention takes the phase's
  active fraction; a bidirectional one sees its whole matrix in every phase.
* A windowed score matrix's key axis is `s_attend`, one query's window
  `Min(s_kv, s_window)`; the K / V operands' rows are `s_kv`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import DECODE, Deployment
from backend.arch.ops_base import _pairs
from tests.snapshots import shared_loader, shared_program, shared_walk

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _program(stem: str) -> Program:
    return shared_program(stem.replace('__', '/', 1))


def test_a_causal_attention_scores_the_frames_pairs():
    frame = {"s_query": 4096, "s_kv": 4096, "s_pairs": 4096 * 4097 // 2}
    assert _pairs(frame) == 4096 * 4097 // 2
    assert _pairs({**frame, "is_causal": 1}) == 4096 * 4097 // 2


def test_a_bidirectional_attention_sees_its_whole_matrix_in_every_phase():
    for pairs in (4096 * 4097 // 2, 4096, sp.Symbol("s_pairs")):
        frame = {"s_query": 4096, "s_kv": 4096, "s_pairs": pairs}
        assert _pairs({**frame, "is_causal": 0}) == 4096 * 4096
    symbol = sp.Symbol("s_pairs")
    assert _pairs({"s_query": 1, "s_kv": 1, "s_pairs": symbol}) == symbol


def test_the_windowed_score_key_axis_is_one_querys_window():
    """gpt-oss-20b's sliding layers (window 128), decoding against a 4096
    cache: one query attends 128 keys, and the score matrix says so."""
    program = _program("openai__gpt-oss-20b")
    window = 128
    analysis = (shared_walk(Deployment(program).at(batch=1, input_tokens=4096)
                .during(DECODE, cache_len=4096)))
    scores = [n for n in program.nodes()
              if n.op == "swa_attention_score" and not n.children]
    assert scores
    for node in scores:
        shape = analysis.shapes.values[node.uid]
        assert int(value_of(shape.out[-1])) == window
        # one decode query reads the window it attends, no more
        assert int(value_of(shape.ins[1][2])) == window


# -- A1: encoder attention is bidirectional ----------------------------------

def test_the_qwen3_vl_vision_score_is_the_whole_matrix():
    """HF Qwen3-VL vision attention runs `is_causal=False`
    (modeling_qwen3_vl.py:260, 300, 319); vLLM's MMEncoderAttention does too.
    The text attention of the same model stays causal. A 1024x1024 image in
    16x16 patches is one segment of 4096 rows, each attending all 4096."""
    program = _program("Qwen__Qwen3-VL-235B-A22B-Instruct")
    tokens = 4096
    encode = (shared_walk(Deployment(program)
              .at(batch=1, media={"pixel_values": {
                  "n_img": 1, "img_h": 1024, "img_w": 1024}})
              .during("encode", stack="visual")))
    assert (1024 // int(encode.binding["s_patch"])) ** 2 == tokens
    heads = int(encode.binding["n_vis_heads"])
    head_dim = int(encode.binding["d_vis_head"])
    # The image tower's own scores; a video call writes the same tower out
    # again under its own stack, which an image-only encode does not run.
    vision = [n for n in program.nodes()
              if n.op == "attention_score" and program.stack_of(n) == "visual"]
    text = [n for n in program.nodes()
            if n.op == "attention_score" and n.uid not in program.encoder_uids]
    assert vision and text
    for node in vision:
        assert int(encode.cost.values[node.uid].flops) == (
            2 * heads * tokens * tokens * head_dim)
    analysis = shared_walk(Deployment(program).at(batch=1, input_tokens=tokens))
    dims = analysis.binding
    q_heads, d_head = int(dims["n_heads"]), int(dims["d_head"])
    for node in text:
        assert analysis.cost.values[node.uid].flops == (
            2 * q_heads * d_head * tokens * (tokens + 1) // 2)


def test_a_node_does_not_bind_the_phase_pairs():
    import pytest
    from backend.arch.schema.graph import GraphNode
    with pytest.raises(ValueError, match="is_causal"):
        GraphNode(op="softmax", inputs=["s"], dims={"s_pairs": "1"})
    GraphNode(op="softmax", inputs=["s"], dims={"is_causal": "0"})


def _overlay_without_mask(tmp_path: Path, stem: str) -> object:
    """Qwen3-VL whose vision block is the library block minus its mask
    declaration, written as the model's own inline block."""
    import yaml
    from backend.arch.loader import SpecLoader, overlay_kwargs
    lib = (PROJECT_ROOT / "backend" / "arch" / "components" / "library"
           / "qwen35_vision_block.yaml").read_text()
    block = yaml.safe_load(lib.replace('is_causal: "0", ', ""))
    model = yaml.safe_load((PROJECT_ROOT / "backend" / "arch" / "models"
                            / f"{stem}.yaml").read_text())
    model.setdefault("blocks", {})["unmasked_vision_block"] = {
        k: block[k] for k in ("description", "graph", "outputs")}
    for layer in model["model"]:
        for sub in layer.get("layers", []) if isinstance(layer, dict) else []:
            if sub.get("template") == "qwen35_vision_block":
                sub["template"] = "unmasked_vision_block"
    (tmp_path / "models").mkdir()
    (tmp_path / "components" / "drafts").mkdir(parents=True)
    (tmp_path / "models" / f"{stem}.yaml").write_text(yaml.safe_dump(model))
    config = json.loads((SNAPSHOTS / f"{stem}.json").read_text())["config"]
    loader = SpecLoader(**overlay_kwargs(str(tmp_path)))
    return loader.match(config, model_id=stem.replace("__", "/", 1))


def test_an_encoder_attention_must_state_its_mask(tmp_path):
    from backend.arch.validate._attention_masks import run_attention_mask_check
    stem = "Qwen__Qwen3-VL-235B-A22B-Instruct"
    assert run_attention_mask_check(
        shared_loader().match(
            json.loads((SNAPSHOTS / f"{stem}.json").read_text())["config"],
            model_id=stem.replace("__", "/", 1))) == []
    findings = run_attention_mask_check(_overlay_without_mask(tmp_path, stem))
    kinds = {f.kind for f in findings}
    assert kinds == {"encoder_mask_undeclared"}
    assert {f.op for f in findings} == {"attention_score", "softmax",
                                         "attention_apply"}
    assert all('is_causal: "0"' in f.detail for f in findings)


# -- A2: a sliding window's pairs and its step read ---------------------------

def _brute_pairs(q: int, n: int, w: int, causal: int, chunked: int) -> int:
    """Live pairs over the last `q` of `n` positions, by enumeration."""
    total = 0
    for i in range(n - q, n):
        for j in range(n):
            if chunked:
                live = i // w == j // w and (j <= i or not causal)
            elif causal:
                live = i - w < j <= i
            else:
                live = abs(i - j) <= (w - 1) // 2
            total += live
    return total


def _law(q: int, keys: int, w: int, causal: int, chunked: int) -> int:
    from backend.arch.ops_registry import _window_pairs
    return int(_window_pairs({"s_query": q, "s_kv": keys, "s_window": w,
                              "is_causal": causal, "is_chunked": chunked}))


def test_the_causal_window_law_counts_every_live_pair():
    """vLLM FlashAttention runs a causal window as (W − 1, 0)
    (v1/attention/backends/flash_attn.py:895): position p sees min(p + 1, W)
    keys. Exact over the step's union and over the whole context alike."""
    for w in (1, 2, 3, 5, 16):
        for n in range(1, 40):
            for q in range(1, n + 1):
                want = _brute_pairs(q, n, w, 1, 0)
                assert _law(q, n, w, 1, 0) == want
                assert _law(q, min(n, q + w - 1), w, 1, 0) == want


def test_the_bidirectional_band_and_the_chunks_count_every_live_pair():
    """A non-causal window is symmetric (vLLM `_maybe_symmetrize_window`;
    MiMo `sliding_window_q = sliding_window_k = w`): |i − j| ≤ w. Chunks:
    a query sees its own chunk (MiMo audio's groups of `group_size`)."""
    for w in (1, 3, 5, 9, 129):
        for s in (1, 2, 7, 64, 200):
            assert _law(s, s, w, 0, 0) == _brute_pairs(s, s, w, 0, 0)
    for g in (1, 2, 3, 4, 7):
        for s in (1, 3, 4, 9, 64):
            assert _law(s, s, g, 0, 1) == _brute_pairs(s, s, g, 0, 1)


def test_the_key0_bias_lands_where_key_0_is_in_the_window():
    """vLLM triton_prefill_attention: `qk + sink` only where the mask holds
    and `pos_k == 0`."""
    from backend.arch.ops_registry import _window_key0_queries
    for w in (1, 3, 129):
        for s in (1, 10, 4096):
            got = _window_key0_queries({"s_query": s, "s_window": w,
                                        "is_causal": 0, "is_chunked": 0})
            assert int(got) == sum(1 for i in range(s) if i <= (w - 1) // 2)


def test_a_prefill_window_reads_every_key_once():
    """gpt-oss-20b sliding layers (window 128), prefill 4096: one query
    attends 128 keys, the step reads all 4096 of them; FLOPs are the causal
    band's pairs, 2·H·d·(S·W − W(W − 1)/2)."""
    program = _program("openai__gpt-oss-20b")
    tokens, window = 4096, 128
    analysis = shared_walk(Deployment(program).at(batch=1, input_tokens=tokens))
    heads, d_head = int(analysis.binding["n_heads"]), int(analysis.binding["d_head"])
    for node in program.nodes():
        if node.op not in ("swa_attention_score", "swa_attention_apply"):
            continue
        shape = analysis.shapes.values[node.uid]
        assert int(value_of(shape.ins[1][2])) == tokens          # K / V rows
        if node.op == "swa_attention_score":
            assert int(value_of(shape.out[-1])) == window          # s_attend
        assert analysis.cost.values[node.uid].flops == (
            2 * heads * d_head * (tokens * window - window * (window - 1) // 2))


def test_a_window_as_wide_as_the_context_is_dense_causal():
    """gemma-2's window is 4096: at a 4096-token prefill the windowed score
    costs the dense causal triangle."""
    program = _program("google__gemma-2-9b")
    tokens = 4096
    analysis = shared_walk(Deployment(program).at(batch=1, input_tokens=tokens))
    heads, d_head = int(analysis.binding["n_heads"]), int(analysis.binding["d_head"])
    scores = [n for n in program.nodes() if n.op == "swa_attention_score"]
    assert scores
    for node in scores:
        assert analysis.cost.values[node.uid].flops == (
            2 * heads * d_head * tokens * (tokens + 1) // 2)


# -- A3: MiMo-V2.5 vision ---------------------------------------------------

def test_the_mimo_vision_window_is_a_bidirectional_band():
    """vLLM mimo_v2_omni.py `_forward_window_attn`: context_attention_fwd(...,
    is_causal=False, sliding_window_q=w, sliding_window_k=w,
    sinks_bias_key0=True); the kernel masks |i − j| ≤ w
    (triton_prefill_attention.py:149-159) and adds the sink where
    `pos_k == 0`. Score pairs are the band's exactly, S(2w+1) − w(w+1); the
    key-0 bias lands on the w + 1 queries whose window holds key 0. Both
    block kinds rotate q and k (:287-290). A 1024x1024 image in 16x16
    patches is one segment of 4096 rows."""
    stem = "XiaomiMiMo__MiMo-V2.5"
    config = json.loads((SNAPSHOTS / f"{stem}.json").read_text())["config"]
    reach = config["vision_config"]["visual_token_window_size"]
    program = _program(stem)
    tokens = 4096
    analysis = (shared_walk(Deployment(program)
                .at(batch=1, media={"pixel_values": {
                    "n_img": 1, "img_h": 1024, "img_w": 1024}})
                .during("encode", stack="visual")))
    assert (1024 // int(analysis.binding["s_patch"])) ** 2 == tokens
    heads = int(analysis.binding["n_vis_heads"])
    head_dim = int(analysis.binding["d_vis_head"])
    pairs = tokens * (2 * reach + 1) - reach * (reach + 1)
    vision = [n for n in program.nodes()
              if n.uid in program.encoder_uids and not n.children
              and (n.stack_name or "").startswith("visual")]
    scores = [n for n in vision if n.op == "swa_attention_score"]
    assert scores
    for node in scores:
        assert analysis.cost.values[node.uid].flops == 2 * heads * pairs * head_dim
    for node in (n for n in vision if n.op == "swa_softmax_key0_bias"):
        assert analysis.cost.values[node.uid].flops == (
            5 * heads * pairs + heads * (reach + 1))
    assert not [n for n in vision if n.op == "softmax_with_sink"]
    ropes = [n for n in vision if n.op == "rope_2d"]
    assert {n.name for n in ropes} == {"attn.q_rope", "attn.k_rope"}
    assert {program.blocks[program.block_of(n.uid)].meta.template_name
            for n in ropes} >= {"mimo_v2_5_vision_block_full",
                                "mimo_v2_5_vision_block_sink"}


# -- B1: attention_value_scale ----------------------------------------------

#: The corpus snapshots whose config carries `attention_value_scale`.
VALUE_SCALE_MODELS = ("XiaomiMiMo__MiMo-V2-Flash", "XiaomiMiMo__MiMo-V2.5",
                      "XiaomiMiMo__MiMo-V2.5-Pro", "XiaomiMiMo__MiMo-V2.6-Pro-RL")

def test_a_mimo_config_with_a_value_scale_scales_v_in_every_attention():
    """vLLM mimo_v2.py: `v = v * self.v_scale` before the attention that
    writes the cache, on the full and the sliding branch (:344-346, 368-397)
    and in MTP (mimo_v2_mtp.py:94)."""
    from backend.arch.ops import OP_REGISTRY
    checked = 0
    for stem in VALUE_SCALE_MODELS:
        config = json.loads((SNAPSHOTS / f"{stem}.json").read_text())["config"]
        assert config.get("attention_value_scale") is not None, stem
        program = _program(stem)
        applies = [n for n in program.nodes()
                   if not n.children and n.uid not in program.encoder_uids
                   and OP_REGISTRY.get(n.op) is not None
                   and OP_REGISTRY[n.op].attention_kernel == "apply"]
        assert applies, stem
        for node in applies:
            v = node.inputs_obj[1].src
            assert v is not None and v.op == "scale_const", (
                f"{stem}: {node.display_path} reads {v and v.op}")
        checked += 1
    assert checked == 4
