"""An encoder stack takes one tensor-parallel decision, keyed on its query heads.

vLLM's ViTs ask `vision.is_vit_use_data_parallel(num_heads)`: when the TP
degree does not divide the tower's query heads, the whole tower runs data
parallel — attention, MLP and projector replicated — and otherwise it splits
as declared, KV heads fewer than the degree replicating as GQA does. The
layout decides each encoder stack once, before the walk, from the `n_heads`
its attention scores declare (`attention_plan.replicated_encoders`). A
`fit: exact` axis the degree does not divide still refuses.
"""
from __future__ import annotations

import pytest

# A tower whose MLP is walked before its attention (12 or 32 query heads,
# `n_vis_kv_heads` KV heads), and a second tower with only an MLP: two
# encoder stacks, decided apart.
_SPEC = """\
match: {{ model_type: [{name}], model_id: [{name}] }}
dims:
  d_model: config.hidden_size
  d_ffn: config.intermediate_size
  n_vis_heads: config.num_heads
  n_vis_kv_heads: config.num_kv_heads
  d_vis_head: "4"
  n_layers: "1"
blocks:
  tower:
    role: aux_block
    graph:
      up:    {{ op: linear, in: x, dims: {{ in: d_model, out: d_ffn }},
               weight_axes: [{{ world: dense_ffn, parts: d_ffn, fit: exact }}, {{ replicated: true }}] }}
      act:   {{ op: gelu, in: up, dim: d_ffn }}
      down:  {{ op: linear, in: act, dims: {{ in: d_ffn, out: d_model }},
               weight_axes: [{{ replicated: true }}, {{ world: dense_ffn, parts: d_ffn, fit: exact }}] }}
      wq:    {{ op: linear, in: down, dims: {{ in: d_model, out: "n_vis_heads * d_vis_head" }},
               weight_axes: [{{ world: attention, parts: n_vis_heads, fit: replica_cells }}, {{ replicated: true }}] }}
      wk:    {{ op: linear, in: down, dims: {{ in: d_model, out: "n_vis_kv_heads * d_vis_head" }},
               weight_axes: [{{ world: attention, parts: n_vis_kv_heads, fit: replica_cells }}, {{ replicated: true }}] }}
      wv:    {{ op: linear, in: down, dims: {{ in: d_model, out: "n_vis_kv_heads * d_vis_head" }},
               weight_axes: [{{ world: attention, parts: n_vis_kv_heads, fit: replica_cells }}, {{ replicated: true }}] }}
      q_v:   {{ op: view_split, in: wq, axis: -1, dims: {{ in: "n_vis_heads * d_vis_head", n_heads: n_vis_heads, d_head: d_vis_head }} }}
      q:     {{ op: transpose, in: q_v, axes: [0, 2, 1, 3] }}
      k_v:   {{ op: view_split, in: wk, axis: -1, dims: {{ in: "n_vis_kv_heads * d_vis_head", n_heads: n_vis_kv_heads, d_head: d_vis_head }} }}
      k:     {{ op: transpose, in: k_v, axes: [0, 2, 1, 3] }}
      v_v:   {{ op: view_split, in: wv, axis: -1, dims: {{ in: "n_vis_kv_heads * d_vis_head", n_heads: n_vis_kv_heads, d_head: d_vis_head }} }}
      v:     {{ op: transpose, in: v_v, axes: [0, 2, 1, 3] }}
      score: {{ op: attention_score, in: [q, k], dims: {{ is_causal: "0", n_heads: n_vis_heads, d_head: d_vis_head }} }}
      soft:  {{ op: softmax, in: score, dims: {{ is_causal: "0", n_heads: n_vis_heads }} }}
      apply: {{ op: attention_apply, in: [soft, v], dims: {{ is_causal: "0", n_heads: n_vis_heads, d_head: d_vis_head }} }}
      o_t:   {{ op: transpose, in: apply, axes: [0, 2, 1, 3] }}
      merge: {{ op: view_merge, in: o_t, dims: {{ n_heads: n_vis_heads, d_head: d_vis_head }} }}
      wo:    {{ op: linear, in: merge, dims: {{ in: "n_vis_heads * d_vis_head", out: d_model }},
               weight_axes: [{{ replicated: true }}, {{ world: attention, parts: n_vis_heads, fit: replica_cells }}] }}
    outputs: [wo]
  mlp_tower:
    role: aux_block
    graph:
      up:   {{ op: linear, in: x, dims: {{ in: d_model, out: d_ffn }},
              weight_axes: [{{ world: dense_ffn, parts: d_ffn, fit: exact }}, {{ replicated: true }}] }}
      act:  {{ op: gelu, in: up, dim: d_ffn }}
      down: {{ op: linear, in: act, dims: {{ in: d_ffn, out: d_model }},
              weight_axes: [{{ replicated: true }}, {{ world: dense_ffn, parts: d_ffn, fit: exact }}] }}
    outputs: [down]
  blk:
    graph:
      n: {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      o: {{ op: add, in: [x, n], dim: d_model }}
    outputs: [o]
model_inputs:
  input_ids: {{ kind: tokens }}
  pixel_values: {{ kind: image, axes: [n_img, img_h, img_w] }}
  crop_values: {{ kind: image, axes: [n_crop, crop_h, crop_w] }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: d_model, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - name: vis
    group: true
    aux_kind: vision
    layers:
      - {{ name: vis.patchify, op: patchify, inputs: [pixel_values],
          dims: {{ patch: "1", temporal: "1", channels: d_model, align: "1" }},
          segment: item, source_anchor: "a test processor" }}
      - {{ name: vis, template: tower, inputs: [vis.patchify] }}
  - name: crop
    group: true
    aux_kind: vision
    layers:
      - {{ name: crop.patchify, op: patchify, inputs: [crop_values],
          dims: {{ patch: "1", temporal: "1", channels: d_model, align: "1" }},
          segment: item, source_anchor: "a test processor" }}
      - {{ name: crop, template: mlp_tower, inputs: [crop.patchify] }}
  - {{ name: merge, op: scatter_rows_replace, inputs: [embed, vis],
      prompt_slots: {{ domain: pixel_values, defines: true, source_anchor: "a test merge" }} }}
  - {{ name: merge_crop, op: scatter_rows_replace, inputs: [merge, crop],
      prompt_slots: {{ domain: crop_values, defines: true, source_anchor: "a test merge" }} }}
  - {{ name: layers, repeat: n_layers, streams: {{ hidden: {{ carries: true }} }}, template: blk, inputs: [merge_crop] }}
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: d_model }}, inputs: [layers],
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""

_MEDIA = {"pixel_values": {"n_img": 1, "img_h": 4, "img_w": 4},
          "crop_values": {"n_crop": 1, "crop_h": 4, "crop_w": 4}}


def _program(name, heads, kv_heads):
    import tempfile
    from pathlib import Path
    from backend.arch.eval.program import Program
    from backend.arch.loader import SpecLoader
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / f"{name}.yaml").write_text(_SPEC.format(name=name))
        return Program.of(SpecLoader(base_dir=str(base)).match(
            {"model_type": name, "torch_dtype": "bfloat16", "hidden_size": 8,
             "intermediate_size": 16, "num_heads": heads,
             "num_kv_heads": kv_heads},
            model_id=name))


def _split(tp, stack, heads=12, kv_heads=12):
    """{node name: whether one rank holds a slice of its weight} for one
    encoder stack's encode walk at tensor-parallel degree `tp`."""
    from backend.arch.eval.transforms import Deployment
    from backend.arch.layout.persistent import (
        physical_artifacts, physical_weight_leaves)
    from backend.inference.parallel_config import ParallelConfig
    analysis = Deployment(
        _program(f"encoder_heads_{tp}_{stack}_{heads}_{kv_heads}",
                 heads, kv_heads),
        config=ParallelConfig(tp_attention=tp, tp_dense_ffn=tp)).at(
        batch=1, media=_MEDIA).during("encode", stack=stack).analyze()
    leaves = physical_weight_leaves(
        analysis, artifacts=physical_artifacts(analysis))
    return {analysis.program.by_uid(uid).name:
            leaf.runtime_shape != leaf.global_shape
            for uid, leaf in leaves.items()
            if analysis.program.stack_of(analysis.program.by_uid(uid)) == stack}


_TOWER = ("up", "down", "wq", "wk", "wv", "wo")


@pytest.mark.parametrize("tp", [2, 4])
def test_heads_the_degree_divides_keep_the_declared_split(tp):
    assert _split(tp, "vis") == dict.fromkeys(_TOWER, True)


def test_heads_the_degree_does_not_divide_replicate_the_whole_stack():
    """12 query heads over TP=8: the MLP, which 8 would divide and which the
    walk lays out before any attention weight, is whole too."""
    assert _split(8, "vis") == dict.fromkeys(_TOWER, False)


def test_kv_heads_below_the_degree_do_not_replicate_the_stack():
    """32 query heads, 8 KV heads, TP=16 (MiMo-V2.5's vision tower): the
    queries split, so the stack does; K/V split their 8 heads and replicate
    each head cell on two ranks, as GQA does."""
    assert _split(16, "vis", heads=32, kv_heads=8) == dict.fromkeys(
        _TOWER, True)


def test_each_encoder_stack_decides_on_its_own_heads():
    """The crop tower has no attention score to key on: its MLP splits at
    TP=8 while the vision tower beside it is replicated."""
    assert _split(8, "crop") == {"up": True, "down": True}


def test_divisible_heads_do_not_rescue_an_indivisible_exact_axis():
    """12 heads divide TP=6, so the stack splits; its 16-wide MLP does not,
    and `fit: exact` refuses as it always did."""
    from backend.arch.eval.refusal import SpecRefusal
    with pytest.raises(SpecRefusal, match="d_ffn"):
        _split(6, "vis")
