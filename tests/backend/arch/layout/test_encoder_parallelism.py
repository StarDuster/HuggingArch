"""A multimodal encoder is split only by tensor parallelism and pipeline stage.

vLLM runs a vision/audio encoder on its TP group (`mm_encoder_tp_mode`) and
every prefill-context-parallel rank runs the whole encoder; sglang runs it on
its TP group. Neither cuts the encoder's tokens by the text model's context
parallelism. An encoder runs in its own encode phase: under CP its walk
carries no CP cut anywhere, while the text backbone's prefill walk, which
reads the encoder's output as a handoff and walks no encoder node, keeps its
CP-cut sequence.
"""
from __future__ import annotations

import pytest

from tests.snapshots import resolve_snapshot, shared_walk


def _program(model_id):
    from tests.snapshots import shared_loader
    from backend.arch.eval.program import Program
    return Program.of(resolve_snapshot(model_id, shared_loader()))


def _analysis(model_id, **parallel):
    from tests.snapshots import shared_loader
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.pipeline import analyze
    return analyze(resolve_snapshot(model_id, shared_loader()),
                   parallel=ParallelConfig(**parallel), batch_size=1,
                   avg_input_tokens=1024, avg_output_tokens=0, phase="prefill", breakdown=False)


def _encode(model_id, stack, media, **parallel):
    from backend.arch.eval.transforms import Deployment
    from backend.inference.parallel_config import ParallelConfig
    return shared_walk(Deployment(_program(model_id),
                      config=ParallelConfig(**parallel)).at(
        batch=1, media=media).during("encode", stack=stack))


def _cut_by(state, world):
    return any(getattr(getattr(axis, "group", None), "id", None) == world
               for axis in state.layout)


# Each encoder of the two models, with one item of its media.
_ENCODERS = [
    ("XiaomiMiMo/MiMo-V2.5", "visual",
     {"pixel_values": {"n_img": 1, "img_h": 64, "img_w": 64}}),
    ("XiaomiMiMo/MiMo-V2.5", "audio",
     {"audio_codes": {"n_aud": 1, "s_audio": 64}}),
    ("stepfun-ai/Step-3.7-Flash", "vision", {"pixel_values": {"n_img": 1},
                                                "patch_pixel_values": {"n_crops": 0}}),
]


@pytest.mark.parametrize("model_id", ["XiaomiMiMo/MiMo-V2.5", "stepfun-ai/Step-3.7-Flash"])
def test_the_backbone_runs_context_parallel_without_the_encoder(model_id):
    res = _analysis(model_id, cp=2)
    encoder = res.program.encoder_uids
    assert encoder, "the model declares an encoder stack"
    states = res.analysis.states.values
    assert not [uid for uid in encoder if uid in states]
    backbone = [uid for uid, st in states.items() if _cut_by(st, "cp")]
    assert backbone, "the text backbone still runs context-parallel"


@pytest.mark.parametrize("model_id, stack, media", _ENCODERS)
def test_context_parallelism_does_not_reach_the_encoder(model_id, stack,
                                                        media):
    states = _encode(model_id, stack, media, cp=2).states.values
    assert states, "the encode walk evaluates the encoder"
    assert not [uid for uid, st in states.items() if _cut_by(st, "cp")]


def test_step_vision_heads_need_not_divide_the_context_degree():
    """Step-3.7's vision attention has 16 heads; at CP=32 the text model's
    Ulysses head repartition would need 32. The encoder is not in that world."""
    _analysis("stepfun-ai/Step-3.7-Flash", cp=32, ep_moe=32)
    model_id, stack, media = _ENCODERS[2]
    _encode(model_id, stack, media, cp=32, ep_moe=32)


# A tower whose last layer is a row-parallel projection: under the dense FFN
# world its output is a pending sum, and the model-level merge reads it by the
# tower's name — a ref with no node behind it in the merge's scope.
_TOWER_SPEC = """\
match: {{ model_type: [{name}], model_id: [{name}] }}
dims:
  d_model: config.hidden_size
  d_ffn: config.intermediate_size
  n_layers: "1"
blocks:
  tower:
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
  - {{ name: merge, op: scatter_rows_replace, inputs: [embed, vis],
      prompt_slots: {{ domain: pixel_values, defines: true, source_anchor: "a test merge" }} }}
  - {{ name: layers, repeat: n_layers, streams: {{ hidden: {{ carries: true }} }}, template: blk, inputs: [merge] }}
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: d_model }}, inputs: [layers],
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""


def _tower_model(name):
    import tempfile
    from pathlib import Path
    from backend.arch.loader import SpecLoader
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / f"{name}.yaml").write_text(
            _TOWER_SPEC.format(name=name))
        return SpecLoader(base_dir=str(base)).match(
            {"model_type": name, "torch_dtype": "bfloat16", "hidden_size": 8, "intermediate_size": 16},
            model_id=name)


@pytest.mark.parametrize("tp", [2, 4])
def test_a_tower_hands_its_partial_sum_to_the_merge_reduced(tp):
    """The tower runs in its own encode walk and the merge reads its output
    as a handoff. The reduction still happens, once, in the encode phase as
    the tower hands its output on: one all_reduce after the row-parallel
    projection, moving the whole [B, rows, d_model] partial over the dense
    FFN world. The language model's merge reads the handoff whole and
    reduced; its step carries none of the media's rows (the encoder is its
    own accounting)."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    from backend.arch.layout.materializer import materialize_comm_tree
    from backend.arch.layout.materialize_comm import Placement
    from backend.inference.parallel_config import ParallelConfig

    # One 4x4 image of 1x1 patches: 16 rows.
    rows = 16
    deployment = Deployment(
        Program.of(_tower_model(f"tower_handoff_{tp}")),
        config=ParallelConfig(tp_dense_ffn=tp)).at(
        batch=1, input_tokens=4,
        media={"pixel_values": {"n_img": 1, "img_h": 4, "img_w": 4}})
    encoded = shared_walk(deployment.during("encode", stack="vis"))
    down = next(n for n in encoded.program.nodes() if n.name == "down")

    tree = materialize_comm_tree(encoded)
    (event,) = [e for e in tree.events if e.attribution_uid == down.uid]
    assert (event.op, event.world, event.group_size) == \
        ("all_reduce", "tp_dense_ffn", tp)
    assert (event.placement, event.placement_uid, event.rewrites) == \
        (Placement.AFTER, down.uid, ())
    payload = 1 * rows * 8 * 2
    assert event.payload_bytes == payload
    assert event.bytes_per_rank == 2 * (tp - 1) / tp * payload

    analysis = shared_walk(deployment)
    merge = next(n for n in analysis.program.nodes() if n.name == "merge")
    # The merge reads the handoff: the reduced value, nothing pending.
    assert [int(e) for e in analysis.shapes.values[merge.uid].ins[1]] == \
        [1, 0, 8]
    assert analysis.input_states[(merge.uid, 1)].carries_partial() is False
