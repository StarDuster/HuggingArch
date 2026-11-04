"""A call (`call:`): an earlier layer — a template call beside it, or a
whole encoder group — run again on other inputs with its weights. Every
weight-holding leaf of the call reads the leaf at its place in the called
layer, so the model stores the weights once and each call computes and
reads them (Step-3.7's crops through the tower its global views ran, a
model's videos through its image tower)."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from backend.arch.eval import traffic as traffic_mod
from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.loader import SpecLoader

_SPEC = """\
match: {{ model_type: [_tied_tower], model_id: [_tied_tower] }}
dims: {{ d_model: config.hidden_size, n_layers: "1" }}
blocks:
  blk:
    graph:
      n: {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
    outputs: [n]
  tower:
    inputs: [x]
    graph:
      proj: {{ op: linear, in: x, dims: {{ in: d_model, out: d_model }},
              weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
    outputs: [proj]
  other_tower:
    inputs: [x]
    graph:
      proj: {{ op: linear, in: x, dims: {{ in: d_model, out: d_model }},
              weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
    outputs: [proj]
model_inputs:
  input_ids: {{ kind: tokens }}
  pixel_values: {{ kind: image, axes: [n_img, img_h, img_w] }}
  crops: {{ kind: image, axes: [n_crops, crop_h, crop_w], fixed: {{ crop_h: "2", crop_w: "2" }},
           per_item_of: pixel_values }}
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
      - {{ name: vis.tower, inputs: [vis.patchify], template: tower }}
      - {{ name: vis.patchify_crops, op: patchify, inputs: [crops],
          dims: {{ patch: "1", temporal: "1", channels: d_model, align: "1" }},
          segment: item, source_anchor: "a test processor" }}
      - {{ name: vis.crop_tower, call: {called}, inputs: [vis.patchify_crops] }}
  - {{ name: merge, op: scatter_rows_replace, inputs: [embed, vis.tower],
      prompt_slots: {{ domain: pixel_values, defines: true, source_anchor: "a test merge" }} }}
  - {{ name: layers, template: blk, repeat: n_layers, inputs: [merge],
      streams: {{ hidden: {{ carries: true }} }} }}
  - {{ name: head, op: linear, boundary_role: output_head, inputs: [layers],
      dims: {{ in: d_model, out: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""

D_MODEL = 8


def _resolved(called: str = "vis.tower"):
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / "_tied_tower.yaml").write_text(
            _SPEC.format(called=called))
        return SpecLoader(base_dir=str(base)).match(
            {"model_type": "_tied_tower", "hidden_size": D_MODEL, "torch_dtype": "bfloat16"},
            model_id="_tied_tower")


def _encode(crops: int):
    return Deployment(Program.of(_resolved())).at(
        batch=1, media={"pixel_values": {"n_img": 1, "img_h": 2, "img_w": 2},
                        "crops": {"n_crops": crops}}).during(
        "encode", stack="vis").analyze()


def test_the_weights_are_stored_once():
    analysis = _encode(3)
    stored = [row for row in analysis.tensors() if not row.is_metadata]
    assert len(stored) == 1
    assert value_of(analysis.total().params) == D_MODEL * D_MODEL


def test_each_call_computes_and_reads_the_weights():
    analysis = _encode(3)
    projs = [node for node in analysis.program.walked_nodes()
             if node.op == "linear"]
    first, second = projs
    assert second.tied_to_uid == first.uid
    # One image of 4 patch rows, three crops of 4 rows each.
    flops = {node.uid: value_of(analysis.cost.values[node.uid].flops)
             for node in projs}
    assert flops == {first.uid: 2 * 4 * D_MODEL ** 2,
                     second.uid: 2 * 12 * D_MODEL ** 2}
    leaves = traffic_mod.derive(analysis).leaves()
    assert value_of(leaves[second.uid].weight_reads) == \
        value_of(leaves[first.uid].weight_reads) > 0


def test_a_call_names_an_earlier_layer_beside_it():
    with pytest.raises(Exception, match="names no earlier layer beside it"):
        _resolved(called="merge")


_GROUPS = """\
match: { model_type: [_tied_group], model_id: [_tied_group] }
dims: { d_model: config.hidden_size, n_layers: "1", n_vis_layers: "2" }
blocks:
  blk:
    graph:
      n: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
    outputs: [n]
  vblk:
    graph:
      fc: { op: linear, in: x, dims: { in: d_model, out: d_model },
            weight_axes: [{ replicated: true }, { replicated: true }] }
    outputs: [fc]
model_inputs:
  input_ids: { kind: tokens }
  pixel_values: { kind: image, axes: [n_img, img_h, img_w] }
  other_pixels: { kind: image, axes: [n_other, other_h, other_w] }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: d_model, d_model: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
  - name: vis
    group: true
    aux_kind: vision
    checkpoint: { prefix: vis.blocks, index: local }
    layers:
      - { name: vis.patchify, op: patchify, inputs: [pixel_values],
          dims: { patch: "1", temporal: "1", channels: d_model, align: "1" },
          segment: item, source_anchor: "a test processor" }
      - { name: vis.proj, op: linear, inputs: [vis.patchify], dims: { in: d_model, out: d_model },
          weight_axes: [{ replicated: true }, { replicated: true }] }
      - { name: vis.blocks, template: vblk, repeat: n_vis_layers, inputs: [vis.proj],
          streams: { hidden: { carries: true } } }
  - { name: other, call: vis, inputs: [other_pixels] }
  - { name: merge, op: scatter_rows_replace, inputs: [embed, vis.blocks],
      prompt_slots: { domain: pixel_values, defines: true, source_anchor: "a test merge" } }
  - { name: merge_other, op: scatter_rows_replace, inputs: [merge, other.blocks],
      prompt_slots: { domain: other_pixels, defines: true, source_anchor: "a test merge" } }
  - { name: layers, template: blk, repeat: n_layers, inputs: [merge_other],
      streams: { hidden: { carries: true } } }
  - { name: head, op: linear, boundary_role: output_head, inputs: [layers],
      dims: { in: d_model, out: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""


def test_a_group_reruns_an_earlier_groups_weights():
    """A call of an encoder group (a model's videos through the tower its
    images run): its own encode step, every one of its weight-bearing
    leaves reading the called group's, nothing stored."""
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / "_tied_group.yaml").write_text(_GROUPS)
        resolved = SpecLoader(base_dir=str(base)).match(
            {"model_type": "_tied_group", "hidden_size": D_MODEL, "torch_dtype": "bfloat16"},
            model_id="_tied_group")
    program = Program.of(resolved)
    assert sorted(s.name for s in program.stacks if s.is_encoder) == [
        "other", "vis"]
    media = {"pixel_values": {"n_img": 1, "img_h": 2, "img_w": 2},
             "other_pixels": {"n_other": 2, "other_h": 2, "other_w": 2}}
    deployment = Deployment(program).at(batch=1, media=media)
    ckpt = deployment.checkpoint()
    stored = sorted(row.name for row in ckpt.tensors())
    # The language model's three and the first group's proj and two layers;
    # the second group stores nothing.
    assert len([n for n in stored if "other" in n]) == 0
    vis, other = (e.analyze() for e in deployment.encoders())
    # The second group walks beside the first, which holds the weights, at
    # no item of the first's: its own stack holds no parameter.
    own = other.by_stack()["other"]
    assert value_of(own.params) == 0
    assert value_of(other.total().flops) == value_of(own.flops) == \
        2 * value_of(vis.total().flops)
    readers = [n for n in other.program.walked_nodes()
               if n.op == "linear"
               and other.program.stack_of(n) == "other"]
    assert readers and all(n.tied_to_uid is not None for n in readers)
    assert deployment.media_tokens() == {"pixel_values": 4,
                                         "other_pixels": 8}


def _load(text: str, model_type: str):
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / f"{model_type}.yaml").write_text(text)
        return SpecLoader(base_dir=str(base)).match(
            {"model_type": model_type, "hidden_size": D_MODEL,
             "torch_dtype": "bfloat16"}, model_id=model_type)


def test_a_call_is_the_called_group_written_out_under_its_name():
    text = _GROUPS.replace(
        "          streams: { hidden: { carries: true } } }\n  - { name: other,",
        "          streams: { hidden: { carries: true },\n"
        "                     feats: { from: hidden, after: every, index: block,\n"
        "                              src: \"feats.append(hidden)\" } } }\n"
        "  - { name: other,")
    spec = _load(text, "_tied_group").spec
    other = next(layer for layer in spec.model if layer.name == "other")
    assert other.call == "vis" and other.group and other.checkpoint is None
    assert [(sub.name, sub.inputs) for sub in other.layers] == [
        ("other.patchify", ["other_pixels"]),
        ("other.proj", ["other.patchify"]),
        ("other.blocks", ["other.proj"])]
    assert sorted(other.layers[2].streams) == ["feats_other", "hidden"]


def test_a_call_runs_the_called_layer_as_it_is():
    text = _GROUPS.replace(
        "  - { name: other, call: vis, inputs: [other_pixels] }",
        "  - { name: other, call: vis, inputs: [other_pixels], "
        "checkpoint: { prefix: other.blocks, index: local } }")
    with pytest.raises(Exception, match="runs the called layer as it is"):
        _load(text, "_tied_group")


def test_a_call_gives_each_input_the_called_layer_reads():
    text = _GROUPS.replace(
        "inputs: [other_pixels] }", "inputs: [other_pixels, input_ids] }")
    with pytest.raises(Exception, match="reads \\['pixel_values'\\]"):
        _load(text, "_tied_group")


def test_a_call_rebinds_only_a_dim_the_called_layer_reads():
    text = _GROUPS.replace(
        "inputs: [other_pixels] }",
        "inputs: [other_pixels], dims: { n_layers: n_vis_layers } }")
    with pytest.raises(Exception, match="does not read"):
        _load(text, "_tied_group")


def test_a_call_runs_an_encoder_tower_not_any_layer():
    # A second call of the language model's stack.
    text = _GROUPS.replace(
        "  - { name: layers, template: blk, repeat: n_layers, inputs: [merge_other],\n"
        "      streams: { hidden: { carries: true } } }",
        "  - { name: layers, template: blk, repeat: n_layers, inputs: [merge_other],\n"
        "      streams: { hidden: { carries: true } } }\n"
        "  - { name: layers2, call: layers, inputs: [layers] }")
    with pytest.raises(Exception, match="a top-level encoder `group:`"):
        _load(text, "_tied_group")


def test_only_an_encoder_reruns_a_call():
    # An auxiliary group over the backbone's output, called again.
    text = _GROUPS.replace(
        "  - { name: head, op: linear,",
        "  - name: side\n"
        "    group: true\n"
        "    aux_kind: speculative\n"
        "    inputs: [layers]\n"
        "    layers:\n"
        "      - { name: side.proj, op: linear, inputs: [layers], "
        "dims: { in: d_model, out: d_model },\n"
        "          weight_axes: [{ replicated: true }, { replicated: true }] }\n"
        "  - { name: side2, call: side, inputs: [layers] }\n"
        "  - { name: head, op: linear,")
    with pytest.raises(Exception, match="only a modality encoder does"):
        _load(text, "_tied_group")


def test_a_rebinding_that_reaches_a_weight_is_refused():
    # A width of the call's own, 16, read through the 8-wide tower's weight.
    text = _GROUPS.replace(
        "dims: { d_model: config.hidden_size,",
        "dim_symbols:\n"
        "  d_other: { kind: whole, group: per_arch_one_offs, "
        "meaning: Width of a second call. }\n"
        "dims: { d_other: \"16\", d_model: config.hidden_size,").replace(
        "inputs: [other_pixels] }",
        "inputs: [other_pixels], dims: { d_model: d_other } }")
    with pytest.raises(Exception,
                       match=re.escape(
                           "reads another weight geometry than 'vis.proj'")):
        _load(text, "_tied_group")


def test_a_rebinding_that_reaches_a_count_is_refused():
    text = _GROUPS.replace(
        "inputs: [other_pixels] }",
        "inputs: [other_pixels], dims: { n_vis_layers: n_layers } }")
    with pytest.raises(Exception, match=re.escape("changes the count of 'vis.blocks'")):
        _load(text, "_tied_group")


def test_a_call_of_a_layer_that_calls_is_refused():
    # Step-3.7's shape — a tower called again beside it in its group — and
    # that whole group called once more.
    text = _SPEC.format(called="vis.tower").replace(
        "  - { name: merge, op: scatter_rows_replace,",
        "  - { name: vis2, call: vis, inputs: [pixel_values, crops] }\n"
        "  - { name: merge, op: scatter_rows_replace,")
    with pytest.raises(Exception, match=r"declares \['call'\]"):
        _load(text.replace("_tied_tower", "_nested"), "_nested")
