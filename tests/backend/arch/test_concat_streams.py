"""Whole-stream joins: a reader that takes an appended stream's k elements
joined along an axis (`read: concat`, or a model-level `op:` with `axis:`).

Llava's `get_image_features` (`transformers/models/llava/modeling_llava.py
:155-172`) reads `hidden_states[vision_feature_layer]`: an int is one hidden
state, read as it is; a list is those hidden states in the order it writes
them — each cropped `hs[:, 1:]` under the default strategy — joined by
`torch.cat(..., dim=-1)` before the projector. EAGLE-3 (vLLM
`interfaces.py:1493-1509, 1580-1601`) keeps the boundary values of layers
`(2, n // 2, n - 3)` and its head's `fc` reads them joined, `k·D -> D`.

The join is not written in the spec — k is known only at resolve — so the
program lowers it to real `slice` / `concat` nodes before the walk
(`Program.lowered`), and they are priced, laid out and drawn like any node.
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from backend.arch.eval import traffic as traffic_mod
from backend.arch.eval.dim import value_of
from backend.arch.eval.program import OpStage, Program
from backend.arch.eval.transforms import Deployment, Workload
from backend.arch.layout.parallel_config import ParallelConfig
from backend.arch.loader import SpecLoadError, SpecLoader
from backend.arch.schema.model import StreamSpec

D = 8
B, S = 1, 5

LLAVA = """\
match: {{ model_type: [{model_type}], model_id: [{model_type}] }}
dims:
  d_model: config.hidden_size
  n_layers: "2"
  n_vis_layers: "6"
blocks:
  blk:
    graph:
      n: {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      o: {{ op: add, in: [x, n], dim: d_model }}
    outputs: [o]
  vblk:
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
    checkpoint: {{ prefix: vis.blocks, index: local }}
    layers:
      - {{ name: vis.patchify, op: patchify, inputs: [pixel_values],
          dims: {{ patch: "1", temporal: "1", channels: d_model, align: "2" }},
          segment: item, source_anchor: "a test processor" }}
      - {{ name: vis.patch, op: linear, inputs: [vis.patchify], dims: {{ in: d_model, out: d_model }},
          weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      - name: vis.blocks
        template: vblk
        repeat: n_vis_layers
        inputs: [vis.patch]
        streams:
          hidden: {{ carries: true }}
          feature: {{ from: hidden, after: {after}, index: hidden_states, {extra}src: "hs_pool = [hidden_states[i] for i in layers]" }}
      - {{ name: projector, op: linear, inputs: [feature], {axis}
          dims: {{ in: "{k} * d_model", out: d_model }},
          weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - {{ name: merge, op: scatter_rows_replace, inputs: [embed, projector],
      prompt_slots: {{ domain: pixel_values, defines: true, source_anchor: "a test merge" }} }}
  - name: layers
    repeat: n_layers
    inputs: [merge]
    streams:
      hidden: {{ carries: true }}
    template: blk
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: d_model }}, inputs: [layers],
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""

CROP = "slice: { axis: 1, start: 1 }, "

EAGLE = """\
match: {{ model_type: [{model_type}], model_id: [{model_type}] }}
dims:
  d_model: config.hidden_size
  n_layers: "7"
blocks:
  blk:
    graph:
      n: {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      o: {{ op: add, in: [x, n], dim: d_model }}
    outputs: [o]
  blk_wide:
    graph:
      n: {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      g: {{ op: mul, in: [x, n], dim: d_model }}
      o: {{ op: add, in: [x, g], dim: d_model }}
    outputs: [o]
  fc:
    inputs: [h]
    graph:
      proj: {{ op: linear, in: h, dims: {{ in: "3 * d_model", out: d_model }},
               weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
    outputs: [proj]
  eagle_head:
    reads: [aux]
    graph:
      fc: {{ op: {fc}, in: aux{fc_dims} }}
      o: {{ op: add, in: [x, fc], dim: d_model }}{second}
    outputs: [o]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: d_model, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - name: layers
    repeat: n_layers
    inputs: [embed]
    streams:
      hidden: {{ carries: true }}
      aux: {{ from: hidden, after: {after}, index: hidden_states, src: "aux_hidden_states.append(hidden_states + residual)" }}
    per_layer:
      assign:
        - {{ type: wide, layers: ["3"] }}
        - {{ type: plain, rest: true }}
      map:
        wide: {{ template: blk_wide }}
        plain: {{ template: blk }}
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: d_model }}, inputs: [layers],
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - name: spec
    group: true
    aux_kind: speculative
    inputs: [layers]
    checkpoint: {{ prefix: spec.layers, index: local }}
    layers:
      - name: spec.layers
        template: eagle_head
        repeat: "1"
        inputs: [layers]
        streams:
          hidden: {{ carries: true }}
          aux: {{ read: concat, axis: -1 }}
"""

LINEAR_FC = dict(fc="linear",
                 fc_dims=(', dims: { in: "3 * d_model", out: d_model }, '
                          'weight_axes: [{ replicated: true }, '
                          '{ replicated: true }]'),
                 second="")


def _load(text: str, model_type: str, config=None):
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / f"{model_type}.yaml").write_text(text)
        loader = SpecLoader(base_dir=str(base))
        return loader.match({"model_type": model_type, "torch_dtype": "bfloat16", "hidden_size": D,
                             **(config or {})}, model_id=model_type)


def _llava(model_type, *, after="[-2]", k=1, crop=True, order=False,
           axis=True, config=None):
    extra = (CROP if crop else "") + ("order: declared, " if order else "")
    return _load(LLAVA.format(model_type=model_type, after=after, k=k,
                              extra=extra,
                              axis="axis: -1," if axis else ""),
                 model_type, config)


def _eagle(model_type, *, after='[2, "n_layers // 2", "n_layers - 3"]',
           **body):
    fields = {**LINEAR_FC, **body}
    return _load(EAGLE.format(model_type=model_type, after=after, **fields),
                 model_type)


def _deployment(resolved, *, media=None, **parallel):
    deployment = Deployment(Program.of(resolved)).at(
        batch=B, input_tokens=S, media=media or {})
    if parallel:
        deployment = replace(deployment, config=ParallelConfig(**parallel))
    return deployment


def _analyze(resolved, **parallel):
    return _deployment(resolved, **parallel).analyze()


#: One 2x2 image: with `patch: "1"` the tower holds its four pixels as rows.
IMAGE = {"pixel_values": {"n_img": 1, "img_h": 2, "img_w": 2}}
IMAGE_ROWS = 4


def _encode(resolved, **parallel):
    """Llava's vision group — tower, joins, projector — in its encode walk
    on `IMAGE`."""
    return _deployment(resolved, media=IMAGE, **parallel).during(
        "encode", stack="vis").analyze()


def _lowered(analysis):
    return [n for n in analysis.program.nodes() if n.provenance == "lowering"]


def _extents(shape):
    return [int(value_of(x)) for x in shape]


def _traffic(analysis):
    return traffic_mod.derive(analysis).values


def _by_name(analysis, name):
    (node,) = [n for n in analysis.program.nodes() if n.name == name]
    return node


#: What a refused spec raises: the schema's error, or the loader's report of
#: a file that failed to load with it.
REFUSED = (ValueError, SpecLoadError)


def _bpe():
    # The default activation width, as `traffic.derive` prices it.
    return 2


class TestDeclaration:
    """What a join and the streams it reads may say."""

    @pytest.mark.parametrize("fields,match", [
        ({"read": "concat"}, "`axis`"),
        ({"read": "each", "axis": -1}, "`axis`"),
        ({"read": "concat", "axis": -1, "src": "a"}, "do not belong"),
        ({"read": "concat", "axis": -1, "order": "declared"}, "`from:`"),
        ({"from": "hidden", "after": "every", "index": "block",
          "order": "declared", "src": "a"}, "explicit list"),
        ({"from": "hidden", "before": [0, 2], "read": "all",
          "order": "declared", "src": "a"}, "`read: all`"),
        ({"from": "hidden", "before": [0, 2], "read": "all",
          "slice": {"axis": 1, "start": 1}, "src": "a"}, "`read: all`"),
        ({"from": "hidden", "after": [1], "index": "block", "src": "a",
          "slice": {"axis": 1, "start": 0}}, "whole value"),
    ])
    def test_refused_at_schema(self, fields, match):
        with pytest.raises(ValueError, match=match):
            StreamSpec(**fields)

    def test_declared_order_keeps_the_list_and_its_repeats(self):
        # HF indexes `hidden_states` in the list's order and may name one
        # twice; the elements follow it.
        spec = StreamSpec(**{"from": "hidden", "after": [-2, -5, 3, -2],
                             "index": "hidden_states", "order": "declared",
                             "src": "a"})
        assert spec.iterations(6, {}, int) == (4, 1, 2, 4)
        plain = StreamSpec(**{"from": "hidden", "after": [-2, -5, 3, -2],
                              "index": "hidden_states", "src": "a"})
        with pytest.raises(ValueError, match="order: declared"):
            plain.iterations(6, {}, int)

    def test_a_list_entry_may_be_a_dim_expression_and_floor_divides(self):
        # EAGLE-3's default when the config names no layers:
        # (2, n // 2, n - 3), with `//` the floor it is in Python.
        spec = StreamSpec(**{"from": "hidden",
                             "after": [2, "n_layers // 2", "n_layers - 3"],
                             "index": "hidden_states", "src": "a"})
        dims = {"n_layers": 7}
        from backend.arch.eval.dim import eval_dim_expr
        assert spec.iterations(7, {}, lambda e: eval_dim_expr(e, dims)) == (
            1, 2, 3)

    @pytest.mark.parametrize("order", [None, "declared"])
    def test_a_config_integer_is_a_list_of_one(self, order):
        # Llava's `vision_feature_layer` is an int as often as a list; the
        # int names one hidden state (HF's direct read).
        spec = StreamSpec(**{"from": "hidden",
                             "after": "config.vision_feature_layer",
                             "index": "hidden_states", "src": "a",
                             **({"order": order} if order else {})})
        assert spec.iterations(6, {"vision_feature_layer": -2}, int) == (4,)
        with pytest.raises(ValueError, match="not a list of integers"):
            spec.iterations(6, {"vision_feature_layer": True}, int)

    def test_a_config_integer_reads_one_element_directly(self):
        resolved = _llava("cc_int", after="config.vision_feature_layer",
                          config={"vision_feature_layer": -2})
        analysis = _encode(resolved)
        assert [n.op for n in _lowered(analysis)] == ["slice"]

    def test_an_unset_join_is_not_in_the_dump(self):
        # A stream that declares none of the new fields dumps as before, so
        # no spec's fingerprint moves.
        spec = StreamSpec(**{"from": "hidden", "after": [1],
                             "index": "block", "src": "a"})
        assert set(spec.model_dump(exclude_unset=False)) >= {"src"}
        assert not {"axis", "order", "slice"} & set(spec.model_dump())


class TestDirectRead:
    """k = 1 is HF's int branch: the hidden state is read as it is."""

    def test_one_cropped_element_is_a_crop_and_no_concat(self):
        analysis = _encode(_llava("cc_one_crop"))
        lowered = _lowered(analysis)
        assert [n.op for n in lowered] == ["slice"]
        (crop,) = lowered
        assert _extents(analysis.shapes.values[crop.uid].out) == [B, IMAGE_ROWS - 1, D]
        projector = _by_name(analysis, "projector")
        assert [_extents(s) for s in
                analysis.shapes.values[projector.uid].ins] == [[B, IMAGE_ROWS - 1, D]]
        traffic = _traffic(analysis)
        # The crop is a view: nothing moves; the projector reads IMAGE_ROWS - 1 rows.
        assert (traffic[crop.uid].act_reads, traffic[crop.uid].act_writes) \
            == (0, 0)
        assert traffic[projector.uid].act_reads == (IMAGE_ROWS - 1) * D * _bpe()

    def test_one_element_uncropped_adds_no_node(self):
        resolved = _llava("cc_one_plain", crop=False)
        program = Program.of(resolved)
        assert program.lowered() is program
        analysis = _encode(resolved)
        assert _lowered(analysis) == []
        projector = _by_name(analysis, "projector")
        assert [_extents(s) for s in
                analysis.shapes.values[projector.uid].ins] == [[B, IMAGE_ROWS, D]]


class TestJoin:
    """k > 1: one real `concat` of the k elements, each cropped first."""

    AFTER = "[-2, -5, 3, -2]"

    def _analysis(self, model_type, **kw):
        return _encode(_llava(model_type, after=self.AFTER, k=4, order=True,
                               **kw))

    def test_the_join_is_a_node_in_the_declared_order(self):
        resolved = _llava("cc_order", after=self.AFTER, k=4, order=True)
        (row,) = [r for r in Program.of(resolved).whole_stream_pairings
                  if r.stream == "feature"]
        # hidden_states[-2, -5, 3, -2] of a 6-deep tower: after iterations
        # 4, 1, 2 and 4 again, in that order.
        assert (row.representation, row.axis, row.visible, row.sources,
                row.reader, row.reader_iterations) == (
            "concat", -1, 4, (4, 1, 2, 4), "projector", ())
        analysis = _encode(resolved)
        lowered = _lowered(analysis)
        assert [n.op for n in lowered] == ["slice"] * 4 + ["concat"]
        *crops, join = lowered
        # Each crop reads its element by position; the join reads the crops.
        assert [ref.external_sid for c in crops for ref in c.inputs_obj] == [
            "feature[0]", "feature[1]", "feature[2]", "feature[3]"]
        assert [ref.src.uid for ref in join.inputs_obj] == [
            c.uid for c in crops]
        projector = _by_name(analysis, "projector")
        (ref,) = [r for r in projector.inputs_obj
                  if r.external_sid == "feature"]
        assert ref.src is join
        # The table names the join once it is a node.
        (lowered_row,) = analysis.program.whole_stream_pairings
        assert lowered_row == replace(row, through_uid=join.uid)
        # The op's stage runs the join first.
        (stage,) = [st for stack in analysis.program.stacks
                    for st in stack.stages if st.name == "projector"]
        assert isinstance(stage, OpStage)
        assert [n.uid for n in stage.roots] == [
            *(n.uid for n in lowered), projector.uid]

    def test_the_crop_comes_before_the_join_in_bytes(self):
        analysis = self._analysis("cc_bytes")
        *crops, join = _lowered(analysis)
        shapes = analysis.shapes.values
        assert [_extents(s) for s in shapes[join.uid].ins] == \
            [[B, IMAGE_ROWS - 1, D]] * 4
        assert _extents(shapes[join.uid].out) == [B, IMAGE_ROWS - 1, 4 * D]
        traffic = _traffic(analysis)
        row = (IMAGE_ROWS - 1) * D * _bpe()
        # k reads of the cropped rows, one write of their sum — the same
        # element named twice is read twice, as `torch.cat` reads it.
        assert (traffic[join.uid].act_reads, traffic[join.uid].act_writes) \
            == (4 * row, 4 * row)
        assert all(traffic[c.uid].act_reads == traffic[c.uid].act_writes == 0
                   for c in crops)
        projector = _by_name(analysis, "projector")
        assert traffic[projector.uid].act_reads == 4 * row

    def test_the_step_graph_draws_no_encoder_node(self):
        """The prefill graph is the language model's step: the tower, its
        joins and the projector are the encode walk's, and the merge reads
        the projector's rows as a handoff — no edge into the encoder is
        drawn, and the graph renders (every edge has a shape)."""
        from backend.inference.framework_bindings import get_binding
        from backend.inference.pipeline import analyze
        from backend.rendering._compute_graph import serialize_post_tree
        from backend.rendering.kv_marks import KVMarks

        resolved = _llava("cc_render", after=self.AFTER, k=4, order=True)
        res = analyze(resolved, parallel=ParallelConfig(),
                      framework_binding=get_binding("vllm"),
                      batch_size=B, avg_input_tokens=S, avg_output_tokens=0, phase="prefill", breakdown=False)
        graph = serialize_post_tree(
            res.post_tree, resolved, pre_tree=res.comm_tree,
            fusion_matches=res.fusion_matches, program=res.program,
            shapes=res.shapes, full_shapes=res.full_shapes,
            pre_program=res.pre_program, pre_shapes=res.pre_shapes,
            pre_full_shapes=res.pre_full_shapes,
            kv=KVMarks.of_serving(res))
        ops = [n["op"] for n in graph["nodes"]]
        assert "concat" not in ops and "patchify" not in ops
        assert not [n for n in graph["nodes"] if n.get("aux_kind") == "vision"]
        by_id = {n["id"]: n for n in graph["nodes"]}
        assert "pixel_values" not in by_id
        # The projector's rows arrive at the boundary, drawn as an input the
        # merge reads, not as the encoder's node.
        assert by_id["projector"]["op"] == "input"
        assert by_id["projector"]["inputs"] == []
        assert "projector" in by_id["merge"]["inputs"]


class TestMismatch:
    """`concat` joins tensors that agree on every axis but the joined one:
    the joined axis sums, whatever each element holds on it."""

    SPEC = """\
match: {{ model_type: [{model_type}], model_id: [{model_type}] }}
dims:
  d_model: config.hidden_size
blocks:
  full:
    graph:
      o: {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
    outputs: [o]
  other:
{other}
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: d_model, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - name: taps
    repeat: "2"
    inputs: [embed]
    streams:
      tap: {{ from: out, after: every, index: block, src: "taps.append(t)" }}
    per_layer:
      assign:
        - {{ type: f, layers: ["0"] }}
        - {{ type: r, rest: true }}
      map:
        f: {{ template: full }}
        r: {{ template: other }}
  - {{ name: joined, op: linear, inputs: [tap], axis: {axis},
      dims: {{ in: "{width} * d_model", out: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: d_model }}, inputs: [joined],
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""

    WIDE = """\
    graph:
      o: { op: linear, in: x, dims: { in: d_model, out: "2 * d_model" },
           weight_axes: [{ replicated: true }, { replicated: true }] }
    outputs: [o]"""

    ROWS = """\
    graph:
      o: { op: slice, in: x, axis: 1, dims: { out: "s_ctx - 1" } }
    outputs: [o]"""

    def test_elements_of_different_width_join_to_their_sum(self):
        analysis = _analyze(_load(self.SPEC.format(
            model_type="cc_wide", other=self.WIDE, axis=-1, width=3),
            "cc_wide"))
        (join,) = _lowered(analysis)
        shapes = analysis.shapes.values
        assert [_extents(s) for s in shapes[join.uid].ins] == [
            [B, S, D], [B, S, 2 * D]]
        assert _extents(shapes[join.uid].out) == [B, S, 3 * D]

    def test_elements_that_differ_on_another_axis_are_refused(self):
        resolved = _load(self.SPEC.format(model_type="cc_rows",
                                          other=self.ROWS, axis=-1, width=1),
                         "cc_rows")
        with pytest.raises(ValueError, match="every other axis equal"):
            _analyze(resolved)

    @pytest.mark.parametrize("axis", [1, -2])
    def test_joined_along_a_middle_axis_of_different_lengths(self, axis):
        # S rows and S - 1 rows joined along the sequence: 2S - 1 rows, D
        # shared. Each operand's joined axis is its own label, so the two
        # lengths are not a contradiction; the shared axes still must agree.
        analysis = _analyze(_load(self.SPEC.format(
            model_type=f"cc_mid{axis}", other=self.ROWS, axis=axis, width=1),
            f"cc_mid{axis}"))
        (join,) = _lowered(analysis)
        shapes = analysis.shapes.values
        assert [_extents(s) for s in shapes[join.uid].ins] == [
            [B, S, D], [B, S - 1, D]]
        assert _extents(shapes[join.uid].out) == [B, 2 * S - 1, D]
        traffic = _traffic(analysis)
        assert (traffic[join.uid].act_reads, traffic[join.uid].act_writes) \
            == ((2 * S - 1) * D * _bpe(), (2 * S - 1) * D * _bpe())
        joined = _by_name(analysis, "joined")
        assert _extents(shapes[joined.uid].out) == [B, 2 * S - 1, D]


class TestRefusals:

    def test_a_model_level_join_of_nothing_is_refused(self):
        # The projector always runs; with no hidden state named there is
        # nothing to join.
        with pytest.raises(REFUSED, match="nothing to join"):
            _llava("cc_empty", after='"config.feature_layers ?? absent"')

    def test_each_and_history_readers_of_a_declared_order_are_refused(self):
        with pytest.raises(REFUSED, match="only a concat"):
            _llava("cc_history", after="[-2, -5]", k=2, order=True,
                   axis=False)

    def test_a_crop_is_read_only_joined(self):
        with pytest.raises(REFUSED, match="only a concat"):
            _llava("cc_crop_history", axis=False)

    def test_an_op_that_takes_an_axis_cannot_also_join(self):
        text = LLAVA.format(model_type="cc_axis_op", after="[-2]", k=1,
                            extra="", axis="axis: -1,").replace(
            "name: projector, op: linear", "name: projector, op: slice")
        with pytest.raises(REFUSED, match="takes an `axis:` of its own"):
            _load(text, "cc_axis_op")

    def test_read_each_of_a_declared_order_is_refused(self):
        text = EAGLE.format(model_type="cc_each_order",
                            after="[4, 2]", **LINEAR_FC).replace(
            'index: hidden_states, src:',
            'index: hidden_states, order: declared, src:').replace(
            "aux: { read: concat, axis: -1 }", "aux: { read: each }")
        with pytest.raises(REFUSED, match="only a concat"):
            _load(text, "cc_each_order")


class TestEagle:
    """EAGLE-3: the head joins the backbone's boundary values of layers
    (2, n // 2, n - 3) — the input of each, `hidden_states[i]`."""

    def test_the_head_joins_the_default_layers(self):
        resolved = _eagle("cc_eagle")
        (row,) = Program.of(resolved).whole_stream_pairings
        # n = 7: layers 2, 3, 4, whose inputs are the exits of 1, 2, 3.
        assert (row.stream, row.representation, row.axis, row.sources,
                row.reader_stack) == ("aux", "concat", -1, (1, 2, 3), "spec")
        analysis = _analyze(resolved)
        (join,) = _lowered(analysis)
        program = analysis.program
        assert join.op == "concat"
        assert program.blocks[program.block_of(join.uid)].meta.template_name \
            == "eagle_head"
        shapes = analysis.shapes.values
        assert [_extents(s) for s in shapes[join.uid].ins] == [[B, S, D]] * 3
        assert _extents(shapes[join.uid].out) == [B, S, 3 * D]
        fc = _by_name(analysis, "fc")
        assert [_extents(s) for s in shapes[fc.uid].ins] == [[B, S, 3 * D]]
        traffic = _traffic(analysis)
        assert traffic[join.uid].act_reads == 3 * S * D * _bpe()

    def test_each_reading_node_joins_on_its_own(self):
        second = ('\n      fc2: { op: linear, in: aux, dims: { in: '
                  '"3 * d_model", out: d_model }, weight_axes: '
                  '[{ replicated: true }, { replicated: true }] }\n'
                  '      o2: { op: add, in: [o, fc2], dim: d_model }')
        text = EAGLE.format(model_type="cc_two", after=(
            '[2, "n_layers // 2", "n_layers - 3"]'), **{
                **LINEAR_FC, "second": second}).replace(
            "    outputs: [o]\nmodel:", "    outputs: [o2]\nmodel:")
        analysis = _analyze(_load(text, "cc_two"))
        joins = _lowered(analysis)
        assert [n.op for n in joins] == ["concat", "concat"]
        traffic = _traffic(analysis)
        assert [traffic[j.uid].act_reads for j in joins] == [
            3 * S * D * _bpe()] * 2
        rows = [r for r in analysis.program.whole_stream_pairings
                if r.stream == "aux"]
        assert sorted(r.reader_uid for r in rows) == sorted(
            _by_name(analysis, n).uid for n in ("fc", "fc2"))

    def test_a_component_call_joins_once_for_its_body(self):
        # A call that reads the stream binds its body to the join; the body
        # names the caller's tensor by the stream's name and reads the
        # same join, not one of its own.
        analysis = _analyze(_eagle(
            "cc_call", fc="fc", fc_dims="", second=""))
        joins = _lowered(analysis)
        assert [n.op for n in joins] == ["concat"]
        proj = _by_name(analysis, "proj")
        assert [_extents(s) for s in analysis.shapes.values[proj.uid].ins] \
            == [[B, S, 3 * D]]

    def test_a_join_with_no_reading_position_is_accepted(self):
        # The stream has no elements and nothing joins it: nothing to pair.
        text = EAGLE.format(model_type="cc_unread",
                            after='"config.aux_layers ?? absent"',
                            **LINEAR_FC).replace(
            "reads: [aux]", "reads: []").replace(
            "fc: { op: linear, in: aux", "fc: { op: linear, in: x").replace(
            '"3 * d_model"', "d_model")
        program = Program.of(_load(text, "cc_unread"))
        assert program.whole_stream_pairings == ()

    def test_a_join_of_nothing_with_a_reader_is_refused(self):
        resolved = _eagle("cc_eagle_empty",
                          after='"config.aux_layers ?? absent"')
        with pytest.raises(ValueError, match="no elements"):
            _ = Program.of(resolved).whole_stream_pairings


class TestPipeline:
    """A join is placed like any node (`layout.stages.StagePlacement`), and
    the elements it reads cross the cuts between their writers and it."""

    def test_the_projector_join_is_no_node_of_the_step(self):
        """Llava's projector joins two elements of the vision tower inside
        the vision group, which its encode walk runs; the language model's
        step reads the projector's output as a handed-over tensor, a free
        source like the token ids. Split over two stages, the step's only
        crossing is its hidden state, and the join is no node of it."""
        from backend.arch.layout.stages import StagePlacement
        resolved = _llava("cc_pp_llava", after="[-2, -5]", k=2, order=True)
        analysis = _deployment(resolved, media=IMAGE, pp=2).analyze()
        assert _lowered(analysis)
        assert not [n for n in _lowered(analysis)
                    if n.uid in analysis.shapes.values]
        placement = StagePlacement.of(analysis)
        assert {c.cut for c in placement.crossings} == {0}
        assert all(c.serves == {""} for c in placement.crossings)
        assert [n.op for n in _lowered(_encode(resolved))] == \
            ["slice", "slice", "concat"]

    def test_the_head_join_reads_its_elements_across_the_cut(self):
        """The EAGLE-3 head reads the stack's output, so it runs on the last
        stage; it joins `hidden_states[2, 3, 4]`, the exits of iterations 1,
        2 and 3 (7 layers: 0-3 | 4-6), so all three cross cut 0 for it. The
        exit of iteration 3 is also the hidden state the cut hands on: one
        hop serves both. The other two serve only the head's stack, and are
        priced only when the head is timed."""
        from backend.arch.layout.materializer import materialize_comm_events
        from backend.arch.layout.stages import StagePlacement
        analysis = _analyze(_eagle("cc_pp_eagle"), pp=2)
        placement = StagePlacement.of(analysis)
        elements = {value for value, _state in analysis.appended["aux"]}
        hops = [c for c in placement.crossings if "spec" in c.serves]
        assert sorted(c.source_iteration for c in hops) == [1, 2, 3]
        assert all(c.producer in elements and c.cut == 0 for c in hops)
        assert [c.serves for c in hops
                if c.source_iteration == 3] == [{"", "spec"}]

        def sends(scope):
            return [e for e in materialize_comm_events(
                analysis, timing_scope=scope) if e.op == "send_recv"]
        assert len(sends(frozenset({""}))) == 1
        assert len(sends(frozenset({"", "spec"}))) == 3


class TestWalk:
    """The walk prices joins only as nodes of its graph, and lays them out
    like any node."""

    def test_an_unlowered_join_is_refused_by_the_walk(self):
        from backend.arch.eval.evaluator import Evaluator
        from backend.arch.layout.propagation import LayoutPropagation
        program = Program.of(_eagle("cc_unlowered"))
        rule = LayoutPropagation.of(ParallelConfig(), "prefill")
        with pytest.raises(ValueError, match="not lowered"):
            Evaluator(program, rule=rule, workload=Workload())
        low = program.lowered()
        assert low.is_lowered and low.lowered() is low
        assert not program.is_lowered

    @pytest.mark.parametrize("build,walk", [
        (lambda: _llava("cc_tp_llava", after="[-2, -5]", k=2, order=True),
         _encode),
        (lambda: _eagle("cc_tp_eagle"), _analyze)])
    def test_a_join_under_tensor_parallelism(self, build, walk):
        resolved = build()
        single = walk(resolved)
        split = walk(resolved, tp_attention=2, tp_dense_ffn=2)
        assert _lowered(single)
        for one, two in zip(_lowered(single), _lowered(split), strict=True):
            assert _extents(single.shapes.values[one.uid].out) == \
                _extents(split.shapes.values[two.uid].out)
