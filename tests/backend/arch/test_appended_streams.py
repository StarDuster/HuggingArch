"""Appended streams: the values a stack appends after chosen iterations, and
the stacks that read them element by element.

Qwen3-VL's vision tower runs 27 blocks and hands the hidden state after blocks
8, 16 and 24 to three mergers (`modeling_qwen3_vl_moe.py:746-750`: `if
layer_num in self.deepstack_visual_indexes: deepstack_merger_list[...]
(hidden_states)`); decoder layer j < 3 then adds merger j's output to the
visual rows of its own output (`:931-937`, `_deepstack_process` `:946-954`).
Without an appended stream the only name for the tower's hidden state is the
stack's, which is its LAST iteration, and every merger and every decoder
layer would read that.
"""
from __future__ import annotations

import re
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.layout.parallel_config import ParallelConfig
from backend.arch.loader import SpecLoader
from backend.arch.schema.model import StreamSpec
from backend.arch.spec_tree.projections import iter_leaves

SPEC = """\
match: {{ model_type: [{model_type}], model_id: [{model_type}] }}
dims:
  d_model: config.hidden_size
  n_layers: "{n_layers}"
  n_vis_layers: "4"
  n_deepstack: {n_deepstack}
blocks:
  blk:
    graph:
      n: {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      o: {{ op: add, in: [x, n], dim: d_model }}
    outputs: [o]
  tap:
    graph:
      proj: {{ op: linear, in: x, dims: {{ in: d_model, out: d_model }},
               weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
    outputs: [proj]
  inject:
    reads: [feat]
    graph:
      n: {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      o: {{ op: add, in: [x, n], dim: d_model }}
      ds: {{ op: scatter_rows_add, in: [o, feat] }}
    outputs: [ds]
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
          dims: {{ patch: "1", temporal: "1", channels: d_model, align: "1" }},
          segment: item, source_anchor: "a test processor" }}
      - {{ name: vis.patch, op: linear, inputs: [vis.patchify], dims: {{ in: d_model, out: d_model }},
          weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      - name: vis.blocks
        template: blk
        repeat: n_vis_layers
        inputs: [vis.patch]
        streams:
          hidden: {{ carries: true }}
          taps: {{ from: hidden, after: {after}, index: {index}, src: "if i in taps" }}
      - name: vis.mergers
        template: tap
        repeat: {reader_repeat}
        checkpoint: {{ prefix: vis.mergers, index: local }}
        inputs: [taps]
        streams:
          feat: {{ from: out, after: every, index: block, src: "feats.append(f)" }}
  - {{ name: merge, op: scatter_rows_replace, inputs: [embed, vis.blocks],
      prompt_slots: {{ domain: pixel_values, defines: true, source_anchor: "a test merge" }} }}
  - name: layers
    repeat: n_layers
    inputs: [merge]
    streams:
      hidden: {{ carries: true }}
      feat: {{ read: each, prompt_slots: {{ domain: pixel_values, defines: false, source_anchor: "a test deepstack add" }} }}
    per_layer:
      assign:
        - {{ type: ds, {ds} }}
        - {{ type: plain, rest: true }}
      map:
        ds: {{ template: inject }}
        plain: {{ template: blk }}
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: d_model }}, inputs: [layers],
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""

#: The decoder's read of the mergers' outputs: it adds onto the image
#: positions the merge defines.
FEAT_READ = next(line + "\n" for line in SPEC.splitlines()
                 if line.startswith("      feat: {{ read: each"))


def _resolved(model_type: str, *, after="[1, 3]", index="block",
              reader_repeat="n_deepstack", n_deepstack='"2"', n_layers=4,
              ds='range: ["0", "n_deepstack - 1"]', config=None,
              edit=lambda spec: spec):
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        spec = edit(SPEC)
        (base / "models" / f"{model_type}.yaml").write_text(spec.format(
            model_type=model_type, after=after, index=index,
            reader_repeat=reader_repeat, n_deepstack=n_deepstack,
            n_layers=n_layers, ds=ds))
        loader = SpecLoader(base_dir=str(base))
        return loader.match({"model_type": model_type, "torch_dtype": "bfloat16", "hidden_size": 8,
                             **(config or {})}, model_id=model_type)


def _no_dim(expr: str) -> int:
    """An `after:` list reads no dim; asking for one is a test failure."""
    raise AssertionError(f"no dim is read, but {expr!r} was asked for")


def _pairs(program: Program):
    return sorted((row.stream, row.element, row.writer, row.writer_iteration,
                   row.reader, row.reader_iteration)
                  for row in program.stream_pairings)


#: One 2x2 image: with `patch: "1"` its rows are its four pixels.
MEDIA = {"pixel_values": {"n_img": 1, "img_h": 2, "img_w": 2}}


def _deployment(program: Program, **parallel):
    deployment = Deployment(program).at(batch=1, input_tokens=4, media=MEDIA)
    if parallel:
        deployment = replace(deployment, config=ParallelConfig(**parallel))
    return deployment


def _analyze(program: Program, **parallel):
    """The language model's prefill step of a request with one image: the
    tower runs in its encode walk, and the step's handoffs carry none of
    its rows (the encoder is its own accounting)."""
    return _deployment(program, **parallel).analyze()


def _encode(program: Program):
    """The vision tower's encode walk on the same image."""
    return Deployment(program).at(batch=1, media=MEDIA).during(
        "encode", stack="vis").analyze()


class TestIterations:
    """What an index counts is declared, because the corpus uses both."""

    def test_block_counts_blocks(self):
        spec = StreamSpec(**{"from": "hidden", "after": [8, 16, 24, -1],
                             "index": "block", "src": "a"})
        assert spec.iterations(27, {}, _no_dim) == (8, 16, 24, 26)

    def test_hidden_states_counts_the_input_first(self):
        # HF `hidden_states[0]` is the stack's input and `[-2]` is after
        # block n-2 (Llava's `vision_feature_layer: -2`).
        spec = StreamSpec(**{"from": "hidden", "after": [0, 9, -2],
                             "index": "hidden_states", "src": "a"})
        assert spec.iterations(27, {}, _no_dim) == (-1, 8, 25)

    def test_every_is_each_iteration(self):
        spec = StreamSpec(**{"from": "out", "after": "every",
                             "index": "block", "src": "a"})
        assert spec.iterations(3, {}, _no_dim) == (0, 1, 2)

    @pytest.mark.parametrize("after,index", [
        ([27], "block"), ([28], "hidden_states"), ([-28], "block"),
        ([3, 3], "block"), ([5, 2], "block")])
    def test_what_names_no_iteration_is_refused(self, after, index):
        spec = StreamSpec(**{"from": "hidden", "after": after,
                             "index": index, "src": "a"})
        with pytest.raises(ValueError):
            spec.iterations(27, {}, _no_dim)

    def test_a_config_list_is_read_and_absence_is_declared(self):
        cfg = {"vision_config": {"deepstack_visual_indexes": [8, 16, 24]}}
        ref = "config.vision_config.deepstack_visual_indexes"

        def spec(after):
            return StreamSpec(**{"from": "hidden", "after": after,
                                 "index": "block", "src": "a"})

        assert spec(ref).iterations(27, cfg, _no_dim) == (8, 16, 24)
        assert spec(f"{ref} ?? absent").iterations(27, {}, _no_dim) == ()
        with pytest.raises(ValueError, match="does not set it"):
            spec(ref).iterations(27, {}, _no_dim)
        with pytest.raises(ValueError, match="not a mapping"):
            spec(f"{ref} ?? absent").iterations(27, {"vision_config": 3}, _no_dim)


class TestDeclaration:
    """What an appended stream and a reference to one may say."""

    @pytest.mark.parametrize("fields,match", [
        ({"read": "all"}, "on the appending declaration"),
        ({"from": "hidden", "after": {"every": "Kar"}, "index": "block",
          "src": "a"}, "not accepted yet"),
        ({"from": "hidden", "after": [1], "src": "a"}, "together"),
        ({"after": [1], "index": "block", "src": "a"}, "together"),
        ({"from": "hidden", "after": [1], "index": "block"}, "`src`"),
        ({"from": "out", "after": "every", "index": "hidden_states",
          "src": "a"}, "`block`"),
        ({"read": "each", "src": "a"}, "do not belong"),
        ({"from": "hidden", "after": [1], "index": "block", "src": "a",
          "shape": ["B"]}, "do not belong"),
    ])
    def test_refused_at_schema(self, fields, match):
        with pytest.raises(ValueError, match=match):
            StreamSpec(**fields)

    def test_read_all_is_refused_on_a_reading_stack(self):
        # `read: all` belongs to the appending declaration; a stack that
        # reads another stack's stream reads it element by element.
        with pytest.raises(Exception, match="on the appending declaration"):
            _resolved("as_read_all", edit=lambda spec: spec.replace(
                "feat: {{ read: each,", "feat: {{ read: all,", 1))

    def test_a_carrying_stack_appends_its_carried_stream(self):
        with pytest.raises(Exception, match="what it appends"):
            _resolved("as_from_out_chain", edit=lambda spec: spec.replace(
                "taps: {{ from: hidden,", "taps: {{ from: out,", 1))

    def test_a_parallel_stack_appends_its_output(self):
        with pytest.raises(Exception, match="`from: out`"):
            _resolved("as_from_hidden_par", edit=lambda spec: spec.replace(
                "feat: {{ from: out,", "feat: {{ from: hidden,", 1))


class TestBoundaryRead:
    """A stack that names the stream in `inputs:` reads element j at its
    iteration j."""

    def test_each_reader_iteration_reads_the_appended_hidden_state(self):
        program = Program.of(_resolved("as_ok"))
        analysis = _encode(program)
        proj = next(n for n in program.nodes() if n.name == "proj")
        block_out = next(n for n in program.nodes()
                         if n.name == "o" and program.stack_of(n) == "vis")
        assert (analysis.shapes.values[proj.uid].ins[0]
                == analysis.shapes.values[block_out.uid].out)
        # Two readers, one per element: the reader bank costs twice one.
        assert proj.multiplicity == 2
        assert [p for p in _pairs(program) if p[0] == "taps"] == [
            ("taps", 0, "vis.blocks", 1, "vis.mergers", 0),
            ("taps", 1, "vis.blocks", 3, "vis.mergers", 1)]

    def test_a_second_stack_in_the_group_names_its_own_tensors(self):
        names = {lt.tensor_name for lt in iter_leaves(_resolved("as_names"))}
        assert "vis.mergers.X.proj.weight" in names
        assert "vis.blocks.X.n.weight" in names

    def test_a_reader_that_does_not_pair_is_refused(self):
        with pytest.raises(Exception, match="has 2 element"):
            _resolved("as_unpaired", reader_repeat='"3"')

    def test_with_no_elements_the_reader_does_not_run(self):
        resolved = _resolved(
            "as_absent",
            after='"config.vision_config.deepstack_visual_indexes ?? absent"',
            n_deepstack='"len(config.vision_config.deepstack_visual_indexes) ?? 0"')
        assert resolved.appended_streams()["taps"].iterations == ()
        program = Program.of(resolved)
        assert not any(n.name in ("proj", "ds") for n in program.nodes())
        assert program.stream_pairings == ()
        _analyze(program)

    def test_hidden_states_zero_is_the_stack_input(self):
        resolved = _resolved("as_entry", after="[0]", index="hidden_states",
                             n_deepstack='"1"')
        assert resolved.appended_streams()["taps"].iterations == (-1,)
        _analyze(Program.of(resolved))

    def test_a_carrying_stack_does_not_read_at_its_boundary(self):
        with pytest.raises(Exception, match="carries no stream"):
            _resolved("as_chain_reader", edit=lambda spec: spec.replace(
                "        streams:\n          feat: {{ from: out,",
                "        streams:\n"
                "          hidden: {{ carries: true }}\n"
                "          feat: {{ from: hidden,", 1))

    def test_a_parallel_stack_is_not_read_by_name(self):
        with pytest.raises(Exception, match=re.escape("reads 'vis.mergers' by name")):
            _resolved("as_read_parallel", edit=lambda spec: spec.replace(
                "inputs: [embed, vis.blocks]", "inputs: [embed, vis.mergers]",
                1))

    def test_a_group_repeat_around_the_writer_is_refused(self):
        with pytest.raises(Exception, match="runs once"):
            _resolved("as_group_repeat", edit=lambda spec: spec.replace(
                "    aux_kind: vision\n",
                "    aux_kind: vision\n    repeat: \"2\"\n", 1))


class TestInnerRead:
    """A stack whose components read the stream: the j-th reading iteration
    reads element j."""

    def test_reader_layers_pair_in_order(self):
        program = Program.of(_resolved("as_inner"))
        _analyze(program)
        assert [p for p in _pairs(program) if p[0] == "feat"] == [
            ("feat", 0, "vis.mergers", 0, "layers", 0),
            ("feat", 1, "vis.mergers", 1, "layers", 1)]
        (node,) = {row.reader_port for row in program.stream_pairings
                   if row.stream == "feat"}
        assert node == "ds"

    def test_reader_count_must_equal_the_elements(self):
        with pytest.raises(Exception, match="in 3 iteration"):
            _analyze(Program.of(_resolved(
                "as_inner_count", ds='range: ["0", "n_deepstack"]')))

    def test_reader_layers_must_be_contiguous(self):
        with pytest.raises(Exception, match="not contiguous"):
            _analyze(Program.of(_resolved(
                "as_inner_gap", ds='layers: ["0", "2"]')))

    def test_with_no_elements_no_layer_reads(self):
        resolved = _resolved(
            "as_inner_absent",
            after='"config.vision_config.deepstack_visual_indexes ?? absent"',
            n_deepstack='"len(config.vision_config.deepstack_visual_indexes) ?? 0"')
        program = Program.of(resolved)
        assert not any(n.name == "ds" for n in program.nodes())
        _analyze(program)

    def test_a_component_reads_only_what_its_stack_declares(self):
        with pytest.raises(Exception, match="does not declare"):
            _analyze(Program.of(_resolved(
                "as_inner_undeclared", edit=lambda spec: spec.replace(
                    FEAT_READ, "", 1))))

    def test_the_reference_names_an_appended_stream(self):
        with pytest.raises(Exception, match="no earlier stack appends"):
            _resolved("as_inner_unknown", edit=lambda spec: spec.replace(
                FEAT_READ, FEAT_READ + "      other: {{ read: each }}\n", 1))


class TestNames:
    """One name, one thing, model-wide: an appended stream's `from:` is its
    producing declaration in the model's stream table."""

    def test_an_appended_stream_is_not_named_like_a_layer(self):
        with pytest.raises(Exception, match="share a name with a model layer"):
            _resolved("as_layer_clash", edit=lambda spec: spec.replace(
                "  - {{ name: merge,", "  - {{ name: taps, op: add, dim: d_model,"
                " inputs: [embed, vis.blocks], prompt_slots: {{ domain: "
                "pixel_values, defines: false, source_anchor: a }} }}\n"
                "  - {{ name: merge,", 1))

    def test_an_appended_stream_has_one_producing_declaration(self):
        with pytest.raises(Exception, match="and again by"):
            _resolved("as_twice", edit=lambda spec: spec.replace(
                "feat: {{ from: out,", "taps: {{ from: out,", 1))

    def test_an_appended_stream_is_not_an_inner_stream_too(self):
        with pytest.raises(Exception, match="both as an appended stream and "
                           "as an inner stream"):
            _resolved("as_inner_clash", edit=lambda spec: spec.replace(
                FEAT_READ, FEAT_READ + "      taps: {{ src: a }}\n", 1))

    def test_a_component_does_not_write_an_appended_stream(self):
        with pytest.raises(Exception, match="written by its stack"):
            _resolved("as_stream_out", edit=lambda spec: spec.replace(
                "    outputs: [proj]\n",
                "    outputs: [proj]\n    stream_out: {{ feat: proj }}\n", 1))


class TestPipeline:
    """The mergers' outputs reach the decoder as handed-over tensors
    (`Program.handoffs`): free sources, like a model input, so a reader on
    any pipeline stage reads its element where it runs and no cut sends one.
    The hidden state crosses every cut (`layout/test_stage_crossings.py`)."""

    @staticmethod
    def _sends(analysis):
        from backend.arch.layout.materializer import materialize_comm_events
        return [e for e in materialize_comm_events(analysis)
                if e.op == "send_recv"]

    @pytest.mark.parametrize("model_type,ds", [
        # Readers 0..1: both on the first of two stages (layers 0-1, 2-3).
        ("as_pp_ok", 'range: ["0", "n_deepstack - 1"]'),
        # Readers 1..2: element 1 is read on the second stage.
        ("as_pp_cross", 'range: ["1", "n_deepstack"]'),
    ])
    def test_only_the_hidden_state_crosses(self, model_type, ds):
        program = Program.of(_resolved(model_type, n_layers=4, ds=ds))
        deployment = _deployment(program)
        assert [(t.rows, t.width) for t in
                deployment.handed_over()["feat"]] == [(4, 8), (4, 8)]
        assert not self._sends(_analyze(program))
        analysis = _analyze(program, pp=2)
        (ds_node,) = [n for n in analysis.program.nodes() if n.name == "ds"]
        assert ds_node.multiplicity == 2
        # Each reader reads its element as the step holds it: [B, 0, width].
        assert [int(value_of(x)) for x in
                analysis.shapes.values[ds_node.uid].ins[1]] == [1, 0, 8]
        (hidden,) = self._sends(analysis)
        assert dict(hidden.meta)["cut"] == 0
        assert (hidden.world, hidden.group_size, hidden.count) == ("pp", 2, 1)
        # The hidden state of the step's 4 tokens.
        assert hidden.payload_bytes == 4 * 8 * 2


def test_qwen3_vl_pairs_six_elements():
    """Qwen3-VL's shape: 27 vision blocks append after 8, 16, 24; three
    mergers read one each and append their outputs; decoder layers 0..2 read
    those. Six pairings, in that order."""
    resolved = _resolved(
        "as_qwen3vl", n_layers=6,
        after="config.vision_config.deepstack_visual_indexes",
        n_deepstack='"len(config.vision_config.deepstack_visual_indexes)"',
        config={"vision_config": {"deepstack_visual_indexes": [8, 16, 24]}},
        edit=lambda spec: spec.replace('n_vis_layers: "4"',
                                       'n_vis_layers: "27"', 1))
    program = Program.of(resolved)
    _analyze(program)
    assert _pairs(program) == [
        ("feat", 0, "vis.mergers", 0, "layers", 0),
        ("feat", 1, "vis.mergers", 1, "layers", 1),
        ("feat", 2, "vis.mergers", 2, "layers", 2),
        ("taps", 0, "vis.blocks", 8, "vis.mergers", 0),
        ("taps", 1, "vis.blocks", 16, "vis.mergers", 1),
        ("taps", 2, "vis.blocks", 24, "vis.mergers", 2),
    ]


class TestParallelGroups:
    """The stack is the layer that repeats — a repeated group around blocks
    that do not repeat is one (DeepSeek's `mtp`). Carrying no stream, its
    name reads no one value; under a group whose block repeats, the
    innermost repeated layer is the stack."""

    @staticmethod
    def _spec(group_streams=None, child_repeat=None, child_streams=None,
              reads="g"):
        from backend.arch.schema import (ArchSpec, BlockGraph, GraphNode,
                                         ModelLayer, StreamSpec)
        hidden = {"hidden": StreamSpec(carries=True)}
        child = ModelLayer(
            name="g.layers", template="block", inputs=["norm"],
            repeat=child_repeat,
            streams=hidden if child_streams else {})
        return ArchSpec(
        model_inputs={"input_ids": {"kind": "tokens"}},
            name="t", dims={"d_model": "8", "n_vocab": "16", "n_layers": "2"},
            blocks={"block": BlockGraph(
                graph={"n": GraphNode(op="rmsnorm", inputs=["x"],
                                      dim="d_model")},
                outputs=["n"])},
            model=[
                ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                           boundary_role="input_embedding",
                           dims={"n_vocab": "n_vocab", "d_model": "d_model"}),
                ModelLayer(name="layers", template="block", repeat="n_layers",
                           inputs=["embed"], streams=hidden),
                ModelLayer(name="norm", op="rmsnorm", dim="d_model",
                           inputs=["layers"]),
                ModelLayer(name="g", group=True, aux_kind="mtp",
                           inputs=["norm"], repeat="3",
                           checkpoint={"prefix": "mtp", "index": "local"},
                           streams=hidden if group_streams else {},
                           layers=[child]),
                ModelLayer(name="out", op="rmsnorm", dim="d_model",
                           inputs=[reads]),
            ])

    def test_a_group_carrying_nothing_is_not_read_by_name(self):
        with pytest.raises(Exception, match="reads 'g' by name"):
            self._spec()

    def test_a_group_carrying_its_stream_is_read_by_name(self):
        self._spec(group_streams=True)

    def test_under_a_group_whose_block_repeats_the_block_is_the_stack(self):
        # The group is not a stack; its repeated child is, and carrying
        # nothing it is not read by name either.
        self._spec(child_repeat="2", child_streams=True)
        with pytest.raises(Exception, match=re.escape("reads 'g.layers' by name")):
            self._spec(child_repeat="2", reads="g.layers")


def test_one_stack_reads_a_stream_one_way():
    with pytest.raises(Exception, match="both at its block boundary"):
        _resolved("as_both_ways", edit=lambda spec: spec.replace(
            "        inputs: [taps]\n        streams:\n",
            "        inputs: [taps]\n        streams:\n"
            "          taps: {{ read: each }}\n", 1))


#: The mergers split like vLLM's deepstack mergers: fc1 column-parallel,
#: fc2 row-parallel, so each element leaves its merger as a pending sum.
SPLIT_TAP = """\
  tap:
    graph:
      fc1: {{ op: linear, in: x, dims: {{ in: d_model, out: d_ffn }},
              weight_axes: [{{ world: dense_ffn, parts: d_ffn, fit: exact }}, {{ replicated: true }}] }}
      fc2: {{ op: linear, in: fc1, dims: {{ in: d_ffn, out: d_model }},
              weight_axes: [{{ replicated: true }}, {{ world: dense_ffn, parts: d_ffn, fit: exact }}] }}
    outputs: [fc2]
"""


def _split_mergers(spec: str) -> str:
    start = spec.index("  tap:\n")
    end = spec.index("  inject:\n")
    return (spec[:start] + SPLIT_TAP + spec[end:]).replace(
        "  d_model: config.hidden_size\n",
        "  d_model: config.hidden_size\n  d_ffn: config.hidden_size\n", 1)


def _encode_reductions(program: Program, **parallel):
    from backend.arch.layout.materializer import materialize_comm_events
    analysis = Deployment(program, config=ParallelConfig(**parallel)).at(
        batch=1, media=MEDIA).during("encode", stack="vis").analyze()
    return [(analysis.program.by_uid(event.placement_uid).display_path,
             event.site.value, event.count)
            for event in materialize_comm_events(analysis)
            if event.op == "all_reduce"]


class TestSplitAppendedValues:
    """A merger split over TP leaves each element it appends as a pending
    sum. The encoder hands every element over whole, so each is reduced
    once, in the encode walk."""

    def test_each_element_is_reduced_once(self):
        program = Program.of(_resolved("as_split", edit=_split_mergers))
        assert _encode_reductions(program, tp_dense_ffn=2) == [
            ("fc2", "model_output", 2)]

    def test_a_stage_exit_settled_by_its_close_is_not_reduced_again(self):
        # The tower ends in a split projection of model-level ops, like
        # Qwen3-VL's final merger: the tower's stage close reduces its exit,
        # and the handoff reads what the close settled.
        head = (
            "      - {{ name: vis.fc1, op: linear, inputs: [vis.blocks],"
            " dims: {{ in: d_model, out: d_ffn }}, weight_axes:"
            " [{{ world: dense_ffn, parts: d_ffn, fit: exact }}, {{ replicated: true }}] }}\n"
            "      - {{ name: vis.fc2, op: linear, inputs: [vis.fc1],"
            " dims: {{ in: d_ffn, out: d_model }}, weight_axes:"
            " [{{ replicated: true }}, {{ world: dense_ffn, parts: d_ffn, fit: exact }}] }}\n")

        def edit(spec):
            return _split_mergers(spec).replace(
                "inputs: [embed, vis.blocks],", "inputs: [embed, vis.fc2],",
                1).replace("  - {{ name: merge,", head + "  - {{ name: merge,", 1)
        program = Program.of(_resolved("as_split_exit", edit=edit))
        assert sorted(_encode_reductions(program, tp_dense_ffn=2)) == [
            ("fc2", "model_output", 2), ("vis.fc2", "stage_close", 1)]
