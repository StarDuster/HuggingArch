"""A stack reading the prefix of the stream it appends to (`read: all`), and
entry appends (`before:`).

Kimi-K3's decoder (`modeling_kimi_linear.py:973-1045`) appends the ENTRY of
layers 0, K, 2K, … to a history `block_residual`, and every layer combines its
input with the whole history appended before it (`_apply_attn_res`,
`:1075-1088`); after the last layer one more combine reads all of it
(`:1215-1232`). A layer's history length grows along the stack, so the layers
one template stands for do not share a shape: they are split into groups by
what they see, and each group is a link of the carried chain.

The toy stack below has 7 layers and K = 3: entries of layers 0, 3 and 6 are
appended, so layers 0..7 (7 = the model-level reader) see 0, 1, 1, 1, 2, 2, 2
and 3 of them.
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from backend.arch.eval import traffic as traffic_mod
from backend.arch.eval.dim import value_of
from backend.arch.eval.program import BlockStage, Program
from backend.arch.eval.transforms import Deployment
from backend.arch.layout.parallel_config import ParallelConfig
from backend.arch.loader import SpecLoader
from backend.arch.ops import OP_REGISTRY
from backend.arch.schema.model import StreamSpec

SPEC = """\
match: {{ model_type: [{model_type}], model_id: [{model_type}] }}
dims:
  d_model: config.hidden_size
  n_vocab: d_model
  n_layers: "7"
  Kar: "3"
blocks:
  first:
    graph:
      n: {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      h: {{ op: stack_init, in: x }}
      w: {{ op: parameter_vector, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      p: {{ op: parameter_tensor, shape: ["1", d_model], weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      c: {{ op: attn_res_combine, in: [n, h, w, p] }}
    outputs: [c]
  plain:
    reads: [hist]
    graph:
      w: {{ op: parameter_vector, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      p: {{ op: parameter_tensor, shape: ["1", d_model], weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      a: {{ op: attn_res_combine, in: [x, hist, w, p] }}
      n: {{ op: rmsnorm, in: a, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      o: {{ op: add, in: [x, n], dim: d_model }}
    outputs: [o]
  start:
    reads: [hist]
    graph:
      w: {{ op: parameter_vector, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      p: {{ op: parameter_tensor, shape: ["1", d_model], weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      a: {{ op: attn_res_combine, in: [x, hist, w, p] }}
      h2: {{ op: stack_append, in: [hist, x] }}
      w2: {{ op: parameter_vector, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      p2: {{ op: parameter_tensor, shape: ["1", d_model], weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      b: {{ op: attn_res_combine, in: [a, h2, w2, p2] }}
    outputs: [b]
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
      hist: {{ from: hidden, before: {before}, read: all, src: "hist.append(h)" }}
    per_layer:
      assign:
        - {{ type: first, layers: [{first}] }}
        - {{ type: start, stride: Kar, offset: Kar }}
        - {{ type: plain, rest: true }}
      map:
        first: {{ template: first }}
        start: {{ template: start }}
        plain: {{ template: plain }}
  - {{ name: out_w, op: parameter_vector, dim: d_model, weight_axes: [{{ replicated: true }}] }}
  - {{ name: out_p, op: parameter_tensor, shape: ["1", d_model], weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - {{ name: out, op: attn_res_combine, inputs: [layers, hist, out_w, out_p] }}
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: d_model }}, inputs: [out],
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""

D = 8
B, S = 1, 4


def _resolved(model_type: str, *, before="{ every: Kar }", first='"0"',
              edit=lambda spec: spec):
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        spec = edit(SPEC).format(model_type=model_type, before=before,
                                 first=first)
        (base / "models" / f"{model_type}.yaml").write_text(spec)
        loader = SpecLoader(base_dir=str(base))
        return loader.match({"model_type": model_type, "torch_dtype": "bfloat16", "hidden_size": D},
                            model_id=model_type)


def _analyze(program: Program, **parallel):
    deployment = Deployment(program).at(batch=B, input_tokens=S)
    if parallel:
        deployment = replace(deployment, config=ParallelConfig(**parallel))
    return deployment.analyze()


def _no_dim(expr: str) -> int:
    raise AssertionError(f"no dim is read, but {expr!r} was asked for")


def _groups(program: Program):
    return [(t.name, t.layer_indices, dict(t.visible)) for t in program.blocks]


def _node(program: Program, template: str, name: str, layer: int):
    (block,) = [t for t in program.blocks
                if t.name == template and layer in t.layer_indices]
    (node,) = [n for n in block.nodes() if n.name == name]
    return node


def _extents(shape):
    return [int(value_of(x)) for x in shape]


class TestBefore:
    """`before: j` appends iteration j's entry; iteration 0's entry is the
    stack's input."""

    def test_entries_and_their_writers(self):
        spec = StreamSpec(**{"from": "hidden", "before": [0, 3, 6],
                             "src": "a"})
        # The value at iteration j's entry is iteration j-1's exit.
        assert spec.iterations(7, {}, _no_dim) == (-1, 2, 5)
        assert spec.writers(7, {}, _no_dim) == (0, 3, 6)

    def test_every_dim_names_0_n_2n(self):
        spec = StreamSpec(**{"from": "hidden", "before": {"every": "Kar"},
                             "src": "a"})
        assert spec.writers(93, {}, {"Kar": 12}.__getitem__) == (
            0, 12, 24, 36, 48, 60, 72, 84)

    def test_every_needs_a_positive_period(self):
        spec = StreamSpec(**{"from": "hidden", "before": {"every": "Kar"},
                             "src": "a"})
        with pytest.raises(ValueError, match="positive"):
            spec.writers(7, {}, {"Kar": 0}.__getitem__)

    @pytest.mark.parametrize("before", [[7], [3, 3], [5, 2]])
    def test_what_names_no_iteration_is_refused(self, before):
        spec = StreamSpec(**{"from": "hidden", "before": before, "src": "a"})
        with pytest.raises(ValueError):
            spec.writers(7, {}, _no_dim)

    @pytest.mark.parametrize("fields,match", [
        # `before: j` IS the entry of j: an index would say it twice.
        ({"from": "hidden", "before": [0], "index": "hidden_states",
          "src": "a"}, "takes no `index`"),
        ({"from": "hidden", "before": [0], "after": [1], "index": "block",
          "src": "a"}, "not both"),
        # Iteration numbers, not positions from the end.
        ({"from": "hidden", "before": [0, -1], "src": "a"}, "not negative"),
        ({"before": [0], "src": "a"}, "needs `from`"),
        ({"from": "hidden", "before": {"every": "Kar", "at": 1},
          "src": "a"}, "one dim expression"),
        ({"from": "hidden", "before": [0]}, "`src`"),
    ])
    def test_refused_at_schema(self, fields, match):
        with pytest.raises(ValueError, match=match):
            StreamSpec(**fields)

    def test_a_block_output_has_no_entry_value(self):
        with pytest.raises(Exception, match="no value at an iteration's entry"):
            _resolved("ps_from_out", edit=lambda spec: spec.replace(
                "      hidden: {{ carries: true }}\n", "", 1)
                .replace("hist: {{ from: hidden,", "hist: {{ from: out,", 1))


class TestReadAll:
    """Every occurrence of a reading component sees the elements appended
    strictly before its iteration's entry."""

    def test_visible_lengths(self):
        stream = _resolved("ps_visible").appended_streams()["hist"]
        assert stream.writers == (0, 3, 6)
        assert [stream.visible(layer) for layer in range(8)] == [
            0, 1, 1, 1, 2, 2, 2, 3]

    def test_an_after_element_is_visible_from_the_next_iteration(self):
        # `after: 2` is written at iteration 2's exit, which is iteration 3's
        # entry: iteration 3 sees it. `before: 3` names the same value, but
        # written at iteration 3's entry, so only iteration 4 on sees it.
        after = _resolved("ps_after", edit=lambda spec: spec.replace(
            "before: {before}", 'after: [2], index: block', 1)
        ).appended_streams()["hist"]
        before = _resolved("ps_before3", before="[3]").appended_streams()[
            "hist"]
        assert after.iterations == before.iterations == (2,)
        assert [after.visible(layer) for layer in (2, 3, 4)] == [0, 1, 1]
        assert [before.visible(layer) for layer in (2, 3, 4)] == [0, 0, 1]

    def test_the_readers_are_split_by_what_they_see(self):
        program = Program.of(_resolved("ps_groups"))
        assert _groups(program) == [
            ("first", (0,), {}),
            ("start", (3,), {"hist": 1}),
            ("start", (6,), {"hist": 2}),
            ("plain", (1, 2), {"hist": 1}),
            ("plain", (4, 5), {"hist": 2}),
        ]
        chain = [st.template.layer_indices
                 for st in program.stacks[0].walk_order()
                 if isinstance(st, BlockStage)]
        assert chain == [(0,), (1, 2), (3,), (4, 5), (6,)]

    def test_each_group_reads_its_own_history(self):
        program = Program.of(_resolved("ps_shapes"))
        analysis = _analyze(program)
        shapes = analysis.shapes.values

        def history_in(template, name, layer, port=1):
            node = _node(program, template, name, layer)
            return _extents(shapes[node.uid].ins[port])

        assert history_in("first", "c", 0) == [B, S, 1, D]
        assert history_in("plain", "a", 1) == [B, S, 1, D]
        assert history_in("plain", "a", 4) == [B, S, 2, D]
        # A segment start's first combine reads the old history, its second
        # the history with its own entry appended.
        assert history_in("start", "a", 3) == [B, S, 1, D]
        assert history_in("start", "b", 3) == [B, S, 2, D]
        assert history_in("start", "a", 6) == [B, S, 2, D]
        assert history_in("start", "b", 6) == [B, S, 3, D]
        out = next(n for n in program.nodes() if n.name == "out")
        assert _extents(shapes[out.uid].ins[1]) == [B, S, 3, D]

    def test_a_combine_costs_its_candidates(self):
        program = Program.of(_resolved("ps_flops"))
        analysis = _analyze(program)
        for template, name, layer, candidates in (
                ("first", "c", 0, 2), ("plain", "a", 4, 3),
                ("start", "b", 6, 4)):
            node = _node(program, template, name, layer)
            # Per layer: ~3 passes over the history and the prefix.
            flops = analysis.cost.values[node.uid].flops
            assert int(flops) == 3 * B * S * candidates * D

    def test_starting_and_growing_a_history_are_copies(self):
        program = Program.of(_resolved("ps_copies"))
        analysis = _analyze(program)
        layer = traffic_mod.derive(analysis)
        entry = B * S * D * 2          # one [B, S, D] entry at 2 bytes
        init = layer.values[_node(program, "first", "h", 0).uid]
        assert (init.act_reads, init.act_writes) == (entry, entry)
        append = layer.values[_node(program, "start", "h2", 6).uid]
        # History of 2 and the entry read; 3 entries written.
        assert (append.act_reads, append.act_writes) == (3 * entry,
                                                         3 * entry)
        for node in (_node(program, "first", "h", 0),
                     _node(program, "start", "h2", 6)):
            assert analysis.cost.values[node.uid].flops == 0

    def test_the_model_level_op_reads_the_final_value(self):
        program = Program.of(_resolved("ps_final"))
        (row,) = [r for r in program.whole_stream_pairings
                  if r.reader == "out"]
        assert (row.visible, row.sources, row.kind, row.reader_iterations,
                row.representation, row.axis) == (
            3, (0, 3, 6), "before", (), "history", None)

    def test_whole_stream_pairings_list_every_group(self):
        program = Program.of(_resolved("ps_pairings"))
        rows = sorted((r.reader_iterations, r.visible, r.sources)
                      for r in program.whole_stream_pairings
                      if r.reader == "layers")
        assert rows == [((1, 2), 1, (0,)), ((3,), 1, (0,)),
                        ((4, 5), 2, (0, 3)), ((6,), 2, (0, 3))]
        assert program.stream_pairings == ()

    def test_a_prefix_read_names_no_source_layer(self):
        """A `read: all` reads the appended prefix the iteration sees; it is
        no inner stream, so its readers declare no `reads_from`."""
        assert Program.of(_resolved("ps_reach")).ir.stream_reads == {}

    def test_the_chain_holds_under_tensor_parallelism(self):
        program = Program.of(_resolved("ps_tp"))
        _analyze(program, tp_attention=2, tp_dense_ffn=2)

    def test_a_reader_that_sees_nothing_is_refused(self):
        # Layer 0 written with the reading variant: no entry is appended
        # before its entry.
        with pytest.raises(Exception, match="no element of it has been"):
            Program.of(_resolved("ps_m0", first='"1"'))

    def test_appending_to_a_history_that_sees_nothing_is_refused(self):
        # The segment-start variant, which appends to the history it reads,
        # at layer 0.
        with pytest.raises(Exception, match="no element of it has been"):
            Program.of(_resolved("ps_append_m0", edit=lambda spec: spec.replace(
                "        - {{ type: first, layers: [{first}] }}\n", "", 1)
                .replace("        first: {{ template: first }}\n", "", 1)
                .replace("stride: Kar, offset: Kar", "stride: Kar", 1)))

    def test_stack_append_extends_at_least_one_entry(self):
        contract = OP_REGISTRY["stack_append"].input_contract
        assert "stack_init" in contract([(B, S, 0, D), (B, S, D)], {})
        assert contract([(B, S, 1, D), (B, S, D)], {}) is None

    def test_read_all_is_written_on_the_appending_declaration(self):
        with pytest.raises(ValueError, match="on the appending declaration"):
            StreamSpec(read="all")

    def test_one_read_all_per_stack(self):
        with pytest.raises(Exception, match="at most one"):
            _resolved("ps_two", edit=lambda spec: spec.replace(
                '      hist: {{ from: hidden,',
                '      hist2: {{ from: hidden, before: [0], read: all, '
                'src: "b" }}\n      hist: {{ from: hidden,', 1))

    def test_another_stack_does_not_read_the_prefix(self):
        # A second stack's component reading `hist` without a declaration.
        def edit(spec):
            return spec.replace(
                "  - {{ name: out_w,",
                "  - name: tail\n    repeat: \"1\"\n    inputs: [layers]\n"
                "    streams:\n      hidden: {{ carries: true }}\n"
                "    template: plain\n  - {{ name: out_w,", 1)
        with pytest.raises(Exception, match="does not declare"):
            _analyze(Program.of(_resolved("ps_other", edit=edit)))


class TestPipeline:
    """Under pipeline parallelism the history crosses every cut it spans:
    the stage before a cut sends the entries appended so far — the ones its
    successors and the model-level reader still read — beside the hidden
    state itself, which crosses every cut."""

    @staticmethod
    def _sends(**parallel):
        from backend.arch.layout.materializer import materialize_comm_events
        analysis = _analyze(Program.of(_resolved("ps_pp")), **parallel)
        return analysis, [e for e in materialize_comm_events(analysis)
                          if e.op == "send_recv"]

    def test_one_stage_sends_nothing(self):
        assert not self._sends()[1]

    @pytest.mark.parametrize("pp,live", [
        # 7 layers: pp=2 cuts at 4 (entries of layers 0, 3 appended
        # before it); pp=4 holds layers 0-1, 2-3, 4-5, 6 and cuts at 2, 4
        # and 6 (one, two and two entries).
        (2, {0: 2}),
        (4, {0: 1, 1: 2, 2: 2}),
    ])
    def test_each_cut_sends_the_entries_appended_before_it(self, pp, live):
        analysis, sends = self._sends(pp=pp)
        by_stage = {}
        for send in sends:
            stage = dict(send.meta)["cut"]
            by_stage[stage] = by_stage.get(stage, 0) + 1
            # One [B, S, D] tensor, once per step, its whole bytes.
            assert send.payload_shape == (B, S, D)
            assert send.count == 1
            assert send.bytes_per_rank == B * S * D * 2
            assert (send.world, send.group_size) == ("pp", pp)
        # The entries, and the hidden state once per cut.
        assert by_stage == {stage: n + 1 for stage, n in live.items()}
        # Entry 0 is the stack's input and every cut before the last
        # reader carries it.
        (uid, _port), _state = analysis.appended["hist"][0]
        assert sorted(dict(s.meta)["cut"] for s in sends
                      if s.source.node_uid == uid) == sorted(live)

    def test_a_stage_pays_its_own_exit_cut(self):
        from backend.arch.layout.materializer import materialize_comm_events
        from backend.arch.layout.stages import StagePlacement
        analysis, _ = self._sends(pp=4)
        placement = StagePlacement.of(analysis)
        # Entries, plus the hidden state across each exit cut.
        for stage, live in ((0, 2), (1, 3), (2, 3), (3, 0)):
            sends = [e for e in materialize_comm_events(
                analysis, stage=placement.view(stage))
                if e.op == "send_recv"]
            assert len(sends) == live
            assert {dict(e.meta)["cut"] for e in sends} <= {stage}


def test_an_unset_before_is_not_in_the_dump():
    """Specs that do not use `before` dump as they did, so their
    fingerprints do not move."""
    assert "before" not in StreamSpec(carries=True).model_dump()
