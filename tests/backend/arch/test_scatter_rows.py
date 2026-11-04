"""Row scatter ops and strict stack binding.

`scatter_rows_replace` / `scatter_rows_add` write `v: [B, N, D]` into the
placeholder rows of `x: [B, S, D]` (HF `masked_scatter`, Qwen3-VL DeepStack).
A stack binds its `inputs:` to each variant's formals one to one: the decoder
that used to name `[embed_tokens, vision]` against a one-formal block had the
vision output dropped by a non-strict `zip`, and now names a merge instead.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from backend.arch.eval.dim import value_of
from backend.arch.ops import OP_REGISTRY

# One 2x3 image per request: patchify at patch 1 makes six rows.
_IMAGE = {"pixel_values": {"n_img": 1, "img_h": 2, "img_w": 3}}

SPEC = """\
match: {{ model_type: [{name}], model_id: [{name}] }}
dims:
  d_model: config.hidden_size
  n_layers: "2"
blocks:
  blk:
    graph:
      n: {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      o: {{ op: add, in: [x, n], dim: d_model }}
    outputs: [o]
  wide:
    inputs: [x, y]
    graph:
      o: {{ op: add, in: [x, y], dim: d_model }}
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
      - {{ name: vis.patch, op: linear, inputs: [vis.patchify], dims: {{ in: d_model, out: d_model }},
          weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
{tower}{stack}
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: d_model }}, inputs: [layers],
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""

_SLOTS = ('prompt_slots: { domain: pixel_values, defines: true, '
          'source_anchor: "a test merge" }')

MERGED = f"""\
  - {{ name: merge, op: scatter_rows_replace, inputs: [embed, vis.patch], {_SLOTS} }}
  - {{ name: layers, repeat: n_layers, streams: {{ hidden: {{ carries: true }} }}, template: blk, inputs: [merge] }}"""

TWO_INPUTS = f"""\
  - {{ name: layers, repeat: n_layers, streams: {{ hidden: {{ carries: true }} }}, template: blk, inputs: [embed, vis.patch], {_SLOTS} }}"""

PER_LAYER_EMPTY_WIDE = f"""\
  - {{ name: merge, op: scatter_rows_replace, inputs: [embed, vis.patch], {_SLOTS} }}
  - name: layers
    repeat: n_layers
    inputs: [merge]
    streams: {{ hidden: {{ carries: true }} }}
    per_layer:
      assign:
        - {{ type: wide, range: ["1", "0"] }}
        - {{ type: narrow, rest: true }}
      map:
        wide: {{ template: wide }}
        narrow: {{ template: blk }}"""


def _load(name: str, stack: str, tower: str = ""):
    from backend.arch.loader import SpecLoader
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / f"{name}.yaml").write_text(
            SPEC.format(name=name, stack=stack, tower=tower))
        return SpecLoader(base_dir=str(base)).match(
            {"model_type": name, "torch_dtype": "bfloat16", "hidden_size": 8}, model_id=name)


class TestStrictStackBinding:
    def test_a_stack_naming_more_inputs_than_formals_is_refused_at_load(self):
        with pytest.raises(Exception, match="one to one"):
            _load("scatter_two_inputs", TWO_INPUTS)

    def test_every_per_layer_variant_is_checked(self):
        # The `wide` assignment selects no layer under this config; another
        # config may select it, so its two formals are held to the stack's
        # one input all the same.
        with pytest.raises(Exception, match="variant 'wide'"):
            _load("scatter_per_layer_empty_wide", PER_LAYER_EMPTY_WIDE)

    def test_the_merge_is_the_stack_s_one_input(self):
        from backend.arch.eval.program import Program
        from backend.arch.eval.transforms import Deployment
        program = Program.of(_load("scatter_merged", MERGED))
        deployment = Deployment(program).at(batch=1, input_tokens=4,
                                            media=_IMAGE)
        analysis = deployment.analyze()
        encoded = deployment.during("encode", stack="vis").analyze()
        merge = next(n for n in program.nodes() if n.name == "merge")
        patch = next(n for n in program.nodes() if n.name == "vis.patch")
        # The vision output is read: the merge's second input is the tensor
        # the tower hands over, which the language model's step holds with
        # no rows (the encoder is its own accounting) ...
        handed = analysis.shapes.values[merge.uid].ins[1]
        assert [value_of(d) for d in handed] == [1, 0, 8]
        # ... and which the encode walk hands over per request: one 2x3
        # image of 1x1 patches.
        (tensor,) = deployment.handed_over()["vis.patch"]
        assert (tensor.rows, value_of(tensor.width)) == (6, 8)
        assert [
            value_of(d) for d in encoded.shapes.values[patch.uid].out] == [1, 6, 8]


class TestRowScatterOps:
    @pytest.mark.parametrize("name,flops", [
        ("scatter_rows_replace", 0), ("scatter_rows_add", 2 * 3 * 4)])
    def test_flops_come_from_the_written_rows(self, name, flops):
        op = OP_REGISTRY[name]
        assert op.flops_fn({"B": 2, "rows": 3, "dim": 4, "s_ctx": 10}) == flops
        assert op.arity() == 2

    @pytest.mark.parametrize("name", ["scatter_rows_replace", "scatter_rows_add"])
    def test_the_row_contract(self, name):
        contract = OP_REGISTRY[name].input_contract
        assert contract([(2, 10, 4), (2, 10, 4)], {}) is None
        assert contract([(2, 10, 4), (2, 3, 4)], {}) is None
        assert "rows into a sequence" in contract([(2, 10, 4), (2, 11, 4)], {})
        assert "disagree on D" in contract([(2, 10, 4), (2, 3, 5)], {})
        assert "disagree on B" in contract([(2, 10, 4), (1, 3, 4)], {})
        # No implicit flatten of a [B*N, D] source.
        assert "ranks" in contract([(2, 10, 4), (6, 4)], {})

    def test_the_walk_refuses_more_rows_than_positions(self):
        from backend.arch.eval.program import Program
        from backend.arch.eval.transforms import Deployment
        # The defining merge fills one prompt position per patch. A second,
        # non-defining edge adds a doubled tower sequence onto those
        # positions: more rows than the domain has, refused where the
        # encoder hands its rows over.
        tower = (
            "      - { name: vis.again, op: rmsnorm, dim: d_model, "
            "weight_axes: [{ replicated: true }], inputs: [vis.patch] }\n"
            "      - { name: vis.more, op: concat, axis: -2, "
            "dims: { in_a: d_model, in_b: d_model, out: d_model }, "
            "inputs: [vis.patch, vis.again] }\n")
        merge, layers = MERGED.split("\n")
        stack = "\n".join([
            merge,
            "  - { name: again, op: scatter_rows_add, inputs: [merge, vis.more], "
            "prompt_slots: { domain: pixel_values, defines: false, "
            "source_anchor: \"a test add\" } }",
            layers.replace("inputs: [merge]", "inputs: [again]")])
        program = Program.of(_load("scatter_too_many_rows", stack, tower))
        with pytest.raises(ValueError, match=r"'vis\.more' adds 12 rows onto "
                           r"the prompt positions of 'pixel_values', and "
                           r"'vis\.patch' defines 6 of them"):
            Deployment(program).at(batch=1, input_tokens=4,
                                   media=_IMAGE).handed_over()


class TestRowScatterLayout:
    """What the dedicated handler refuses: any placement it declares no row
    co-shard for, and a B or D placement `v` does not share with `x`."""

    @staticmethod
    def _rule():
        from backend.arch.layout.propagation import LayoutPropagation
        from backend.inference.parallel_config import ParallelConfig
        return LayoutPropagation.of(
            ParallelConfig(cp=2, tp_attention=2, tp_dense_ffn=2), "prefill")

    @staticmethod
    def _node():
        class Node:
            display_path = "merge"
        return Node()

    def _call(self, x, v):
        from backend.arch.layout.geometry import scatter_rows
        return scatter_rows(self._rule(), self._node(), [x, v],
                            [(1, 16, 8), (1, 5, 8)])

    def _states(self):
        from backend.arch.layout.edge import EdgeState
        from backend.arch.layout.layout import sharded
        rule = self._rule()
        rep = rule.replicated()
        cp = sharded(rule.families["cp"])
        tp = sharded(rule.families["tp_attention"])
        return EdgeState, rep, cp, tp

    def test_the_declared_co_shard_passes(self):
        EdgeState, rep, cp, _tp = self._states()
        required, out = self._call(EdgeState((rep, cp, rep)),
                                   EdgeState((rep, rep, rep)))
        rows = required[1].layout[1]
        assert rows.pad_to == 1 and rows.group.id == "cp"
        assert out.layout == (rep, cp, rep)

    @pytest.mark.parametrize("case", ["rows_cut_seq_whole", "rows_other_world",
                                      "batch", "feature"])
    def test_undeclared_placements_are_refused(self, case):
        EdgeState, rep, cp, tp = self._states()
        x, v = {
            "rows_cut_seq_whole": ((rep, rep, rep), (rep, cp, rep)),
            "rows_other_world": ((rep, cp, rep), (rep, tp, rep)),
            "batch": ((tp, cp, rep), (rep, rep, rep)),
            "feature": ((rep, cp, tp), (rep, rep, rep)),
        }[case]
        with pytest.raises(ValueError):
            self._call(EdgeState(x), EdgeState(v))

    def test_mixed_reduction_states_are_refused(self):
        from backend.arch.layout.value import SUM, Partial
        EdgeState, rep, cp, _tp = self._states()
        pending = Partial(reducer=SUM, group=self._rule().families["tp_attention"])
        with pytest.raises(ValueError, match="reduction states"):
            self._call(EdgeState((rep, cp, rep), pending),
                       EdgeState((rep, rep, rep)))
        # The same pending sum on both sides rides through an add.
        _required, out = self._call(EdgeState((rep, cp, rep), pending),
                                    EdgeState((rep, rep, rep), pending))
        assert out.value == pending

    def test_only_the_add_defers_a_pending_sum(self):
        from backend.arch.layout.foldlaw import (HOMOMORPHISM_REGISTRY,
                                                 may_defer_reduction)
        from backend.arch.layout.value import COMPLETE, SUM, Partial
        assert "scatter_rows_replace" not in HOMOMORPHISM_REGISTRY
        law = HOMOMORPHISM_REGISTRY["scatter_rows_add"]
        p = Partial(reducer=SUM, group=self._rule().families["tp_attention"])
        yes = lambda slot, cls: True
        assert may_defer_reduction(law, (("x", p), ("v", p)), yes)
        assert not may_defer_reduction(law, (("x", p), ("v", COMPLETE)), yes)
