"""A weight may carry axes ahead of its op's own, and `linear` batches over
them the way `torch.matmul` does.

A node states the shape of the weight it stores (`shape:`). Its trailing axes
are the op's own weight (`linear`'s `(out, in)`); every axis before them is a
batch axis, and the activation's batch axes must end in those same axes,
position by position, each the same extent — matrix `i` of the weight
contracts batch slice `i` of the activation. An expert bank is one such
weight: `[n_expert_copies, d_ffn, d_model]` against an activation
`[n_expert_copies, M, d_model]`. Nothing here is about experts: the leading
axes are whatever the declaration writes, as many as it writes.

What this file pins:

* the stored tensors carry the leading axes — the weight, its bias
  (`[..., out]`), and a quant format's scale grid (the leading axes pass
  through, the last two axes are tiled);
* the parameter count is the elements of the declared shape, however the op
  counts parameters (a `tensors` list, a `params_fn`, a custom op's
  `params:`);
* an activation whose batch axes do not end in the weight's leading axes is
  refused, as is a declared shape that does not end in the op's own weight;
* the FLOPs come off the shape, with nothing restated;
* a node that declares no shape is exactly what it was.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.dim import eval_dim_expr, value_of
from backend.arch.eval.formats import FormatPolicy
from backend.arch.eval.op_ref import OpRef
from backend.arch.eval.refusal import SpecRefusal
from backend.arch.eval.storage import StorageKind
from backend.arch.ops import get_op
from backend.arch.schema.custom_ops import CustomOpSpec

E, G, M, IN, OUT = 3, 2, 5, 64, 32


def _expr(expr, dims):
    return eval_dim_expr(str(expr), dims)


def _ref(name: str) -> OpRef:
    return OpRef(name, get_op(name), False)


def _stored(ref: OpRef, dims):
    params = ref.params(dims, _expr)
    return params, ref.stored_tensors(dims, params, "experts.gate_proj",
                                      _expr)


def _ints(extents):
    return tuple(int(value_of(x)) for x in extents)


def _shapes(tensors):
    return [_ints(t.shape) for t in tensors]


LINEAR = {"in": IN, "out": OUT, "bias": 1}


class TestLinear(unittest.TestCase):
    def test_a_node_that_declares_no_shape_is_what_it_was(self):
        params, tensors = _stored(_ref("linear"), LINEAR)
        self.assertEqual(int(value_of(params)), IN * OUT + OUT)
        self.assertEqual(_shapes(tensors), [(OUT, IN), (OUT,)])
        self.assertEqual([t.axis_refs for t in tensors], [None, (0,)])

    def test_one_leading_axis_stacks_the_weight_and_its_bias(self):
        dims = {**LINEAR, "_shape": [E, OUT, IN]}
        params, tensors = _stored(_ref("linear"), dims)
        self.assertEqual(int(value_of(params)), E * (IN * OUT + OUT))
        self.assertEqual(_shapes(tensors), [(E, OUT, IN), (E, OUT)])
        # The bias carries the leading axis, then follows the output axis.
        self.assertEqual(tensors[1].axis_refs, (0, 1))
        # Each matrix of it is still a matrix: the last two axes.
        self.assertEqual(tensors[0].kind, StorageKind.MATRIX)
        self.assertEqual(_ints(tensors[0].matrix_dims), (IN, OUT))

    def test_any_number_of_leading_axes(self):
        dims = {**LINEAR, "_shape": [G, E, OUT, IN]}
        params, tensors = _stored(_ref("linear"), dims)
        self.assertEqual(int(value_of(params)), G * E * (IN * OUT + OUT))
        self.assertEqual(_shapes(tensors), [(G, E, OUT, IN), (G, E, OUT)])
        self.assertEqual(tensors[1].axis_refs, (0, 1, 2))

    def test_a_declared_shape_must_end_in_the_ops_own_weight(self):
        with self.assertRaises(SpecRefusal):
            _stored(_ref("linear"), {**LINEAR, "_shape": [E, IN, OUT]})

    def test_the_activation_batch_axes_must_end_in_the_leading_axes(self):
        ref = _ref("linear")
        one = {**LINEAR, "_shape": [E, OUT, IN]}
        two = {**LINEAR, "_shape": [G, E, OUT, IN]}
        self.assertIsNone(ref.input_contract_violation([[E, M, IN]], one))
        # More batch axes than the weight has: the extra ones broadcast.
        self.assertIsNone(ref.input_contract_violation([[G, E, M, IN]], one))
        self.assertIsNone(ref.input_contract_violation([[G, E, M, IN]], two))
        for refused, dims in (([E + 1, M, IN], one),     # a different extent
                              ([E, G, M, IN], two),      # out of position
                              ([M, IN], one)):           # no batch axis
            with self.subTest(activation=refused):
                self.assertIsNotNone(
                    ref.input_contract_violation([refused], dims))
        # With no leading axes the weight broadcasts over any batch.
        self.assertIsNone(ref.input_contract_violation([[E + 1, M, IN]],
                                                       LINEAR))


class TestOpsThatCountWithAFunction(unittest.TestCase):
    def test_linear_param(self):
        dims = {"in": IN, "out": OUT, "_shape": [E, OUT, IN]}
        params, tensors = _stored(_ref("linear_param"), dims)
        self.assertEqual(int(value_of(params)), E * IN * OUT)
        self.assertEqual(_shapes(tensors), [(E, OUT, IN)])

    def test_a_custom_op(self):
        """gpt-oss's `mxfp4_fused_linear` shape, declared in YAML."""
        cop = CustomOpSpec(
            flops="B * s_ctx * Out + 2 * B * s_ctx * In * Out",
            params="In * Out", weight_shape=["Out", "In"],
            signature={"x": ["B", "s_ctx", "In"]},
            shape_kind="axis_replace", out_key="Out", axis_locked=-1,
            couples_on=["In"],
            description="A 4-bit packed linear whose dequantisation no "
                        "registered primitive expresses.")
        dims = {"In": IN, "Out": OUT, "_shape": [E, OUT, IN]}
        params, tensors = _stored(OpRef("packed_linear", cop, True), dims)
        self.assertEqual(int(value_of(params)), E * IN * OUT)
        self.assertEqual(_shapes(tensors), [(E, OUT, IN)])

    def test_an_op_whose_params_are_not_its_weight_cannot_declare_one(self):
        cop = CustomOpSpec(
            params="In * Out + 7", weight_shape=["Out", "In"],
            signature={"x": ["B", "s_ctx", "In"]},
            shape_kind="axis_replace", out_key="Out", axis_locked=-1,
            couples_on=["In"],
            description="Counts parameters its weight does not hold.")
        dims = {"In": IN, "Out": OUT, "_shape": [E, OUT, IN]}
        with self.assertRaises(ValueError):
            OpRef("odd", cop, True).params(dims, _expr)


class TestTheScaleGridPassesTheLeadingAxesThrough(unittest.TestCase):
    """MXFP4's `_scales` tile one matrix, `[out, in / 32]`; a weight with
    leading axes carries one grid per matrix, so the grid keeps them — the
    checkpoint's `[E, out, in / 32]`."""

    def test_mxfp4_scales(self):
        from tests.backend.arch.eval.test_eval_formats import MXFP4

        for lead in ((), (E,), (G, E)):
            with self.subTest(lead=lead):
                dims = {**LINEAR, "_shape": [*lead, OUT, IN]}
                _, (weight, _bias) = _stored(_ref("linear"), dims)
                rep = FormatPolicy(per_role={"ffn_moe": MXFP4}).for_node(
                    weight.name, "ffn_moe", weight)
                (scale,) = rep.metadata
                self.assertEqual(_shapes([scale]), [(*lead, OUT, IN // 32)])
                self.assertEqual(rep.metadata_geometry[0].stored_transform,
                                 "matrix")
                # A deployment re-projects the same grid.
                (projected,) = rep.project_metadata(weight)
                self.assertEqual(_shapes([projected]),
                                 [(*lead, OUT, IN // 32)])


class TestEndToEnd(unittest.TestCase):
    """One walk: a `[B, s, d]` activation through a linear whose weight is
    `[n_w, f, d]` with `n_w` bound to the batch — one weight per request,
    the smallest graph whose activation already leads with the axis."""

    @classmethod
    def setUpClass(cls):
        from backend.arch.eval.program import Program
        from backend.arch.loader import ResolvedSpec
        from backend.arch.schema import (ArchSpec, BlockGraph, GraphNode,
                                         ModelLayer)

        block = BlockGraph(inputs=["x"], graph={
            "w": GraphNode(op="linear", inputs=["x"],
                           dims={"in": "d_model", "out": "d_f"},
                           shape=["n_w", "d_f", "d_model"],
                           weight_axes=[{"replicated": True}] * 3),
        }, outputs=["w"])
        spec = ArchSpec(
            model_inputs={"input_ids": {"kind": "tokens"}}, name="demo",
            dims={"d_model": "config.hidden_size", "n_w": "config.n_w",
                  "d_f": "config.d_f"},
            model=[
                ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                           boundary_role="input_embedding",
                           dims={"n_vocab": "n_vocab", "d_model": "d_model"}),
                ModelLayer(name="layers", template="block", repeat="n_layers",
                           inputs=["embed"]),
            ],
            blocks={"block": block})
        dims = {"d_model": 24, "n_w": 3, "d_f": 5, "n_layers": 1,
                "n_vocab": 32}
        config = {"hidden_size": 24, "n_w": 3, "d_f": 5,
                  "torch_dtype": "bfloat16"}
        cls.program = Program.of(ResolvedSpec(spec, dims, config, {}))
        cls.node = next(n for n in cls.program.nodes() if n.name == "w")

    def _analysis(self, batch):
        from backend.arch.eval.transforms import Deployment
        return Deployment(self.program).at(batch=batch, input_tokens=4) \
            .analyze()

    def test_the_weight_is_stored_with_its_leading_axis(self):
        analysis = self._analysis(3)
        (weight,) = analysis.storage.values[self.node.uid]
        self.assertEqual(_ints(weight.shape), (3, 5, 24))
        self.assertEqual(int(value_of(
            analysis.cost.values[self.node.uid].params)), 3 * 5 * 24)

    def test_the_flops_and_the_output_come_off_the_shape(self):
        analysis = self._analysis(3)
        cost = analysis.cost.values[self.node.uid]
        self.assertEqual(int(value_of(cost.flops)), 2 * 3 * 4 * 24 * 5)
        self.assertEqual(_ints(analysis.shapes.values[self.node.uid].out),
                         (3, 4, 5))

    def test_an_activation_that_does_not_lead_with_it_is_refused(self):
        with self.assertRaises(SpecRefusal):
            self._analysis(2)


class TestTheOutputRankIsTheActivations(unittest.TestCase):
    """A weight-bearing op's `shape:` is its WEIGHT's. Only a source op's
    declared shape is its output (`ShapeNode.out_shape`), so a `[B, E, s, d]`
    activation through a linear whose weight is `[E, f, d]` comes out
    `[B, E, s, f]` — rank 4, as it went in — and not the weight's rank."""

    @classmethod
    def setUpClass(cls):
        from backend.arch.eval.program import Program
        from backend.arch.eval.transforms import Deployment
        from backend.arch.loader import ResolvedSpec
        from backend.arch.schema import (ArchSpec, BlockGraph, GraphNode,
                                         ModelLayer)

        heads = {"n_heads": "n_e", "d_head": "d_e"}
        block = BlockGraph(inputs=["x"], graph={
            "xs": GraphNode(op="view_split", inputs=["x"], axis=-1,
                            dims={"in": "d_model", **heads}),
            "xt": GraphNode(op="transpose", inputs=["xs"], axes=[0, 2, 1, 3]),
            "w": GraphNode(op="linear", inputs=["xt"],
                           dims={"in": "d_e", "out": "d_f"},
                           shape=["n_e", "d_f", "d_e"],
                           weight_axes=[{"replicated": True}] * 3),
        }, outputs=["w"])
        spec = ArchSpec(
            model_inputs={"input_ids": {"kind": "tokens"}}, name="demo",
            dims={"d_model": "config.hidden_size", "n_e": "config.n_e",
                  "d_e": "config.d_e", "d_f": "config.d_f"},
            model=[
                ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                           boundary_role="input_embedding",
                           dims={"n_vocab": "n_vocab", "d_model": "d_model"}),
                ModelLayer(name="layers", template="block", repeat="n_layers",
                           inputs=["embed"]),
            ],
            blocks={"block": block})
        dims = {"d_model": 24, "n_e": 3, "d_e": 8, "d_f": 5, "n_layers": 1,
                "n_vocab": 32}
        config = {"hidden_size": 24, "n_e": 3, "d_e": 8, "d_f": 5,
                  "torch_dtype": "bfloat16"}
        program = Program.of(ResolvedSpec(spec, dims, config, {}))
        cls.analysis = Deployment(program).at(batch=2, input_tokens=4) \
            .analyze()
        cls.node = next(n for n in program.nodes() if n.name == "w")

    def test_a_rank_four_activation_stays_rank_four(self):
        self.assertEqual(
            _ints(self.analysis.shapes.values[self.node.uid].out),
            (2, 3, 4, 5))

    def test_its_flops_and_weight(self):
        cost = self.analysis.cost.values[self.node.uid]
        self.assertEqual(int(value_of(cost.flops)), 2 * 2 * 3 * 4 * 8 * 5)
        (weight,) = self.analysis.storage.values[self.node.uid]
        self.assertEqual(_ints(weight.shape), (3, 5, 8))

    def test_a_declared_weight_shape_is_not_the_nodes_output(self):
        self.assertIsNone(self.node.out_shape)
        self.assertEqual(len(self.node.shape), 3)


if __name__ == "__main__":
    unittest.main()
