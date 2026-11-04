"""FLOPs bound to the tensors that arrived, not to the dim environment.

Falsification targets:
  - the batch fold must be exact, because it is the whole of the rope fix: a
    per-head op on `[B, N, S, D]` described by a rank-3 template must cost N
    times what the template alone says, and getting it wrong replaces an
    N-fold under-count with an N-fold over-count;
  - a template that does not describe the tensors must produce NOTHING, not a
    partial binding — when one input's rank-3 template meets a rank-4 tensor,
    folding that input's leading axis in as everyone's batch triples the op;
  - a broadcast operand must not veto the derivation, or every `mul` against a
    per-channel vector falls back;
  - the environment must still answer for labels no input carries, or a
    projection's output width has nowhere to come from.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.dim import (
    Dim, ReadTracking, eval_dim_expr,
)
from backend.arch.eval.flops import (
    bind_labels, flops_from_shape, shape_derived_dims,
)
from backend.arch.axis_propagation import compute_shape_out
from backend.arch.ops import OP_REGISTRY


@dataclass(frozen=True)
class FakeOp:
    """The input-geometry fields the derivation reads."""
    shape_in: Any = ()
    shape_kind: str = "explicit"
    couples_on: Any = ()


ELEMENTWISE = FakeOp(("B", "s_ctx", "dim"))
LINEAR = FakeOp(("B", "s_ctx", "in"))
ATTN_SCORE = FakeOp((("B", "n_heads", "s_query", "d_head"),
                     ("B", "n_heads", "s_kv", "d_head")))


class TestExactAlignment(unittest.TestCase):
    def test_template_matching_the_rank_binds_every_label(self):
        labels, batch = bind_labels(LINEAR, [(2, 128, 4096)])
        self.assertEqual(labels, {"B": 2, "s_ctx": 128, "in": 4096})
        self.assertEqual(batch, 1)

    def test_shared_labels_across_inputs_agree(self):
        labels, batch = bind_labels(
            ATTN_SCORE, [(2, 64, 128, 96), (2, 64, 512, 96)])
        self.assertEqual(labels["n_heads"], 64)
        self.assertEqual(labels["d_head"], 96)
        self.assertEqual((labels["s_query"], labels["s_kv"]), (128, 512))
        self.assertEqual(batch, 1)

    def test_a_disagreeing_shared_label_is_a_conflict(self):
        """Both inputs claim `d_head` and they differ. The walk and the
        per-rank projection read the same shapes, so both refuse: picking one
        extent would be a silent factor, and dropping the label quietly would
        let the walk price a node the projection rejects."""
        with self.assertRaises(ValueError):
            bind_labels(ATTN_SCORE, [(2, 64, 128, 96), (2, 64, 512, 128)])

    def test_a_caller_may_ask_for_the_lenient_reading(self):
        """`require_consistent_labels=False` is the stated opt-in: the label is
        dropped and the environment's dim answers; agreeing labels survive."""
        labels, _ = bind_labels(
            ATTN_SCORE, [(2, 64, 128, 96), (2, 64, 512, 128)],
            require_consistent_labels=False)
        self.assertNotIn("d_head", labels)
        self.assertEqual(labels["n_heads"], 64)

    def test_missing_input_keeps_later_port_on_its_own_template(self):
        asymmetric = FakeOp((("left",), ("right",)))
        labels, batch = bind_labels(asymmetric, [None, (7,)])

        self.assertEqual(labels, {"right": 7})
        self.assertEqual(batch, 1)

    def test_strict_broadcast_uses_the_non_unit_label_extent(self):
        operands = ((1, 128, 2048), (1, 128, 1))
        for shapes in (operands, tuple(reversed(operands))):
            with self.subTest(shapes=shapes):
                labels, _ = bind_labels(
                    OP_REGISTRY["mul"],
                    shapes,
                    require_consistent_labels=True,
                )
                self.assertEqual(labels["dim"], 2048)

        labels, _ = bind_labels(
            OP_REGISTRY["mul"],
            [(1, 128, 1), (1, 128, 1)],
            require_consistent_labels=True,
        )
        self.assertEqual(labels["dim"], 1)

    def test_strict_non_broadcast_rejects_a_shared_label_conflict(self):
        with self.assertRaisesRegex(ValueError, "d_head.*conflicting"):
            bind_labels(
                ATTN_SCORE,
                [(2, 64, 128, 96), (2, 64, 512, 128)],
                require_consistent_labels=True,
            )


class TestBatchFold(unittest.TestCase):
    """The rope case: one rank-3 template, tensors of any rank."""

    def test_uncovered_leading_axes_multiply(self):
        labels, batch = bind_labels(ELEMENTWISE, [(2, 48, 1024, 64)])
        self.assertEqual(labels, {"B": 48, "s_ctx": 1024, "dim": 64},
                         "trailing-aligned: the formula is about the feature end")
        self.assertEqual(batch, 2)

    def test_the_product_is_the_whole_tensor(self):
        """Whatever the labels are NAMED, coeff * B * s_ctx * dim * batch must
        come to the element count — that is what makes the head axis count."""
        shape = (2, 48, 1024, 64)
        derived, batch = shape_derived_dims(ELEMENTWISE, {"B": 2, "s_ctx": 1024,
                                                          "dim": 64}, [shape])
        space = derived["B"] * derived["s_ctx"] * derived["dim"] * batch
        self.assertEqual(space, 2 * 48 * 1024 * 64)

    def test_a_rotation_is_three_flops_per_rotated_element(self):
        """x' = x·cos − y·sin, y' = y·cos + x·sin: 4 multiplies and 2 adds per
        rotated pair (vLLM csrc/libtorch_stable/pos_encoding_kernels.cu:31-32),
        3 per element, for every rotation op — one position stream or three,
        forward or inverse."""
        from backend.arch.ops import OP_REGISTRY
        env = {"B": 2, "s_ctx": 16, "dim": 64, "n_heads": 1}
        for name in ("rope", "rope_emb", "rope_emb_inverse", "mrope"):
            self.assertEqual(OP_REGISTRY[name].flops_fn(env), 3 * 2 * 16 * 64, name)

    def test_a_hadamard_pays_butterfly_adds_and_the_scale(self):
        """A fast Walsh-Hadamard transform has log2(n) butterfly adds and one
        scaling multiply per output element."""
        import sympy as sp
        from backend.arch.ops import OP_REGISTRY
        op = OP_REGISTRY["hadamard_transform"]
        env = {"B": 2, "s_ctx": 16, "dim": 128}
        self.assertEqual(op.flops_fn(env), 2 * 16 * 128 * (7 + 1))
        self.assertEqual(op.params_fn(env), 0)
        n = sp.Symbol("n")
        self.assertEqual(
            op.flops_fn({"B": 1, "s_ctx": 1, "dim": n}),
            n * (sp.log(n, 2) + 1))

    def test_rope_now_counts_every_head(self):
        rope = lambda d: 3 * d["B"] * d["s_ctx"] * d["dim"]
        env = {"B": 2, "s_ctx": 1024, "dim": 64}
        self.assertEqual(rope(env), 3 * 2 * 1024 * 64)
        self.assertEqual(
            flops_from_shape(ELEMENTWISE, env, [(2, 48, 1024, 64)], rope),
            3 * 2 * 48 * 1024 * 64,
            "48 heads, not one",
        )


class TestTemplateThatDoesNotDescribeTheTensors(unittest.TestCase):
    def test_shorter_tensor_than_template_does_not_vote(self):
        """A per-channel vector broadcast against a 4-D activation. It cannot be
        aligned, so it contributes nothing — and must not veto the operand that
        CAN be aligned."""
        mul = FakeOp((("B", "s_ctx", "dim"), ("B", "s_ctx", "dim")))
        labels, batch = bind_labels(mul, [(2, 96, 1024, 128), (128,)])
        self.assertEqual(labels, {"B": 96, "s_ctx": 1024, "dim": 128})
        self.assertEqual(batch, 2)

    def test_inconsistent_remainders_derive_nothing(self):
        """Rank-4 inputs against rank-4 templates (remainder 0) beside one
        whose template is rank-3 while the tensor is rank-4 (remainder 1). One
        input's leading axis is not everyone's batch; folding it in would
        triple the op."""
        gdr = FakeOp((("B", "s_ctx", "n_heads", "d_head"),
                      ("B", "s_ctx", "n_heads")))
        labels, batch = bind_labels(gdr, [(2, 96, 1024, 128), (2, 96, 1024, 128)])
        self.assertEqual(labels, {}, "the declaration is not evidence here")
        self.assertEqual(batch, 1)

    def test_no_alignable_input_leaves_the_environment_alone(self):
        env = {"B": 2, "s_ctx": 1024, "dim": 64}
        derived, batch = shape_derived_dims(ELEMENTWISE, env, [(64,)])
        self.assertEqual(derived, env)
        self.assertEqual(batch, 1)


class TestTheEnvironmentStillAnswersForWhatNoTensorCarries(unittest.TestCase):
    def test_output_width_keeps_the_environment_value(self):
        """`out` is a property of the weight, not of the activation. Nothing
        upstream can answer it and nothing should try."""
        linear = lambda d: 2 * d["B"] * d["s_ctx"] * d["in"] * d["out"]
        got = flops_from_shape(LINEAR, {"B": 1, "s_ctx": 1, "in": 1, "out": 512},
                               [(2, 128, 4096)], linear)
        self.assertEqual(got, 2 * 2 * 128 * 4096 * 512)

    def test_literal_axes_are_not_labels(self):
        """The dispatch-zone template `("M", "1", "d_model")` — `1` is a shape
        constant, and binding it would create a dim named `1`."""
        banked = FakeOp(("M", "1", "d_model"))
        labels, _ = bind_labels(banked, [(132, 1, 2048)])
        self.assertEqual(labels, {"M": 132, "d_model": 2048})

    def test_zero_cost_stays_zero(self):
        zero = lambda d: 0
        self.assertEqual(
            flops_from_shape(ELEMENTWISE, {}, [(2, 48, 1024, 64)], zero), 0)

    def test_a_template_that_binds_nothing_is_loud(self):
        """Voters with disagreeing remainders used to fall back to Binding.
        A shorter tensor than its template is a non-voter, not a raise."""
        gdr = FakeOp((("B", "s_ctx", "n_heads", "d_head"),
                      ("B", "s_ctx", "n_heads")))
        env = {"B": 2, "s_ctx": 1024, "n_heads": 96, "d_head": 128}
        with self.assertRaisesRegex(ValueError, "does not describe"):
            flops_from_shape(
                gdr, env, [(2, 96, 1024, 128), (2, 96, 1024, 128)],
                lambda d: d["B"] * d["s_ctx"])
        linear = lambda d: 2 * d["B"] * d["s_ctx"] * d["in"] * d["out"]
        self.assertEqual(
            flops_from_shape(
                LINEAR, {"B": 1, "s_ctx": 128, "in": 8, "out": 16},
                [(8,)], linear),
            2 * 1 * 128 * 8 * 16)
        zero = lambda d: 0
        self.assertEqual(
            flops_from_shape(FakeOp(), {}, [(2, 48, 1024, 64)], zero), 0)

    def test_no_template_is_a_no_op(self):
        src = FakeOp(())
        f = lambda d: d.get("k", 7)
        self.assertEqual(flops_from_shape(src, {"k": 9}, [], f), 9)


class TestSubstitutionRecord(unittest.TestCase):
    """The record is the substitution the evaluation used — no more."""

    def test_record_holds_only_what_the_formula_read(self):
        """A dim sitting unread in the environment — a bare `n_heads` whose
        global value is stale under TP — must not enter the record."""
        B, s_ctx, in_, n_heads, out = sp.symbols("B s_ctx in n_heads out")
        env = {
            "B": Dim(B, 2),
            "s_ctx": Dim(s_ctx, 128),
            "in": Dim(in_, 4096),
            "out": Dim(out, 512),
            "n_heads": Dim(n_heads, 128),       # never read: the stale global
        }
        record = {}
        linear = lambda d: 2 * d["B"] * d["s_ctx"] * d["in"] * d["out"]
        flops_from_shape(LINEAR, env,
                         [(Dim(B, 2), Dim(s_ctx, 128), Dim(in_, 4096))],
                         linear, record=record)
        self.assertEqual(record, {B: 2, s_ctx: 128, in_: 4096, out: 512})

    def test_env_guard_fires_only_on_env_tier_reads(self):
        """Labels come off the arriving tensors (the shape tier); the guard
        exists for the tier where a stale global binding can pass for a
        per-rank value."""
        out = sp.Symbol("out")
        env = {"B": 1, "s_ctx": 1, "in": 1, "out": Dim(out, 512)}
        seen = []
        linear = lambda d: 2 * d["B"] * d["s_ctx"] * d["in"] * d["out"]
        flops_from_shape(LINEAR, env, [(2, 128, 4096)], linear,
                         record={}, env_guard=lambda n, e: seen.append(n))
        self.assertEqual(seen, ["out"])

    def test_env_guard_refusal_propagates(self):
        def guard(name, extent):
            raise ValueError("no proven per-rank extent")
        out = sp.Symbol("out")
        env = {"B": 1, "s_ctx": 1, "in": 1, "out": Dim(out, 512)}
        linear = lambda d: 2 * d["B"] * d["s_ctx"] * d["in"] * d["out"]
        with self.assertRaisesRegex(ValueError, "no proven per-rank extent"):
            flops_from_shape(LINEAR, env, [(2, 128, 4096)], linear,
                             record={}, env_guard=guard)


class TestReadTracking(unittest.TestCase):
    def test_only_subscripting_is_a_read(self):
        inner = {"a": 1, "b": 2}
        tracked = ReadTracking(inner)
        self.assertEqual(len(tracked), 2)
        self.assertIn("a", tracked)
        self.assertEqual(sorted(tracked.keys()), ["a", "b"])
        self.assertEqual(tracked.read, set())
        _ = tracked["a"]
        self.assertEqual(tracked.read, {"a"})

    def test_get_on_a_missing_key_marks_it(self):
        """The subscript runs before the KeyError, so consumers of `read`
        must filter against the backing map."""
        tracked = ReadTracking({"a": 1})
        self.assertIsNone(tracked.get("missing"))
        self.assertEqual(tracked.read, {"missing"})

    def test_eval_dim_expr_reads_only_free_symbols(self):
        d_model, n_heads = sp.symbols("d_model n_heads")
        # An unrelated tracked dim no longer donates its identity.
        got = eval_dim_expr(
            "d_model", {"d_model": 4096, "n_heads": Dim(n_heads, 8)})
        self.assertEqual(got, 4096)
        self.assertNotIsInstance(got, Dim)
        # A read one still does.
        got = eval_dim_expr(
            "d_model // n_heads",
            {"d_model": Dim(d_model, 4096), "n_heads": Dim(n_heads, 8)})
        self.assertIsInstance(got, Dim)
        self.assertEqual(int(got), 512)


class TestPositionTableAdd(unittest.TestCase):
    def test_reference_shape_drives_output_and_add_cost(self):
        op = OP_REGISTRY["position_table_add"]
        reference = (2, 17, 64)
        dims = {"B": 1, "s_ctx": 1, "dim": 64}

        for table in ((4, 64), (2, 2, 64)):
            with self.subTest(table=table):
                shapes = (reference, table)
                self.assertEqual(
                    compute_shape_out(op, shapes, dims), reference)
                self.assertEqual(
                    flops_from_shape(
                        op, dims, shapes,
                        op.flops_fn,
                    ),
                    2 * 17 * 64,
                )


class TestGatedDeltaRuleCountsTheRecurrentStep(unittest.TestCase):
    """The serving recurrent kernel (FLA `fused_recurrent.py:125-148`) per
    token and value head over its [d_v, d_k] state: decay (1 mul), readout
    (2), rank-1 write (2) and output (2) per state element, plus the query
    scale (d_k) and the value correction and beta (2·d_v)."""

    def test_seven_flops_per_state_element(self):
        d = {"B": 2, "s_ctx": 3, "n_lin_v_heads": 4,
             "d_lin_k_head": 128, "d_lin_v_head": 64}
        per_head_token = 7 * 128 * 64 + 128 + 2 * 64
        self.assertEqual(OP_REGISTRY["gated_delta_rule"].flops_fn(d),
                         2 * 3 * 4 * per_head_token)


if __name__ == "__main__":
    unittest.main()


class TestSeparableOpOnALowerRankTensor(unittest.TestCase):
    """An elementwise op costs the elements that arrive. `neg_exp` on a
    per-head `A_log` of 32 values is 32 applications, not B·s_ctx·32."""

    ACT = FakeOp(("B", "s_ctx", "dim"), shape_kind="preserve")
    MUL = FakeOp((("B", "s_ctx", "dim"), ("B", "s_ctx", "dim")), shape_kind="broadcast")

    def test_the_tensor_votes_for_the_axes_it_has(self):
        labels, batch = bind_labels(self.ACT, [(32,)])
        self.assertEqual(labels, {"dim": 32})
        self.assertEqual(batch, 1)

    def test_axes_the_tensor_lacks_count_once_not_the_environment(self):
        from backend.arch.eval.flops import flops_from_shape
        cost = flops_from_shape(self.ACT, {"B": 2, "s_ctx": 1024, "dim": 32}, [(32,)],
                                lambda d: 3 * d["B"] * d["s_ctx"] * d["dim"])
        self.assertEqual(int(cost), 3 * 32)

    def test_a_broadcast_operand_leaves_the_batch_to_the_full_tensor(self):
        """KDA's decay: a [B, H, S, D] operand times a [D] vector."""
        labels, batch = bind_labels(self.MUL, [(2, 96, 1024, 128), (128,)])
        self.assertEqual(labels, {"B": 96, "s_ctx": 1024, "dim": 128})
        self.assertEqual(batch, 2)

    def test_a_coupled_op_does_not_align_a_short_tensor(self):
        norm = FakeOp(("B", "s_ctx", "dim"), shape_kind="preserve", couples_on=("dim",))
        labels, _ = bind_labels(norm, [(32,)])
        self.assertEqual(labels, {})

