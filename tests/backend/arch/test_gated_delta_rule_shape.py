"""Gated DeltaNet linear-attention output shape.

The `gated_delta_rule` op's output is V-shaped [B, N_v, S, D_v]. Taking
the head count from Q is right only when N_q == N_v — false for
asymmetric linear attention (Qwen3-Next: N_lk=16 key/query heads,
N_lv=32 value heads). The DimMap copies V.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.axis_propagation import compute_shape_out
from backend.arch.ops import OP_REGISTRY


class TestGatedDeltaRuleShape(unittest.TestCase):
    def test_output_is_v_shaped_under_asymmetric_heads(self):
        op = OP_REGISTRY["gated_delta_rule"]
        # Layout [B, N, S, D]: q/k = 16 heads, v = 32 heads (Nlk < Nlv).
        shape_in = [
            [2, 16, 4, 128],   # q
            [2, 16, 4, 128],   # k
            [2, 32, 4, 128],   # v
            [32],              # g_gate
            [2, 4, 32],        # beta
        ]
        out = compute_shape_out(
            op, shape_in=shape_in, dims={"B": 2, "s_ctx": 4, "n_lin_v_heads": 32, "d_lin_v_head": 128},
            axis=None, axes=None,
        )
        # V-shaped: 32 value heads, NOT Q's 16.
        self.assertEqual(list(out), [2, 32, 4, 128])

    def test_inferred_shape_matches_its_own_template(self):
        # The op is `explicit`; inferred output must equal the shape_out template
        # ("B","n_lin_v_heads","s_ctx","d_lin_v_head") resolved against dims.
        op = OP_REGISTRY["gated_delta_rule"]
        self.assertEqual(op.shape_out, ("B", "n_lin_v_heads", "s_ctx", "d_lin_v_head"))
        shape_in = [[2, 16, 4, 128], [2, 16, 4, 128], [2, 32, 4, 128], [32], [2, 4, 32]]
        out = list(compute_shape_out(
            op, shape_in=shape_in, dims={"B": 2, "s_ctx": 4, "n_lin_v_heads": 32, "d_lin_v_head": 128},
            axis=None, axes=None))
        resolved_template = [2, 32, 4, 128]  # (B, Nlv, S, Dlv)
        self.assertEqual(out, resolved_template)


class TestGatedDeltaRuleLabels(unittest.TestCase):
    """V is value-wide and the (KDA) decay key-wide. With d_k == d_v, as in
    every shipped model, a template that swapped the two could not be told
    apart; with d_k != d_v the labels must still bind, each to its own width."""

    def test_key_and_value_widths_bind_separately(self):
        from backend.arch.eval.flops import bind_labels
        op = OP_REGISTRY["gated_delta_rule"]
        d_k, d_v = 64, 128
        shapes = [
            [2, 8, 4, d_k],    # q
            [2, 8, 4, d_k],    # k
            [2, 8, 4, d_v],    # v
            [2, 8, 4, d_k],    # decay, per key channel (KDA)
            [2, 4, 8],         # beta
        ]
        labels, _ = bind_labels(op, shapes)
        self.assertEqual(labels["d_lin_k_head"], d_k)
        self.assertEqual(labels["d_lin_v_head"], d_v)


class TestQueryAndKeyEnterTheRecurrenceL2Normalized:
    """Every delta-rule forward in the corpus passes
    `use_qk_l2norm_in_kernel=True`: q and k are divided by their L2 norm per
    key head before the recurrence (transformers `torch_chunk_gated_delta_rule`,
    FLA `chunk_gated_delta_rule` -> `l2norm_fwd`). The norm is an `l2norm` node
    on each of them; the d_k^-0.5 query scale stays inside the op."""

    @pytest.mark.parametrize("component", [
        "gated_delta_net_attention",
        "gated_delta_net_fused_attention",
        "kimi_k3_kda_attention",
    ])
    def test_q_and_k_are_l2norm_outputs(self, component):
        from backend.arch.loader import SpecLoader
        graph = SpecLoader().library_component(component).graph
        q, k = graph["delta_rule"].inputs[:2]
        for name, head in ((q, "q_split_heads"), (k, "k_split_heads")):
            node = graph[name]
            assert (node.op, node.inputs, node.dim) == (
                "l2norm", [head], "d_lin_k_head")

    def test_l2norm_is_priced_like_the_weightless_rms_norm(self):
        from sympy import Symbol
        dims = {k: Symbol(k) for k in ("B", "s_ctx", "dim")}
        l2, rms = OP_REGISTRY["l2norm"], OP_REGISTRY["rmsnorm_unscaled"]
        assert l2.flops_fn(dims) == rms.flops_fn(dims)
        assert l2.params_fn(dims) == 0


if __name__ == "__main__":
    unittest.main()
