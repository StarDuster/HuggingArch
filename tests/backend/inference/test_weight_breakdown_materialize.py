"""Lock the invariant: `per_rank_weight_breakdown` sources per-leaf
storage bytes from `materialize_for_resolved_spec` (CLAUDE.md §2.3 —
single quant fusion entry).

The pre-refactor version computed `params × uniform_weight_dtype_bytes`
plus a synthetic `_quant_overhead_per_elem` scale overhead. Mixed
precision (MiMo GQA/SWA/MoE-expert at different dtypes; DSv3 FP8
attention + BF16 norms; gpt-oss MXFP4 fused) silently miscounted bytes.

The tests below lock:
  1. Uniform-BF16 models (Llama-3.2-3B) match the old "params × 2 B"
     total within 0.1% — no regression on the dominant case.
  2. Mixed-precision deployment_total exactly equals materialize's
     parameter + metadata rows; resident buffers are not weights.
  3. Quantized models surface per-leaf dtype labels through
     `TreeNode.weight_dtype` (FP8 attention rows show "F8_E4M3",
     BF16 norm rows show "BF16" — not one global label).
  4. The synthetic `quant_overhead_per_elem` sentinel is dead (= 0.0).
  5. Scale bytes come from real metadata rows, not synthesized.
  6. gpt-oss fused custom-op rows show up in the deployment total via
     residual without double-counting constituent leaves.
"""
from __future__ import annotations

import sys
import unittest

from tests.snapshots import resolve_snapshot
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.loader import ResolvedSpec
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.schema import (
    ArchSpec,
    BlockGraph,
    GraphNode,
    ModelLayer,
    StreamSpec,
)
from backend.inference.parallel_config import ParallelConfig
from backend.inference.weight_breakdown import (
    per_rank_weight_breakdown,
)


def _analysis(resolved):
    return Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=8,
    ).analyze()



def _resolved(model_name):
    return resolve_snapshot(model_name)


def _walk_leaves(node: dict, out: list) -> None:
    if not node.get("is_component", False) and not node.get("children"):
        out.append(node)
        return
    for c in node.get("children") or []:
        _walk_leaves(c, out)


def _structural_layer_fixture():
    weightless = BlockGraph(
        inputs=["x"],
        graph={
            "act": GraphNode(op="silu", inputs=["x"], dim="d_model"),
        },
        outputs=["act"],
    )
    weighted_singleton = BlockGraph(
        inputs=["x"],
        graph={
            "norm": GraphNode(
                op="rmsnorm",
                inputs=["x"],
                dim="d_model",
                weight_axes=[{"replicated": True}],
            ),
        },
        outputs=["norm"],
    )
    spec = ArchSpec(
        name="structural-layer-count",
        dims={"d_model": "config.hidden_size", "n_layers": "3"},
        model_inputs={"input_ids": {"kind": "tokens"}},
        model=[
            ModelLayer(
                name="embed",
                op="embedding",
                inputs=["input_ids"],
                boundary_role="input_embedding",
                dims={"n_vocab": "d_model", "d_model": "d_model"},
                weight_axes=[{"replicated": True}, {"replicated": True}],
            ),
            ModelLayer(
                name="layers",
                template="weightless",
                repeat="n_layers",
                inputs=["embed"],
                streams={"hidden": StreamSpec(carries=True)},
            ),
            ModelLayer(
                name="head", template="weighted_singleton", inputs=["layers"],
            ),
        ],
        blocks={
            "weightless": weightless,
            "weighted_singleton": weighted_singleton,
        },
    )
    return ResolvedSpec(spec, {"d_model": 16, "n_layers": 3},
                        {"hidden_size": 16, "torch_dtype": "bfloat16"}, {})


class TestMaterializeIsSourceOfTruth(unittest.TestCase):
    def test_per_rank_total_equals_resident_artifact_rows(self):
        """The reporting tree is a grouping of physical artifact bytes.

        This catches duplicate template invocations and any adapter-side
        divisor that makes the tree disagree with its own Analysis.
        """
        from backend.arch.layout import persistent
        from backend.arch.layout.persistent import physical_artifacts
        from backend.inference.step_eval import evaluate_step

        models = (
            "MiniMaxAI__MiniMax-M2.7",
            "Qwen__Qwen2.5-7B",
            "XiaomiMiMo__MiMo-V2.5",
            "google__gemma-4-12B-it",
            "inclusionAI__Ling-2.6-1T",
            "meituan-longcat__LongCat-Flash-Thinking-2601",
            "moonshotai__Kimi-K2.5",
            "openai__gpt-oss-20b",
            "zai-org__GLM-5.1",
            "zai-org__GLM-5.2",
        )
        deployments = (
            ParallelConfig(),
            ParallelConfig(
                tp_attention=8,
                tp_dense_ffn=8,
                tp_moe_expert=8,
            ),
        )
        for name in models:
            resolved = _resolved(name)
            if resolved is None:
                continue
            for parallel in deployments:
                with self.subTest(name=name, parallel=repr(parallel)):
                    analysis = evaluate_step(
                        resolved,
                        parallel,
                        phase="prefill",
                        batch=1,
                        input_tokens=128, output_tokens=0,
                    ).analysis
                    artifacts = physical_artifacts(analysis)
                    included = analysis.program.included_stack_names()
                    expected = sum(
                        row.resident_bytes_per_rank
                        for row in artifacts
                        if row.stack in included
                        and (
                            (
                                row.surface == persistent.STORED
                                and row.is_parameter
                            )
                            or row.surface
                            == persistent.RESIDENT_METADATA
                        )
                    )
                    got = per_rank_weight_breakdown(
                        resolved,
                        parallel,
                        analysis=analysis,
                        artifacts=artifacts,
                    )["total_bytes_per_rank"]
                    self.assertEqual(got, expected)

    def test_num_layers_is_structural_not_a_weight_presence_proxy(self):
        wb = per_rank_weight_breakdown(
            _structural_layer_fixture(),
            ParallelConfig(),
            )

        self.assertEqual(wb["num_layers"], 3)
        self.assertEqual(
            [group["name"] for group in wb["groups"]],
            ["weighted_singleton × 1  [head]"],
        )

    def test_deployment_total_equals_materialize_sum(self):
        """For any model, `deployment_total_bytes` must equal the sum of
        `storage_bytes × multiplicity` across every materialized weight row
        (parameters + metadata + custom-op, excluding buffers). Load-bearing invariant —
        any drift means weight_breakdown and matcher/validator/UI badge
        are looking at different facts.
        """
        for name in ("meta-llama__Llama-3.2-3B",
                     "XiaomiMiMo__MiMo-V2.5",
                     "deepseek-ai__DeepSeek-V3",
                     "zai-org__GLM-5.1",
                     "zai-org__GLM-5.2",
                     "moonshotai__Kimi-K2-Instruct",
                     "openai__gpt-oss-20b"):
            resolved = _resolved(name)
            if resolved is None:
                continue
            analysis = _analysis(resolved)
            mat_total = sum(
                float(row.storage_bytes) * int(row.count)
                for row in analysis.iter_checkpoint() if not row.is_buffer
            )
            # materialize emits every tensor the language model's walk
            # holds — backbone + aux (MTP); an encoder is its own phase's
            # walk and in neither. The faithful-summation invariant is only
            # apples-to-apples against the full breakdown, so include aux
            # weights here. (`include_aux_weights=False`, the serving
            # default, deliberately drops aux groups for the backbone-only
            # HBM footprint — a subset, not a summation-faithfulness claim.)
            wb = per_rank_weight_breakdown(
                resolved, ParallelConfig(), include_aux_weights=True)
            self.assertAlmostEqual(
                wb["deployment_total_bytes"], mat_total,
                delta=max(1.0, mat_total * 1e-9),
                msg=f"{name}: deployment_total={wb['deployment_total_bytes']:.0f} "
                    f"vs materialize_sum={mat_total:.0f}",
            )

    def test_quant_overhead_per_elem_is_dead_sentinel(self):
        """Synthetic per-element scale overhead is gone — scales are now
        sourced per-leaf from real metadata rows."""
        for name in ("meta-llama__Llama-3.2-3B",
                     "deepseek-ai__DeepSeek-V3",
                     "moonshotai__Kimi-K2-Instruct",
                     "openai__gpt-oss-20b"):
            resolved = _resolved(name)
            if resolved is None:
                continue
            wb = per_rank_weight_breakdown(
                resolved, ParallelConfig(), quant_granularity="per-group",
            )
            self.assertEqual(
                wb["quant_overhead_per_elem"], 0.0,
                msg=f"{name}: synthetic overhead still set",
            )


class TestPerLeafDtypeFlows(unittest.TestCase):
    def test_mixed_precision_surfaces_multiple_dtype_labels(self):
        """DSv3 (FP8 attention + BF16 norms) must show >1 distinct
        per-leaf `weight_dtype` label in the tree — proves the per-leaf
        dtype flowed through, replacing the old uniform global label.
        """
        resolved = _resolved("deepseek-ai__DeepSeek-V3")
        wb = per_rank_weight_breakdown(
            resolved, ParallelConfig())
        leaves: list = []
        for g in wb["groups"]:
            _walk_leaves(g, leaves)
        for m in wb["model_level"]:
            _walk_leaves(m, leaves)
        labels = {leaf["weight_dtype"] for leaf in leaves if leaf["bytes_full"] > 0}
        self.assertGreater(
            len(labels), 1,
            msg=f"DSv3 should expose mixed dtype labels; got {labels}",
        )

    def test_uniform_bf16_single_label(self):
        """Llama-3.2-3B at BF16: every leaf carries a single dtype label
        (BF16 / F16 — the materialized storage_dtype, normalized)."""
        resolved = _resolved("meta-llama__Llama-3.2-3B")
        wb = per_rank_weight_breakdown(
            resolved, ParallelConfig())
        leaves: list = []
        for g in wb["groups"]:
            _walk_leaves(g, leaves)
        labels = {leaf["weight_dtype"] for leaf in leaves if leaf["bytes_full"] > 0}
        # Uniform-precision BF16 model — exactly one label across all
        # weight-bearing leaves.
        self.assertEqual(
            len(labels), 1,
            msg=f"Llama-3.2-3B should expose a single dtype; got {labels}",
        )


class TestScaleBytesAreReal(unittest.TestCase):
    def test_quantized_model_has_nonzero_scale_bytes(self):
        """A quantized snapshot must surface non-zero `scale_bytes_full`
        somewhere in the tree — proves scales were attributed to parent
        leaves, not silently dropped."""
        for name in ("deepseek-ai__DeepSeek-V3",
                     "moonshotai__Kimi-K2-Instruct"):
            resolved = _resolved(name)
            if resolved is None:
                continue
            wb = per_rank_weight_breakdown(
                resolved, ParallelConfig())
            scale_total = 0.0
            for g in wb["groups"]:
                scale_total += g.get("scale_bytes_full", 0.0)
            for m in wb["model_level"]:
                scale_total += m.get("scale_bytes_full", 0.0)
            self.assertGreater(
                scale_total, 0.0,
                msg=f"{name}: tree scale_bytes_full sum is zero; "
                    f"metadata rows not attributed",
            )


class TestLeafFormulaIsArithmetic(unittest.TestCase):
    def test_values_line_computes_the_leaf_and_names_its_scales(self):
        """A weight leaf's values line is the leaf law at its dims — every
        factor a dim, the format's scales as grid × width — and, times its
        copies, computes `bytes_full`. Never a folded product or a lump."""
        resolved = _resolved("deepseek-ai__DeepSeek-V3")
        wb = per_rank_weight_breakdown(
            resolved, ParallelConfig(), analysis=_analysis(resolved))
        leaves: list = []
        for node in (*wb["groups"], *wb["model_level"]):
            _walk_leaves(node, leaves)
        by_name = {}
        for leaf in leaves:
            got = eval(leaf["leaf_substituted"], {"__builtins__": {}})
            self.assertAlmostEqual(got * leaf["leaf_cnt"], leaf["bytes_full"],
                                   delta=1e-6 * leaf["bytes_full"],
                                   msg=leaf["name"])
            self.assertNotIn(" + meta", leaf["leaf_symbolic"])
            by_name.setdefault(leaf["name"], leaf)
        q_b = by_name["q_b_proj"]
        self.assertEqual(
            q_b["leaf_symbolic"],
            "(d_q_lora*n_heads*(d_qk_nope + d_qk_rope))*dt_w"
            " + scale_grid*scale_bytes")
        self.assertEqual(q_b["leaf_substituted"],
                         "(1536*128*(128 + 64))*1 + 192*12*4")


class TestCustomOpResidual(unittest.TestCase):
    def test_gpt_oss_fused_op_residual_closes_total(self):
        """gpt-oss-20b has `mxfp4_fused_linear` custom-op rows whose
        constituent leaves get rewritten away. The deployment total
        must still match materialize's sum — implies the fused
        residual is attributed correctly."""
        resolved = _resolved("openai__gpt-oss-20b")
        analysis = _analysis(resolved)
        fused_total = sum(
            float(row.storage_bytes) * int(row.count)
            for row in analysis.iter_checkpoint()
            if getattr(analysis.program.by_uid(row.uid), "from_custom_op", False)
        )
        self.assertGreater(
            fused_total, 0.0,
            msg="gpt-oss-20b should have non-zero fused custom-op bytes",
        )
        wb = per_rank_weight_breakdown(
            resolved, ParallelConfig())
        mat_total = sum(
            float(row.storage_bytes) * int(row.count)
            for row in analysis.iter_checkpoint()
        )
        self.assertAlmostEqual(
            wb["deployment_total_bytes"], mat_total,
            delta=max(1.0, mat_total * 1e-9),
            msg=f"deployment_total={wb['deployment_total_bytes']:.0f} "
                f"vs materialize_sum={mat_total:.0f} "
                f"(fused-op residual missing or double-counted)",
        )


class TestMissingJoin(unittest.TestCase):
    """A leaf whose node stores no row used to count as 0 bytes."""

    def _leaf(self, **kw):
        from backend.arch.spec_tree import LeafTensor
        fields = dict(
            tensor_name="foo.weight", op="linear", shape=(4, 4),
            multiplicity=1, role="ffn",
            template="",
        )
        fields.update(kw)
        return LeafTensor(**fields)

    def test_ordinary_leaf_without_checkpoint_row_raises(self):
        from backend.inference.weight_breakdown import _resolve_leaf_bytes
        with self.assertRaisesRegex(ValueError, "no checkpoint row"):
            _resolve_leaf_bytes(self._leaf(), {}, 2.0, "BF16")

    def test_fused_custom_op_without_checkpoint_row_is_zero(self):
        from backend.inference.weight_breakdown import _resolve_leaf_bytes
        data, scale, _label, _bpe, params = _resolve_leaf_bytes(
            self._leaf(from_custom_op=True), {}, 2.0, "BF16")
        self.assertEqual((data, scale), (0.0, 0.0))
        # No row, so no bytes — but the leaf still declares a shape, and
        # that shape times its copies is the parameter count the tree shows.
        self.assertGreater(params, 0)


class TestLeafBytesFormulaIsTheParamsIdentity(unittest.TestCase):
    def test_kv_a_proj_is_not_bytes_full(self):
        """A replicated MLA down-proj's Full hover is params * dt_w,
        products kept — never the `bytes_full` placeholder."""
        import sympy as sp
        resolved = _resolved("deepseek-ai__DeepSeek-V3")
        wb = per_rank_weight_breakdown(
            resolved, ParallelConfig())
        leaves: list = []
        for g in wb["groups"]:
            _walk_leaves(g, leaves)
        kv_a = next(
            (n for n in leaves if "kv_a_proj" in (n.get("name") or "")),
            None,
        )
        self.assertIsNotNone(kv_a)
        sym = kv_a.get("leaf_symbolic") or kv_a.get("shard_symbolic") or ""
        sub = kv_a.get("leaf_substituted") or kv_a.get("shard_substituted") or ""
        self.assertNotEqual(sym, "bytes_full")
        self.assertIn("dt_w", sym)
        self.assertIn("*", sub)
        self.assertNotRegex(sub, r"^[0-9.]+$")
        # The whole values line is arithmetic now — the scales too.
        self.assertGreater(int(sp.sympify(sub)), 1)

    def test_a_routed_experts_copies_are_its_layers_and_its_shape_the_experts(self):
        """A routed expert bank is repeated by its layers only; its experts
        are the leading axis of its weight's shape, never one
        '14848 layers' blob."""
        resolved = _resolved("deepseek-ai__DeepSeek-V3")
        wb = per_rank_weight_breakdown(
            resolved, ParallelConfig())
        leaves: list = []
        for g in wb["groups"]:
            _walk_leaves(g, leaves)
        gate = next(
            (n for n in leaves
             if n.get("name") == "gate_proj"
             and n.get("class_") == "moe_routed"),
            None,
        )
        self.assertIsNotNone(gate)
        factors = gate.get("leaf_copy_factors") or []
        self.assertEqual([(f["unit"], int(f["n"])) for f in factors],
                         [("layers", int(gate["leaf_cnt"]))])
        self.assertEqual(gate["weight_shape"], "[256, 2048, 7168]")


if __name__ == "__main__":
    unittest.main()
