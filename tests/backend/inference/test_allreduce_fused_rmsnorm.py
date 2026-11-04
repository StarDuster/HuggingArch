"""Fused all-reduce + RMSNorm plan: registration, TP matching, input order.

Covers:
  * `allreduce_fused_rmsnorm` plan is registered and bound to both
    vllm and sglang.
  * Pipeline integration: at TP=1 only `fused_add_rmsnorm` fires
    (the 2-node pattern). At TP>1 the 3-node `allreduce_fused_rmsnorm`
    fires and consumes the chain — `fused_add_rmsnorm` is shut out
    (its naive subgraph no longer matches because an all_reduce node
    sits between the row-parallel linear and the residual add).
  * The plan's input-position order matches the spec convention
    (`add(residual, sub_output)`) — wrong order makes the matcher
    silently miss every fire.
"""
from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import shared_loader
from backend.inference.framework_bindings import get_binding
from backend.inference.fusion_plans import FUSION_SCHEMES
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze


class TestAllReduceFusedRMSNormCatalog(unittest.TestCase):
    """Plan + binding presence locks in the C3 deliverable. Future
    edits that accidentally drop the plan or unbind it here would
    silently regress every TP>1 inference run."""

    def test_plan_registered(self):
        self.assertIn("allreduce_fused_rmsnorm", FUSION_SCHEMES)

    def test_plan_is_three_node_chain(self):
        pattern = FUSION_SCHEMES["allreduce_fused_rmsnorm"].naive_pattern
        nodes = pattern.expanded_nodes()
        self.assertEqual(set(nodes.keys()), {"allreduce", "add", "rmsnorm"})
        # Locking input ORDER: matcher is positional. Position 0 of
        # add must be `external.residual` (matches the spec
        # `add(residual, sub_output)` convention) — reversing this
        # silently makes the plan match zero chains.
        self.assertEqual(
            nodes["add"].inputs,
            ["external.residual", "allreduce.out"],
        )

    def test_bound_in_vllm_and_sglang(self):
        for framework in ("vllm", "sglang"):
            binding = get_binding(framework)
            self.assertIsNotNone(binding)
            impl = binding.get("allreduce_fused_rmsnorm")
            self.assertIsNotNone(
                impl,
                f"{framework} must bind allreduce_fused_rmsnorm",
            )
            self.assertTrue(impl.source_anchor)


class TestTPSwitchesBetweenNormFusions(unittest.TestCase):
    """End-to-end at the analyze() boundary: changing ParallelConfig
    flips which RMSNorm fusion fires, with no other code change.
    This is the headline C3 result — the comm-injection + fusion
    catalog cooperate to give a TP-correct fused leaf."""

    MODELS = (
        "meta-llama/Llama-2-7b-hf",
        "Qwen/Qwen3-30B-A3B",
        "deepseek-ai/DeepSeek-V3",
    )

    @classmethod
    def setUpClass(cls):
        from backend.analyzer import fetch_config
        cls.resolved_by_id = {}
        for mid in cls.MODELS:
            cfg = fetch_config(mid)
            cls.resolved_by_id[mid] = shared_loader().match(cfg, model_id=mid)
        cls.binding = get_binding("vllm")

    def _matches(self, resolved, parallel: ParallelConfig) -> Counter:
        result = analyze(
            resolved,
            parallel=parallel,
            framework_binding=self.binding,
            batch_size=1,
            avg_input_tokens=1, avg_output_tokens=0,
            breakdown=False,
        )
        return Counter(m["fused_op"] for m in result.fusion_matches)

    def test_tp1_fires_fused_add_rmsnorm_only(self):
        # Without comm injection, the 2-node chain matches; the
        # 3-node allreduce-fused variant has no all_reduce to anchor.
        for mid in self.MODELS:
            with self.subTest(model=mid):
                matches = self._matches(
                    self.resolved_by_id[mid], ParallelConfig(),
                )
                self.assertEqual(
                    matches.get("allreduce_fused_rmsnorm", 0), 0,
                    f"{mid}: at TP=1, no all_reduce nodes exist; "
                    f"allreduce_fused_rmsnorm cannot fire",
                )
                self.assertGreaterEqual(
                    matches.get("fused_add_rmsnorm", 0), 1,
                    f"{mid}: fused_add_rmsnorm must fire at TP=1",
                )

    def test_tp8_switches_to_allreduce_fused_rmsnorm(self):
        # With comm injection, the 3-node chain wins (higher
        # specificity) and consumes the same residual add + norm
        # boundary; fused_add_rmsnorm is shut out.
        pc = ParallelConfig(tp_attention=8, tp_dense_ffn=8)
        for mid in self.MODELS:
            with self.subTest(model=mid):
                matches = self._matches(self.resolved_by_id[mid], pc)
                self.assertGreaterEqual(
                    matches.get("allreduce_fused_rmsnorm", 0), 1,
                    f"{mid}: at TP=8 the 3-node chain "
                    f"allreduce → add → rmsnorm must fuse",
                )
                self.assertEqual(
                    matches.get("fused_add_rmsnorm", 0), 0,
                    f"{mid}: 2-node fused_add_rmsnorm should NOT fire "
                    f"at TP>1; the 3-node variant consumes the chain",
                )

    def test_llama_tp8_one_attention_one_ffn_pattern(self):
        # Llama has a clean 2-block (attn + ffn) template; the
        # all_reduce after o_proj is followed by add+rmsnorm
        # (post_attention_layernorm) and fuses cleanly. The
        # all_reduce after down_proj feeds the next-block's
        # input_layernorm, which crosses the template boundary —
        # The fusion matcher operates within one template, so
        # it stays as a bare all_reduce. Pin this asymmetry so a
        # future cross-template fusion attempt doesn't silently
        # change the count.
        pc = ParallelConfig(tp_attention=8, tp_dense_ffn=8)
        result = analyze(
            self.resolved_by_id["meta-llama/Llama-2-7b-hf"],
            parallel=pc,
            framework_binding=self.binding,
            batch_size=1,
            avg_input_tokens=1, avg_output_tokens=0,
            breakdown=False,
        )
        matches = Counter(m["fused_op"] for m in result.fusion_matches)
        # Exactly 1 fused (attention side) + 1 bare all_reduce
        # surviving (ffn side) per block template.
        self.assertEqual(matches.get("allreduce_fused_rmsnorm", 0), 1)
        # Count the block-level FFN-exit all_reduce only; the top-level
        # vocab-parallel embedding all_reduce is a separate model-level
        # collective, not part of the per-block fusion pattern this pins.
        bare_all_reduces = sum(
            1 for node in result.program.nodes()
            if not node.children
            and node.op == "all_reduce"
            and "vocab_" not in node.aggregation_id
        )
        self.assertEqual(bare_all_reduces, 1)

    def test_hybrid_model_fuses_every_block_not_just_one(self):
        # Regression for the cross-template spec_id collision: DSv3's
        # dense and MoE blocks are different templates that share the
        # same attention / norm sub-structure (`self_attn.attn_apply`,
        # `post_attention_layernorm`). A single global spec_id index
        # collapsed the duplicates (last-wins), so inject_comm + fusion
        # processed only ONE block — the other silently kept no
        # all_reduce and fused the TP=1 `fused_add_rmsnorm`. Both passes
        # now run per-block scope, so EVERY block fuses the TP-correct
        # `allreduce_fused_rmsnorm` and NO `fused_add_rmsnorm` survives.
        pc = ParallelConfig(tp_attention=8, tp_dense_ffn=8)
        result = analyze(
            self.resolved_by_id["deepseek-ai/DeepSeek-V3"],
            parallel=pc, framework_binding=self.binding,
            batch_size=1, avg_input_tokens=1, avg_output_tokens=0,
            breakdown=False,
        )

        def walk(ns):
            for n in ns:
                yield n
                yield from walk(n.children or ())

        # Every block's post_attention_layernorm fuses allreduce-variant.
        norm_markers = {}  # template_name -> marker
        for meta, roots in result.post_tree.blocks:
            for n in walk(roots):
                if n.display_path in ("post_attention_layernorm",
                                      "transformer.post_attention_layernorm"):
                    norm_markers[meta.template_name] = n.fused_from
        self.assertTrue(norm_markers)
        for tmpl, marker in norm_markers.items():
            self.assertEqual(
                marker, "allreduce_fused_rmsnorm",
                f"{tmpl}: norm should fuse the TP-allreduce variant, "
                f"got {marker!r} — a block was missed by the per-block pass",
            )
        # No block fell back to the TP=1 two-node fusion.
        all_markers = Counter(
            n.fused_from
            for _meta, roots in result.post_tree.blocks
            for n in walk(roots)
        )
        self.assertEqual(all_markers.get("fused_add_rmsnorm", 0), 0)


if __name__ == "__main__":
    unittest.main()
