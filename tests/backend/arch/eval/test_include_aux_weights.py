"""`include_aux_weights` toggle: the language model's aux modules (MTP) are
dropped from the weight breakdown by default so the weight footprint
matches the FLOPs path's text-only anchor (those blocks are also
filtered out of per_op_timing at pipeline.py). Enabling the toggle
reserves them in HBM. An encoder (vision / audio) is in neither position:
it is no part of the language model's step, and its weights are the encode
phase's own accounting.

The aux/backbone split is purely structural. Program owns the stack policy,
and both block groups and model-level singleton leaves are admitted through
the same set of included stack identities, with no per-model branching.

These tests pin both directions on MiMo-V2.5 (MTP aux beside a vision and
an audio encoder)
and lock byte-identical behavior for aux-free models (Kimi / Llama),
where the toggle must be a no-op.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import resolve_snapshot
from backend.arch.eval.capacity import weights
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.spec_tree import iter_blocks, iter_leaves
from backend.inference.parallel_config import ParallelConfig
from backend.inference.weight_breakdown import per_rank_weight_breakdown



def _resolve(snapshot_filename):
    return resolve_snapshot(snapshot_filename)


def _ledger_weights(resolved):
    program = Program.of(resolved)
    analysis = Deployment(program).at(batch=1, input_tokens=8).analyze()
    return program, weights(analysis)


class TestIncludeAuxWeightsHybrid(unittest.TestCase):
    """MiMo-V2.5: 48-layer backbone + vision tower + audio + 3 MTP layers,
    all tagged aux_kind. Default drops them; toggle restores them."""

    def setUp(self):
        self.resolved = _resolve("XiaomiMiMo__MiMo-V2.5.json")
        # A group card is named by its invocation (`BlockMeta.label`).
        self.aux_templates = {
            m.label for m in iter_blocks(self.resolved) if m.aux_kind
        }
        self.encoder_templates = {
            m.label for m in iter_blocks(self.resolved)
            if m.aux_kind in ("vision", "audio")
        }
        self.assertTrue(
            self.encoder_templates and self.aux_templates - self.encoder_templates,
            "MiMo-V2.5 should carry encoder groups (vision / audio) and MTP",
        )

    def test_default_excludes_aux_groups(self):
        wb = per_rank_weight_breakdown(
            self.resolved, ParallelConfig())
        group_templates = {g["template"] for g in wb["groups"]}
        leaked = group_templates & self.aux_templates
        self.assertEqual(
            leaked, set(),
            f"aux groups must be dropped by default; leaked: {leaked}",
        )

    def test_toggle_on_includes_aux_groups(self):
        wb = per_rank_weight_breakdown(
            self.resolved, ParallelConfig(), include_aux_weights=True)
        group_templates = {g["template"] for g in wb["groups"]}
        present = group_templates & self.aux_templates
        self.assertEqual(
            present, self.aux_templates - self.encoder_templates,
            f"the language model's aux groups, and no encoder, appear when "
            f"toggled on; got {present}",
        )

    def test_toggle_matches_the_structural_stack_ledger(self):
        """Both toggle positions match an independent per-stack ledger."""
        off = per_rank_weight_breakdown(
            self.resolved, ParallelConfig())
        on = per_rank_weight_breakdown(
            self.resolved, ParallelConfig(), include_aux_weights=True)
        program, ledger = _ledger_weights(self.resolved)
        self.assertAlmostEqual(
            off["deployment_total_bytes"], float(ledger.without_aux(program)),
            delta=1.0,
        )
        self.assertAlmostEqual(
            on["deployment_total_bytes"], float(ledger.resident), delta=1.0,
        )

    def test_backbone_layer_count_unchanged_by_toggle(self):
        """num_layers counts backbone only (aux never contributes), so
        it's invariant to the toggle — guards against aux leaking into
        the main-body layer count."""
        off = per_rank_weight_breakdown(
            self.resolved, ParallelConfig())
        on = per_rank_weight_breakdown(
            self.resolved, ParallelConfig(), include_aux_weights=True)
        self.assertEqual(off["num_layers"], on["num_layers"])


class TestIncludeAuxWeightsTopLevel(unittest.TestCase):
    """Aux singleton weights follow the same policy as repeated blocks."""

    def test_default_excludes_aux_model_level_leaves(self):
        snapshots = (
            "XiaomiMiMo__MiMo-V2.5.json",
            "google__gemma-4-12B-it.json",
            "meituan-longcat__LongCat-Flash-Thinking-2601.json",
        )
        for snapshot in snapshots:
            with self.subTest(snapshot=snapshot):
                resolved = _resolve(snapshot)
                program = Program.of(resolved)
                aux_stacks = {
                    stack.name for stack in program.stacks if stack.is_aux
                }
                encoders = {
                    stack.name for stack in program.stacks if stack.is_encoder
                }
                aux_ids = {
                    lt.display_path for lt in iter_leaves(resolved)
                    if not lt.template
                    and program.stack_of(lt.ir_node) in aux_stacks
                }
                encoder_ids = {
                    lt.display_path for lt in iter_leaves(resolved)
                    if not lt.template
                    and program.stack_of(lt.ir_node) in encoders
                }
                self.assertTrue(
                    aux_ids, "fixture must have model-level aux weights",
                )

                off = per_rank_weight_breakdown(
                    resolved, ParallelConfig())
                on = per_rank_weight_breakdown(
                    resolved, ParallelConfig(), include_aux_weights=True)
                # A model-level row is named by its leaf's display path. Its
                # `uid` is the breakdown's own analysis clone's, not this
                # tree's, so the two are joined by path.
                off_ids = {row["name"] for row in off["model_level"]}
                on_ids = {row["name"] for row in on["model_level"]}
                self.assertFalse(aux_ids & off_ids)
                self.assertEqual(aux_ids & on_ids, aux_ids - encoder_ids)


class TestIncludeAuxWeightsNoAuxIsNoOp(unittest.TestCase):
    """Aux-free models: the toggle must be byte-identical in both
    positions — no spurious filtering of backbone blocks."""

    def _check_noop(self, snapshot: str):
        resolved = _resolve(snapshot)
        # Confirm there's genuinely no aux to filter.
        self.assertEqual(
            [m.template_name for m in iter_blocks(resolved) if m.aux_kind],
            [],
            f"{snapshot} unexpectedly carries aux groups",
        )
        off = per_rank_weight_breakdown(
            resolved, ParallelConfig())
        on = per_rank_weight_breakdown(
            resolved, ParallelConfig(), include_aux_weights=True)
        self.assertEqual(
            off["deployment_total_bytes"], on["deployment_total_bytes"],
            f"{snapshot}: toggle changed total on an aux-free model",
        )
        self.assertEqual(
            off["total_bytes_per_rank"], on["total_bytes_per_rank"],
            f"{snapshot}: toggle changed per-rank on an aux-free model",
        )

    def test_kimi_k2_no_aux(self):
        self._check_noop("moonshotai__Kimi-K2-Instruct.json")

    def test_llama_no_aux(self):
        self._check_noop("meta-llama__Llama-3.2-3B.json")


class TestIncludeAuxWeightsRouteThreading(unittest.TestCase):
    """End-to-end through /inference/estimate: the request flag must
    reach the weight breakdown. Pins the wire contract so the field
    isn't silently dropped at the route / capacity boundary."""

    def _estimate(self, include_aux):
        from fastapi.testclient import TestClient
        from backend.api import app
        client = TestClient(app)
        body = {
            "model_id": "XiaomiMiMo/MiMo-V2.5",
            "gpu_sku": "H100-SXM",
            "avg_input_tokens": 4096,
            "avg_output_tokens": 256,
        }
        if include_aux is not None:
            body["include_aux_weights"] = include_aux
        resp = client.post("/inference/estimate", json=body)
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def test_route_default_drops_aux_toggle_restores(self):
        default = self._estimate(None)["weight_breakdown"]
        on = self._estimate(True)["weight_breakdown"]
        self.assertGreater(
            on["deployment_total_bytes"], default["deployment_total_bytes"],
            "include_aux_weights=true must add aux weight via the route",
        )
        # Default and explicit-false agree.
        off = self._estimate(False)["weight_breakdown"]
        self.assertEqual(
            default["deployment_total_bytes"], off["deployment_total_bytes"],
        )


if __name__ == "__main__":
    unittest.main()
