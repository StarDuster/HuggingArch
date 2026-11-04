"""Megatron sequence parallelism (TP-SP) — cost/shape modeling.

SP shards the activations of the SP region (the block-level norm / residual-add
between TP components) on the sequence axis by `tp`, cutting activation memory,
while the TP regions stay full-S + head-sharded. The layout rule records the
cut as its `residual_sequence_family` (the attention TP family when SP is on,
none when off); the physical shape layer carries TP/CP but not SP, so it shows
full-S everywhere. This verifies the rule's residual sequence family and that
the per-rank shapes stay full-S.
"""
from __future__ import annotations

import json
import sys
import unittest

from tests.snapshots import shared_loader, snapshot_config
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from backend.arch.layout.propagation import LayoutPropagation

SNAP = PROJECT_ROOT / "backend/arch/snapshots/meta-llama__Llama-3.2-3B.json"


def _analyze(sp: bool):
    resolved = shared_loader().match(json.loads(SNAP.read_text())["config"],
                                     model_id=SNAP.stem.replace("__", "/", 1))
    pc = ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8,
                        sequence_parallel=sp)
    return analyze(resolved_spec=resolved, parallel=pc,
                   batch_size=1, avg_input_tokens=4096, avg_output_tokens=0, phase="prefill", breakdown=False)


class TestSequenceParallel(unittest.TestCase):
    def setUp(self):
        snapshot_config(SNAP.name)

    # The per-rank physical shape layer carries TP/CP but not SP, so the
    # SP-region norm's shape stays full-S. The sequence cut is pinned on the
    # layout rule in `test_sp_cuts_residual_sequence_on_attention_tp`.

    def test_tp_region_projection_not_sp_sharded(self):
        r = _analyze(sp=True)
        qp = [
            r.pre_shapes.values[node.uid]
            for node in r.pre_program.nodes()
            if node.aggregation_id.endswith("self_attn.q_proj")
        ]
        self.assertTrue(qp)
        # full S (4096), head-sharded feature (3072/8=384) — NOT S-sharded.
        self.assertEqual(int(qp[0].out[1]), 4096)

    def test_sp_cuts_residual_sequence_on_attention_tp(self):
        pc = ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8,
                            sequence_parallel=True)
        rule = LayoutPropagation.of(pc, "prefill")
        family = rule.residual_sequence_family
        self.assertIsNotNone(family)
        self.assertEqual(family, rule.families["tp_attention"])
        self.assertEqual(family.group_size, 8)

    def test_sp_off_unchanged(self):
        # sequence_parallel=False: no residual sequence cut, norms full-S.
        r = _analyze(sp=False)
        pc = ParallelConfig(tp_attention=8, sequence_parallel=False)
        self.assertIsNone(
            LayoutPropagation.of(pc, "prefill").residual_sequence_family)
        norm = next(node for node in r.pre_program.nodes()
                    if node.aggregation_id.endswith("input_layernorm"))
        shp = r.pre_shapes.values[norm.uid]
        self.assertEqual(int(shp.out[1]), 4096)


if __name__ == "__main__":
    unittest.main()
