"""A per-op row is a collective because its op says so.

`compute_per_op_timing` used to accept a second witness — a `comm_world`
key in the node's comm metadata — beside `OpSpec.is_comm_op`. The metadata
is written for the collective nodes layout emits and for nothing else, so
the second witness never fired alone; this pins that, so the row's
`is_comm` (which the frontend now reads instead of its own op-name list)
rests on the declaration.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.layout.materializer import materialize_comm_tree
from tests.snapshots import shared_loader
from backend.arch.ops import OP_REGISTRY
from backend.inference.parallel_config import ParallelConfig
from backend.inference.step_eval import evaluate_step

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"


class TestCommMetaOnlyOnCollectives(unittest.TestCase):
    def test_every_comm_meta_row_is_a_declared_collective(self):
        config = json.loads((SNAPSHOTS / "openai__gpt-oss-20b.json").read_text())["config"]
        step = evaluate_step(
            shared_loader().match(config, model_id="openai/gpt-oss-20b"),
            ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8),
            phase="prefill", batch=1, input_tokens=4096, output_tokens=0, s_prefix_cached=3968)
        tree = materialize_comm_tree(step.analysis)
        meta = tree.comm_meta.values
        self.assertTrue(meta, "TP8 gpt-oss emits collectives")
        ops = {tree.program.by_uid(uid).op for uid in meta}
        self.assertTrue(all(OP_REGISTRY[o].is_comm_op for o in ops), ops)


if __name__ == "__main__":
    unittest.main()
