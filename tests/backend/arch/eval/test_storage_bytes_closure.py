"""Weight bytes are the checkpoint's own account, block by block.

`Analysis.tensors()` prices every stored tensor at its own storage width.
The Layer Groups table used to take the model-wide total, divide it by the
parameter count, and multiply that average back per block — so on a
mixed-precision checkpoint every block was wrong by the same construction:
gpt-oss's MXFP4 expert blocks were priced too high and its BF16 attention
too low, summing to a total that looked right.

The facts pinned here are the ones that construction could not satisfy:
the blocks and the model-level nodes close on the checkpoint total, and no
block of a mixed-precision model sits at the average.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import shared_loader
from backend.arch.eval.program import Program
from backend.arch.eval.projection.result import project_analysis
from backend.arch.eval.transforms import Deployment

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _analysis(stem: str):
    config = json.loads((SNAPSHOTS / f"{stem}.json").read_text())["config"]
    resolved = shared_loader().match(config, model_id=stem.replace("__", "/", 1))
    return Deployment(Program.of(resolved)).at(batch=1, input_tokens=8).analyze()


class TestBlockStorageClosesOnTheCheckpoint(unittest.TestCase):
    MODELS = ("openai__gpt-oss-20b", "deepseek-ai__DeepSeek-V3",
              "meta-llama__Llama-3.2-1B")

    def test_blocks_plus_roots_equal_checkpoint_param_bytes(self):
        for stem in self.MODELS:
            with self.subTest(model=stem):
                analysis = _analysis(stem)
                report = project_analysis(analysis)
                blocks = sum(count * br.total_storage_bytes
                             for _t, count, br in report.block_results)
                roots = sum(node.storage_bytes
                            for node in report.model_level_nodes.values())
                self.assertEqual(blocks + roots,
                                 analysis.checkpoint_param_bytes())

    def test_a_mixed_precision_block_is_not_priced_at_the_average(self):
        """The old construction's signature: every block's bytes were
        exactly `params × (total bytes / total params)`."""
        analysis = _analysis("openai__gpt-oss-20b")
        report = project_analysis(analysis)
        average = analysis.checkpoint_param_bytes() / report.total_params
        at_average = [
            br.template_name for _t, _c, br in report.block_results
            if abs(br.total_storage_bytes - br.total_params * average)
            < 1e-6 * br.total_storage_bytes]
        self.assertEqual(at_average, [], (
            "gpt-oss stores MXFP4 experts beside BF16 attention; a block "
            f"priced at the model average is the old proration: {at_average}"))

    def test_a_tied_head_stores_nothing(self):
        """Llama-3.2-1B ties its head to the embedding: one stored copy,
        on the embedding's row; the head's own storage is zero."""
        report = project_analysis(_analysis("meta-llama__Llama-3.2-1B"))
        head = [n for n in report.model_level_nodes.values()
                if n.boundary_role == "output_head"]
        self.assertEqual(len(head), 1)
        self.assertEqual(head[0].storage_bytes, 0)
        self.assertGreater(head[0].flops, 0, "it still projects every step")


if __name__ == "__main__":
    unittest.main()
