"""An expert bank's size is declared, and the access row reads all of it.

The bank's experts are the leading axis of its weights, written in each
weight's `shape:` (`[n_expert_copies, out, in]`). A stored tensor is then the
whole bank, so an access row streams it once — no multiplier stands between
the stored bytes and the row, on the global path or the per-rank one.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import shared_loader
from backend.arch.eval.analysis import pair_stored
from backend.arch.eval.traffic import _borrowed_sources
from backend.arch.eval.storage import leads_with_expert_axis
from backend.arch.eval.dim import value_of
from backend.arch.ops import is_identity_source
from backend.arch.layout.physical_cost import physical_access
from backend.inference.parallel_config import ParallelConfig
from backend.inference.step_eval import evaluate_step

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"


class TestTheBankIsTheDeclaredExpertAxis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads(
            (SNAPSHOTS / "openai__gpt-oss-20b.json").read_text())["config"]
        cls.n_experts = int(config["num_local_experts"])
        cls.analysis = evaluate_step(
            shared_loader().match(config, model_id="openai/gpt-oss-20b"),
            ParallelConfig(tp_attention=1, tp_dense_ffn=1, tp_moe_expert=1),
            phase="prefill", batch=1, input_tokens=4096, output_tokens=0, s_prefix_cached=3968,
        ).analysis
        cls.banks = {
            uid: tensors for uid, tensors in cls.analysis.storage.values.items()
            if any(leads_with_expert_axis(t) for t in tensors)}

    def test_every_bank_leads_with_the_configs_expert_count(self):
        self.assertTrue(self.banks, "gpt-oss declares expert banks")
        self.assertEqual(
            {int(value_of(t.shape[0])) for tensors in self.banks.values()
             for t in tensors},
            {self.n_experts})

    def _stored_bytes(self, uid: int) -> int:
        total = 0
        for row in pair_stored(
                uid, self.analysis.storage.values.get(uid),
                self.analysis.formats.values.get(uid)):
            if row.rep is not None:
                total += row.rep.resident.weight_bytes_for(row.tensor.elements)
        return total

    def test_the_access_row_streams_every_expert_once(self):
        """The stored bytes of the whole bank, plus the parameters threaded
        into the kernel (gpt-oss's expert biases arrive as inputs and are
        read once) — nothing multiplies either."""
        access = physical_access(self.analysis, scale_dtype_bytes=4.0)
        borrowed_by = _borrowed_sources(self.analysis.program)
        kernels = [uid for uid in self.banks if not is_identity_source(
            self.analysis.program.by_uid(uid).op)]
        self.assertTrue(kernels)
        for uid in kernels:
            own = self._stored_bytes(uid)
            self.assertGreater(own, 0)
            threaded = sum(self._stored_bytes(src)
                           for src in borrowed_by.get(uid, ()))
            self.assertEqual(access.values[uid].weight_bytes, own + threaded)


if __name__ == "__main__":
    unittest.main()
