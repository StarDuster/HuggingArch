"""Which launches write their operand's elements unchanged, in both layers.

`OpSignature.gathers_from` is the measurement side's reading of the arch
op's `OpSpec.gather_source` — the input position whose activation the
output is an index gather of. Two declarations because the harness and the
cost model may not look an op up by name (`tests/meta/test_eval_boundary.py`
bans `OP_REGISTRY` in `backend/inference` and `kernel_bench`), so this test
is where the two are held to each other: a test may read both layers, and
without it the catalog's copy would drift the first time an op's own
declaration moved.

What rests on it: a quantised operand's record (its elements and the scale
each row carries) travels whole through a launch that only re-lays it out,
so such a cell is one record MOVED (`kernel_bench.quant.ActivationCast.
moved_by`) rather than a cast; a launch that computes its output stays a
refusal however equal its two dtype slots.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from backend.arch.ops import OP_REGISTRY
from backend.inference.op_signatures import all_signatures


def _declared_source(sig) -> object:
    """The gather the ops this signature binds declare, or None."""
    sources = {OP_REGISTRY[op].gather_source for op in sig.binds_ops
               if op in OP_REGISTRY}
    sources.discard(None)
    if len(sources) > 1:
        raise AssertionError(f"{sig.name} binds ops that disagree: {sources}")
    return next(iter(sources), None)


class TestGatheringSignatures(unittest.TestCase):
    def test_each_signature_states_the_gather_its_ops_declare(self):
        for sig in all_signatures():
            with self.subTest(signature=sig.name):
                self.assertEqual(sig.gathers_from, _declared_source(sig))

    def test_the_moe_dispatch_gathers_and_its_sibling_does_not(self):
        """The discriminating pair: the dispatch moves its rows, and the
        combine that mirrors it is the same boundary and REDUCES (a weighted
        sum), so an FP8 record does not travel whole through it."""
        by_name = {sig.name: sig for sig in all_signatures()}
        self.assertEqual(by_name["moe_permute"].gathers_from, 0)
        self.assertEqual(by_name["gather"].gathers_from, 0)
        for computing in ("moe_combine", "quantise_activation",
                          "silu_and_mul", "gemm"):
            self.assertIsNone(by_name[computing].gathers_from, computing)


if __name__ == "__main__":
    unittest.main()
