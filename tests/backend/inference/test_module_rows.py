"""`physical_module_rows` — the per-op table walked off the grafted +
fused tree. These lock the contracts the reviews pinned down:

- one row per uid, even for a top-level timed collective that also has
  an OpStage (a duplicate used to be possible when the grafted-node pass
  didn't share bookkeeping with the stage walk);
- top-level grafted collectives sit at their anchor in flow position;
- a multi-output fused leaf shows the concat its GEMM physically emits,
  with named ports on the hover;
- depth-0 sums reconcile with the per-op rows exactly.
"""
import unittest

import pytest

from tests.snapshots import resolve_snapshot

from backend.inference.framework_bindings import get_binding
from backend.inference.module_rows import physical_module_rows
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from backend.inference.phase_formulas import enrich_module_breakdown


# -- the priced step, once per deployment --------------------------------
#
# Every test here asks about the same step: Llama-3.2-3B decoding at TP=4.
# Pricing it is the whole cost of this file (5.7 of its 5.8 seconds are
# inside `pipeline.analyze`) and it was priced six times, once per test.
#
# So the module keeps the result per deployment and the file's tests share
# it. The ROWS are not shared: `enrich_module_breakdown` writes into the row
# dicts it is given, and the two refusal tests copy each timing row before
# editing it, so each test gets its own rows built off the one result.

_STEPS: dict = {}


def _step(model: str, parallel: ParallelConfig):
    """The priced step, built once per (model, deployment) for this file."""
    # A `ParallelConfig` is a dataclass with generated equality but no hash,
    # so the mesh enters the key as the `repr` its fields produce.
    key = (model, repr(parallel))
    if key not in _STEPS:
        _STEPS[key] = analyze(
            resolve_snapshot(model), parallel=parallel,
            framework_binding=get_binding("vllm"),
            batch_size=1, avg_input_tokens=128, avg_output_tokens=0,
            phase="decode", breakdown=True,
        )
    return _STEPS[key]


@pytest.fixture(scope="module", autouse=True)
def _forget_steps():
    """Bound the memo to this module: a module-level dict would
    otherwise hold the steps it kept for the rest of the session,
    and one file's worth resident at a time is the point."""
    yield
    _STEPS.clear()


def _rows(model: str, parallel: ParallelConfig):
    result = _step(model, parallel)
    return physical_module_rows(result, result.per_op_timing), result


class TestOneRowPerUid(unittest.TestCase):

    def test_no_duplicate_uids_anywhere(self):
        rows, _ = _rows("meta-llama/Llama-3.2-3B",
                        ParallelConfig(tp_attention=4, tp_dense_ffn=4))
        uids = [r["uid"] for r in rows if r.get("uid") is not None]
        self.assertEqual(len(uids), len(set(uids)),
                         "every uid appears exactly once")

    def test_top_level_collective_sits_at_its_anchor(self):
        rows, _ = _rows("meta-llama/Llama-3.2-3B",
                        ParallelConfig(tp_attention=4, tp_dense_ffn=4))
        names = [r["name"] for r in rows]
        embed = names.index("embed_tokens")
        self.assertEqual(names[embed + 1], "embed_tokens.vocab_all_reduce",
                         "the embedding all_reduce follows its anchor")
        self.assertTrue(rows[embed + 1]["is_comm"])

    def test_depth0_sums_reconcile_with_the_per_op_rows(self):
        rows, result = _rows("meta-llama/Llama-3.2-3B",
                             ParallelConfig(tp_attention=4, tp_dense_ffn=4))
        phase = {"flops_by_module": rows}
        enrich_module_breakdown(phase)
        d0 = [r for r in rows if r["depth"] == 0]
        self.assertEqual(
            sum(r["flops_per_rank"] for r in d0),
            sum(x["flops_per_rank"] for x in result.per_op_timing))
        self.assertEqual(
            sum((r.get("access_bytes_agg") or 0) for r in d0),
            sum((x.get("access_bytes_per_rank") or 0)
                for x in result.per_op_timing))


class TestFusedMultiOutputShape(unittest.TestCase):

    def test_fused_gemm_shows_the_concat_it_emits(self):
        rows, _ = _rows("meta-llama/Llama-3.2-3B",
                        ParallelConfig(tp_attention=4, tp_dense_ffn=4))
        qkv = next(
            r for r in rows
            if r["op"] in ("qkv_parallel_linear", "merged_column_parallel_linear")
        )
        # One tensor, not a null primary output: the per-port shapes
        # concatenated on the last axis, with named ports on the hover.
        self.assertIsNotNone(qkv.get("shape_out"))
        ports = qkv.get("shape_outs")
        self.assertIsInstance(ports, dict)
        self.assertTrue(all(p.startswith("output_") for p in ports))
        self.assertEqual(qkv["shape_out"][-1],
                         sum(s[-1] for s in ports.values()))


if __name__ == "__main__":
    unittest.main()


class TestCountsAreWholeAndPresent(unittest.TestCase):
    """A count is how many times something runs: a fraction or a missing
    count is an error, never rounded down or stood in for by 1."""

    def test_whole_count(self):
        from fractions import Fraction
        from backend.inference.module_rows import whole_count
        self.assertEqual(whole_count(58, "x"), 58)
        self.assertEqual(whole_count(58.0, "x"), 58)
        for bad in (2.5, Fraction(256, 3), None):
            with self.assertRaises(ValueError):
                whole_count(bad, "x")

    def test_instance_count_refuses_an_uncounted_node(self):
        from types import SimpleNamespace
        from backend.inference.module_rows import instance_count
        layer = SimpleNamespace(values={1: 3, 2: 1.5})
        self.assertEqual(instance_count(layer, 1), 3)
        with self.assertRaises(ValueError):
            instance_count(layer, 2)
        with self.assertRaises(ValueError):
            instance_count(layer, 3)

    def test_a_timing_row_without_a_whole_count_is_refused(self):
        _, result = _rows("meta-llama/Llama-3.2-3B",
                          ParallelConfig(tp_attention=4, tp_dense_ffn=4))
        for bad in (None, 2.5):
            timing = [dict(r) for r in result.per_op_timing]
            if bad is None:
                timing[0].pop("count_per_step")
            else:
                timing[0]["count_per_step"] = bad
            with self.assertRaises(ValueError):
                physical_module_rows(result, timing)

    def test_a_collective_whose_anchor_is_not_drawn_is_refused(self):
        _, result = _rows("meta-llama/Llama-3.2-3B",
                          ParallelConfig(tp_attention=4, tp_dense_ffn=4))
        timing = [dict(r) for r in result.per_op_timing]
        comm = next(r for r in timing if r.get("is_comm")
                    and r.get("parent_aggregation_id") == "embed_tokens")
        comm["parent_aggregation_id"] = "no_such_row"
        with self.assertRaisesRegex(ValueError, "no_such_row"):
            physical_module_rows(result, timing)
