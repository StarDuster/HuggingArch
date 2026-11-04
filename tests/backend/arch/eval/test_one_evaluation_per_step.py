"""Each distinct inference question performs exactly one evaluator walk.

A step is walked once over the workload's symbols and read at each point
(`step_eval.Cell`), so one capacity estimate asks two questions: the
prefill step — read for the workload-free weight geometry, the liveness
peak and the phase table, each at its own point — and decode. Shape,
FLOPs, cache, storage and communication are folds of those ``Analysis``
objects and must never repeat a question already evaluated.
"""
from __future__ import annotations

import collections
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

SNAPSHOT_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"

#: The model the claim in `step_eval.py` was measured on.
MODEL = "meta-llama__Llama-3.2-3B"

#: Distinct questions per estimate, as enumerated in the module docstring.
EXPECTED_WALKS = 2


def _argument_set(evaluator, binding) -> tuple:
    """Facts that identify one semantic Program/Binding question: the
    program, the phase, the deployment, and the workload it walks (a step
    walks its workload as symbols, so every point of one step is one
    question)."""
    rule = evaluator.rule
    deployment = getattr(rule, "parallel", None)
    return (
        id(evaluator.program.resolved),
        rule.phase,
        repr(evaluator.workload),
        repr(deployment),
    )


class TestOneEvaluationPerStep(unittest.TestCase):

    maxDiff = None

    def _walks(self) -> collections.Counter:
        from backend.arch.eval.evaluator import Evaluator
        from backend.inference.capacity import estimate_capacity
        from backend.inference.parallel_config import ParallelConfig

        config = json.loads(
            (SNAPSHOT_DIR / f"{MODEL}.json").read_text())["config"]
        from tests.snapshots import shared_loader
        resolved = shared_loader().match(config, model_id=MODEL.replace("__", "/", 1))
        # A process walks a model's cells once and later requests read
        # them (`capacity._model_cells`): the count is of an estimate that
        # finds none walked.
        from backend.inference import walk_registry
        walk_registry.WALK_CELLS.clear()
        seen: collections.Counter = collections.Counter()
        real = Evaluator.run

        def spy(evaluator, binding, **kwargs):
            seen[_argument_set(evaluator, binding)] += 1
            return real(evaluator, binding, **kwargs)

        # A cold estimate: the cells an earlier request walked are read by
        # a later one (`capacity._model_cells`), which walks nothing.
        walk_registry.WALK_CELLS.clear()
        with mock.patch.object(Evaluator, "run", spy):
            estimate_capacity(
                 config=config, resolved=resolved, gpu_sku="H200", num_gpus=8,
                avg_input_tokens=4096, avg_output_tokens=256,
                parallel=ParallelConfig(tp_attention=8, tp_dense_ffn=8,
                                        tp_moe_expert=8),
            )
        return seen

    def test_no_argument_set_is_walked_twice(self):
        seen = self._walks()
        self.assertTrue(seen, "no walk was observed — the spy missed the "
                              "call site, so this test proved nothing")
        repeated = {args: n for args, n in seen.items() if n > 1}
        self.assertEqual(
            repeated, {},
            "the same question was asked twice: a second walk at an argument "
            "set already walked is a second opinion on one number. Hand the "
            "consumer the `StepEvaluation` the first walk produced.\n"
            + "\n".join(f"  {n}x {args}" for args, n in repeated.items()),
        )

    def test_the_walk_count_is_the_pinned_one(self):
        seen = self._walks()
        self.assertEqual(
            sum(seen.values()), EXPECTED_WALKS,
            f"{sum(seen.values())} walks, expected {EXPECTED_WALKS}. Each is "
            f"named in this module's docstring; if one is new, it belongs to a "
            f"consumer that should be folding an existing evaluation "
            f"instead.\n" + "\n".join(f"  {args}" for args in sorted(
                map(str, seen))),
        )

    def test_capacity_and_both_workload_phases_are_present(self):
        phases = collections.Counter(args[1] for args in self._walks())
        self.assertEqual(
            dict(phases), {"prefill": 1, "decode": 1},
            "capacity reads one prefill walk (weight geometry, liveness, "
            "the phase table) and one decode walk",
        )


if __name__ == "__main__":
    unittest.main()
