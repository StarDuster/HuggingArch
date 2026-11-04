"""The walk decides nothing about the workload.

For one model, deployment, phase and precision the topology is fixed and the
workload only changes numbers. So a walk over the workload as symbols — the
batch, the prompt, the cached prefix, the output, the verify width — asks no
question of them: a value that differs between points is written as the
cases it is (`dim.where`, `Min`, `Max`, `floor`), a condition on the point
is recorded (`dim.require`), never chosen.

The algebra enforces it: a comparison of two laws in the workload's symbols
(`dim.equal`, `dim.is_zero`, a `Dim`'s truth) raises `dim.Undecided`. This
walks the cohort below with no point at all — the walk and every reader of
it (per-rank flops, shapes, cost, HBM traffic, cache) — so a decision
anywhere on that road fails here.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.dim import maximum
from backend.arch.eval.transforms import Deployment
from backend.arch.layout.parallel_config import ParallelConfig
from tests.backend.arch.eval.test_eval_symbolic_workload import _readers
from tests.snapshots import shared_program, shared_walk

MODELS = ("meta-llama/Llama-3.2-1B", "deepseek-ai/DeepSeek-V3",
          "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "openai/gpt-oss-20b",
          "deepseek-ai/DeepSeek-V4-Flash")
MESHES = {"tp1": {}, "dp8ep8": {"dp_attention": 8, "ep_moe": 8},
          "cp8": {"cp": 8}}


def _symbolic_deployment(model, mesh, phase):
    program = shared_program(model)
    deployment = Deployment(program).at(
        batch=1, input_tokens=4096, output_tokens=1024, s_prefix_cached=0,
        s_verify=1)
    if phase == "decode":
        deployment = deployment.during("decode")
    if mesh:
        deployment = replace(deployment,
                             config=ParallelConfig.from_dict(mesh))
    workload, _point = deployment.workload.symbolic()
    workload = replace(workload, logit_positions=(
        1 if phase == "prefill" else maximum(1, workload.s_verify)))
    return replace(deployment, workload=workload)


@pytest.mark.regression
class TestTheWalkAsksNothingOfTheWorkload(unittest.TestCase):

    def test_no_decision_reads_a_workload_symbol(self):
        for model in MODELS:
            for mesh_name, mesh in MESHES.items():
                for phase in ("prefill", "decode"):
                    with self.subTest(model=model, mesh=mesh_name,
                                      phase=phase):
                        _readers(shared_walk(_symbolic_deployment(
                            model, mesh, phase)))


class TestNoWalkTakesTheCountsAsNumbers(unittest.TestCase):
    """There is one walk: over the workload's counts as symbols. A
    deployment whose counts are numbers is that walk read at its point
    (`Deployment.analyze`), and the walk itself refuses numbers — there is
    no second, numeric walk to agree or disagree with it."""

    def test_a_walk_of_numeric_counts_is_refused(self):
        deployment = Deployment(shared_program(MODELS[0])).at(
            batch=2, input_tokens=64)
        with self.assertRaises(TypeError):
            deployment._walk(deployment.binding())

    def test_a_count_stated_as_a_sympy_number_is_refused(self):
        """A SymPy number is a number: it neither walks as one nor slips past
        the point as a law of the caller's own."""
        import sympy as sp
        program = shared_program(MODELS[0])
        for number in (sp.Integer(2), sp.Rational(4, 2), sp.Float(2.0)):
            with self.subTest(number=repr(number)):
                with self.assertRaises(ValueError):
                    shared_walk(Deployment(program).at(batch=number,
                                           input_tokens=64))
                deployment = Deployment(program).at(
                    batch=2, input_tokens=sp.Integer(64))
                workload, _point = Deployment(program).at(
                    batch=2, input_tokens=64).workload.symbolic()
                with self.assertRaises(TypeError):
                    replace(deployment, workload=replace(
                        workload, batch=number))._walk(deployment.binding())

    def test_a_numeric_deployment_is_the_symbolic_walk_at_its_point(self):
        deployment = Deployment(shared_program(MODELS[0])).at(
            batch=2, input_tokens=64)
        workload, point = deployment.workload.symbolic()
        walk = shared_walk(replace(deployment, workload=workload))
        self.assertEqual(walk.at(point).total(), shared_walk(deployment).total())


if __name__ == "__main__":
    unittest.main()
