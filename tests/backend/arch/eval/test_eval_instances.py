"""The copy count, against the one `iter_tree` baked.

`Analysis.instances` re-derives what `SpecIRNode.multiplicity` already holds —
block count times every enclosing `repeat:` — and that is only worth doing
because the walk resolves the counts against ITS binding rather than the spec's
global one. Which is also the risk: a re-derivation is a second answer until
something checks it against the first.

So this pins the two together at the binding where they must agree (the spec's
own), across every shipped snapshot. An expert bank is not a copy count: its
experts are the leading axis of its weights, so expert parallelism divides
that axis on a rank, and a token runs the router's share of it.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.transforms import Deployment
from tests.snapshots import shared_loader, shared_program, shared_walk
from backend.inference.parallel_config import ParallelConfig

pytestmark = pytest.mark.regression  # walks every snapshot

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _analysis(name: str, **degrees: Any):
    program = shared_program(name.replace('__', '/', 1))
    base = Deployment(program).at(batch=1, input_tokens=128)
    if not degrees:
        return program, shared_walk(base)
    ep = int(degrees.get("ep", 1) or 1)
    return program, shared_walk(Deployment(
        program,
        config=ParallelConfig(ep_moe=ep),
    ).at(batch=1, input_tokens=128))


class TestAgainstTheBakedCount(unittest.TestCase):
    def test_every_snapshot_reproduces_its_multiplicity(self):
        """At the spec's own binding the walk must reproduce `iter_tree` exactly.

        Every node, every model. A disagreement here means the walk's product of
        declared `repeat:`s is not the product `iter_tree` accumulated — which
        would make the re-derivation a second answer rather than the same answer
        asked at a different binding. The checkpoint walk is the one that
        reaches every node: the language model's step, and each encoder in its
        own encode binding.
        """
        for path in sorted(SNAPSHOTS.glob("*.json")):
            program = shared_program(path.stem.replace('__', '/', 1))
            analysis = Deployment(program).at(
                batch=1, input_tokens=128).checkpoint()
            wrong: List[Tuple[str, Any, Any]] = []
            missing: List[str] = []
            for node in program.nodes():
                if node.uid not in analysis.instances:
                    missing.append(node.name)
                    continue
                got = analysis.instances[node.uid]
                want = node.multiplicity if node.multiplicity is not None else 1
                if got != want:
                    wrong.append((node.name, got, want))
            self.assertEqual(wrong, [], f"{path.stem}: {wrong[:5]}")
            # EVERY node, not most of them. A count is written before the node
            # is evaluated — a node the walk refuses still exists that many
            # times — so anything absent here was never reached at all, and a
            # comparison that quietly skips it would pass by covering less.
            self.assertEqual(missing, [], f"{path.stem}: {missing[:5]}")

    def test_storage_totals_match_the_parameter_total(self):
        """The count is what turns per-copy storage into a model total.

        `Cost` gets its multiplier during the walk and storage does not, so if
        the two counts ever came apart this is where it would show: the same
        model, totalled two ways.
        """
        for name in ("Qwen__Qwen3-30B-A3B", "deepseek-ai__DeepSeek-V2-Lite",
                     "openai__gpt-oss-20b", "meta-llama__Llama-3.2-1B"):
            with self.subTest(model=name):
                _program, analysis = _analysis(name)
                summed = sum(row.tensor.elements * row.count
                             for row in analysis.tensors()
                             if row.tensor.is_parameter)
                self.assertEqual(summed, analysis.total().params)

    def test_a_bank_runs_the_routers_share_of_its_experts(self):
        """A token runs `top-k / destinations` of a bank's experts: its active
        parameters are that share of its parameters. Qwen3-30B's experts ARE
        its destinations, so the share is `n_experts_active / n_expert_copies`.
        """
        program, analysis = _analysis("Qwen__Qwen3-30B-A3B")
        binding = program.base_binding
        self.assertEqual(binding["n_route_targets"], binding["n_expert_copies"])
        bank = next(node for node in program.nodes()
                    if node.display_path == "mlp.experts.gate_proj")
        cost = analysis.cost[bank.uid]
        self.assertEqual(
            cost.active_params * binding["n_expert_copies"],
            cost.params * binding["n_experts_active"])


class TestPerRank(unittest.TestCase):
    """Counts follow the binding, which is the whole reason to re-derive them."""

    MOE = ("Qwen__Qwen3-30B-A3B", "deepseek-ai__DeepSeek-V2-Lite",
           "openai__gpt-oss-20b", "MiniMaxAI__MiniMax-M2.7")

    def test_expert_parallelism_divides_the_bank(self):
        """A rank holds `n_expert_copies/ep` of each bank's experts, and
        nothing else divides.

        Both halves matter. Something must divide — a per-rank walk that finds
        the whole bank on every rank has not modelled the deployment. And only
        weights led by the expert axis may divide: `n_route_targets` is the
        dispatch denominator and the gate's width, and a walk that shards it
        has corrupted `M` or shrunk the router to a fraction of its classes.
        """
        from backend.arch.eval.storage import leads_with_expert_axis
        from backend.arch.layout.physical_cost import physical_cost_plan

        for name in self.MOE:
            with self.subTest(model=name):
                _program, whole = _analysis(name)
                _program, per_rank = _analysis(name, ep=4)
                local = physical_cost_plan(per_rank).cost.values
                divided = {uid for uid, cost in whole.cost.values.items()
                           if cost.params and local[uid].params != cost.params}
                self.assertTrue(
                    divided, "the expert bank stayed whole on every rank")
                for uid in divided:
                    self.assertEqual(whole.cost[uid].params // 4,
                                     local[uid].params)
                    self.assertTrue(all(
                        leads_with_expert_axis(t)
                        for t in whole.storage.values[uid] if t.is_parameter))

    def test_no_bank_repeats_over_the_routing_denominator(self):
        """The declaration itself, across the whole corpus.

        `n_experts` answers "how many experts hold weights" and
        `n_expert_copies` answers "which axis of a tensor are they"; they are
        equal by default and only the second may be divided. The component
        library used to write the first where it meant the second — right
        number, wrong axis — and downstream had to guess the copies back from
        a leaf's class, which over-divided every replicated tensor that shared
        it.
        """
        offenders: Dict[str, int] = {}
        for path in sorted(SNAPSHOTS.glob("*.json")):
            program, _ = _analysis(path.stem)
            bad = sum(1 for n in program.nodes()
                      if n.repeat_raw and "n_experts" in _symbols(n.repeat_raw))
            if bad:
                offenders[path.stem] = bad
        self.assertEqual(offenders, {})


class TestZeroComputeExpertsAreDestinations(unittest.TestCase):
    """LongCat-Flash routes to 768 destinations of which 512 hold weights.

    `LongcatFlashExperts.forward` sends `expert_idx >= n_routed_experts` to an
    `nn.Identity()` — 256 zero-compute slots kept as load-balance reserve for
    the aux-loss-free router. They are real routing destinations and they do no
    work, so a token that picks 12 of the 768 lands on 12 x 512/768 = 8 banks
    that hold weights.

    The model's own published figures are that arithmetic: 18.6B-31.3B
    activated parameters per token, averaging 27B, is the range over how many
    of the 12 draws happen to hold weights, with 8 as the balanced expectation.
    Dividing by the weighted count instead reports 31.3B — the top of the
    range — for every token.
    """

    def test_twelve_draws_of_seven_sixty_eight_activate_eight_banks(self):
        program, analysis = _analysis(
            "meituan-longcat__LongCat-Flash-Thinking-2601")
        banks = [n for n in program.nodes()
                 if n.display_path.endswith("mlp.experts")]
        self.assertTrue(banks, "LongCat has no routed expert bank")
        for bank in banks:
            gate = next(c for c in bank.children if c.name == "gate_proj")
            self.assertEqual(
                int(analysis.storage.values[gate.uid][0].shape[0]), 512)
            cost = analysis.cost[gate.uid]
            self.assertEqual(cost.active_params * 512, cost.params * 8)

    def test_every_other_model_routes_to_exactly_its_weighted_experts(self):
        """`n_route_targets` is `n_experts` wherever nothing is zero-compute.

        The denominator has to be a symbol of its own — it differs from
        `n_experts` by LongCat's zero-compute reserve, and it is the one factor
        the expert-count identity scales beside the bank's axis. But a symbol
        of its own is also a symbol that can drift, so this pins it back to
        `n_experts` everywhere the reserve is empty.
        """
        for path in sorted(SNAPSHOTS.glob("*.json")):
            dims = shared_loader().match(
                json.loads(path.read_text())["config"],
                model_id=path.stem.replace("__", "/", 1)).dim_values
            if "n_experts" not in dims:
                continue
            with self.subTest(path.stem):
                self.assertEqual(
                    dims["n_route_targets"],
                    dims["n_experts"] + dims["n_zero_experts"])
                if not dims["n_zero_experts"]:
                    self.assertEqual(dims["n_route_targets"], dims["n_experts"])


def _symbols(expr: str) -> set:
    import re
    return set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", str(expr)))


if __name__ == "__main__":
    unittest.main()
