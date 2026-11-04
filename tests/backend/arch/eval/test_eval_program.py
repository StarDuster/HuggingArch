"""`Program` — and the two places the new evaluator is anchored to the old one.

The new machinery is being built whole before it is wired in, which means it has
no protection from the existing suite: nothing calls it, so nothing catches it
drifting. These are the substitutes, and both compare against the engine that is
in production rather than against my own expectations:

  - `Program.base_binding` seeds the walk's starting dims on
    every shipped snapshot. That map is not just the spec's `dims:` block — it
    carries the optional flags a spec may omit (`Rb=0`, `Es=0`) and the mirrored
    defaults (`Ec=E`, `Is=Im`, `Tg=Ng`), and getting one of those wrong shifts
    every downstream number by a factor nobody would question.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.program import Program
from tests.snapshots import shared_loader
from backend.arch.spec_tree import SpecIRNode

pytestmark = pytest.mark.regression  # walks every snapshot

SNAPSHOT_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _resolve(model_id: str):
    """One snapshot's resolved spec, by model id."""
    path = SNAPSHOT_DIR / (model_id.replace("/", "__", 1) + ".json")
    config = json.loads(path.read_text())["config"]
    return shared_loader().match(config, model_id=model_id)


def _resolved_snapshots():
    for path in sorted(glob.glob(str(SNAPSHOT_DIR / "*.json"))):
        try:
            config = json.loads(Path(path).read_text())["config"]
            yield os.path.basename(path), shared_loader().match(
                config, model_id=Path(path).stem.replace("__", "/", 1))
        except Exception:
            continue


class TestBaseBinding(unittest.TestCase):
    def test_copies_axis_stays_shardable(self):
        """`Ec = E` is a COPIED default, not a lookup-derived value, so expert
        parallelism can shard it — and must, since EP shards exactly the
        expert-copies axis and `Ec` is defaulted on nearly every MoE spec.

        The dispatch denominator `n_route_targets` is what keeps this
        single-division: it is deliberately never in a shard policy, and
        neither is `n_experts`, so `M = B*S*K*dp/n_route_targets` stays global
        while the copies divide.
        """
        checked = 0
        for name, resolved in _resolved_snapshots():
            b = Program.of(resolved).base_binding
            if not b.get("n_experts"):
                continue
            checked += 1
            with self.subTest(model=name):
                self.assertEqual(b["n_expert_copies"], b["n_experts"],
                                 "copies axis defaults to the expert count")
        self.assertGreater(checked, 5, "no MoE snapshots exercised")


class TestProgramStructure(unittest.TestCase):
    def test_nodes_are_uid_unique(self):
        """`Layer` keys on uid, so a collision would silently merge two nodes'
        values instead of failing."""
        for name, resolved in _resolved_snapshots():
            with self.subTest(model=name):
                p = Program.of(resolved)
                uids = [n.uid for n in p.nodes()]
                self.assertEqual(len(uids), len(set(uids)),
                                 f"{name}: duplicate uid in the IR")

    def test_by_uid_round_trips(self):
        _name, resolved = next(_resolved_snapshots())
        p = Program.of(resolved)
        for node in p.nodes():
            self.assertIs(p.by_uid(node.uid), node)

    def test_blocks_are_exposed(self):
        _name, resolved = next(_resolved_snapshots())
        self.assertTrue(Program.of(resolved).blocks)

    def test_component_outputs_share_the_input_reference_grammar(self):
        split = SpecIRNode("split", "hc_split_sinkhorn", False, uid=3)
        inner = SpecIRNode(
            "inner",
            "inner_comp",
            True,
            output_names=("split",),
            secondary_outputs=(("post", "split.post"), ("comb", "split.comb")),
            children=[split],
            uid=2,
        )
        dotted = SpecIRNode("branch.result", "identity", False, uid=4)
        outer = SpecIRNode(
            "outer",
            "outer_comp",
            True,
            output_names=("branch.result",),
            secondary_outputs=(
                ("comb", "inner.comb"),
                ("dotted", "branch.result"),
            ),
            children=[inner, dotted],
            uid=1,
        )
        resolved = SimpleNamespace(
            spec=SimpleNamespace(custom_ops={}),
        )
        program = Program(
            resolved=resolved,
            ir=SimpleNamespace(),
        )

        self.assertEqual(program.output_leaf(outer), (dotted, "out"))
        self.assertEqual(program.output_leaf(inner, "post"), (split, "post"))
        self.assertEqual(program.output_leaf(outer, "comb"), (split, "comb"))
        self.assertEqual(program.output_leaf(outer, "dotted"), (dotted, "out"))


if __name__ == "__main__":
    unittest.main()
