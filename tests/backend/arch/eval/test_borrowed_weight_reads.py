"""A stored tensor is read by whoever consumes it.

`_weight_traffic` says "the read bills whoever moves the bytes". Wherever
a weight reaches its consumer along a dataflow edge instead of being owned
by it — a bias `add`, `F.linear(x, W)`, a sink softmax, a positional table
— it did not: the read was billed to the `parameter_*` node holding the
tensor, and the kernel that streams it was billed zero. One kernel, two
rows, neither of them a kernel that exists.

Worse, the same bytes appeared twice. An edge from a parameter is also an
input edge, so `_activation_reads` priced it as well — at the activation
dtype, which for DeepSeek-V4's F32 hyper-connection matrix is a different
number from the same tensor's weight price.

Two facts here, in the order they have to hold:

  * an identity source reads nothing;
  * its consumer reads it once, at the storage width — or, if the consumer
    looks rows up in it (`OpSpec.lookup`), one row per id.
"""
from __future__ import annotations

import collections
import json
import sys
import unittest
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import shared_loader
from backend.arch.eval import traffic as T
from backend.arch.eval.program import Program
from backend.arch.eval.fusion import Framework
from backend.arch.eval.transforms import Deployment
from backend.inference.framework_bindings import get_binding
from backend.inference.fusion_plans import FUSION_SCHEMES
from backend.arch.eval.analysis import pair_stored
from backend.arch.ops import is_identity_source
from backend.arch.eval.values import product

pytestmark = pytest.mark.regression

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _analyses():
    loader = shared_loader()
    for snap in sorted(SNAPSHOTS.glob("*.json")):
        config = json.loads(snap.read_text(encoding="utf-8"))["config"]
        resolved = loader.match(config, model_id=snap.stem.replace("__", "/", 1))
        if resolved is None:
            continue
        # Every stored tensor and every reader: the language model's walk
        # with each encoder walked beside it.
        yield snap.stem, Deployment(Program.of(resolved)).at(
            batch=1, input_tokens=8).checkpoint()


class TestTheReadBillsWhoeverMovesTheBytes(unittest.TestCase):

    def test_an_identity_source_moves_nothing_at_all(self):
        """Every term, not just the read. A write is the same ghost kernel
        in a different column, and `access.py` carries `act_writes` into
        `access_bytes` and the per-op `t_sol` exactly as it carries reads."""
        billed = []
        for name, analysis in _analyses():
            layer = T.derive(analysis)
            for node in analysis.program.nodes():
                if not is_identity_source(node.op):
                    continue
                traffic = layer.values.get(node.uid)
                if traffic is None:
                    continue
                moved = {
                    field: getattr(traffic, field)
                    for field in ("weight_reads", "metadata_reads",
                                  "act_reads", "act_writes", "kv_reads",
                                  "borrowed_weight_reads",
                                  "borrowed_metadata_reads")
                    if getattr(traffic, field)
                }
                if moved:
                    billed.append((name, node.name, moved))
        self.assertEqual(billed, [], (
            "an identity source's output IS the tensor it stores, so it "
            f"moves nothing and can be billed nothing: {billed[:5]}"))

    def test_the_consumer_is_billed_exactly_the_source_bytes(self):
        """Not "is billed something" — one byte would pass that. The
        resident bytes of the tensors at the other end of the edge,
        through the same `pair_stored` the weight column uses."""
        for name, analysis in _analyses():
            with self.subTest(model=name):
                layer = T.derive(analysis)
                for reader_uid, sources in T._borrowed_sources(
                        analysis.program).items():
                    traffic = layer.values.get(reader_uid)
                    if traffic is None:
                        continue
                    expected = 0
                    for src in sources:
                        for row in pair_stored(
                                src, analysis.storage.values.get(src),
                                analysis.formats.values.get(src)):
                            if row.rep is None:
                                continue
                            expected += row.rep.resident.weight_bytes_for(
                                row.tensor.elements)
                    site = analysis.program.lookups.get(reader_uid)
                    if site is not None:
                        # A lookup is handed the whole table and reads one
                        # row per id: the ids' elements, each a row of the
                        # table's leading axis, never more than the table.
                        (table,) = analysis.storage.values[site.table]
                        ids = analysis.shapes.values[reader_uid].ins[site.ids]
                        rows = min(1.0, float(product(ids))
                                   / float(table.shape[0]))
                        self.assertEqual(
                            float(traffic.table.weight), float(expected),
                            f"{name}: node {reader_uid} is handed "
                            f"{traffic.table.weight}, its table holds "
                            f"{expected}")
                        expected = float(expected) * rows
                    self.assertEqual(
                        float(traffic.borrowed_weight_reads), float(expected),
                        f"{name}: node {reader_uid} borrows "
                        f"{traffic.borrowed_weight_reads}, its sources hold "
                        f"{expected}")

    def test_a_component_wrapper_is_not_a_second_reader(self):
        """A wrapper repeats its body's `inputs:` and has no cost of its
        own; counting it would bill the same read twice."""
        for name, analysis in _analyses():
            borrowed = T._borrowed_sources(analysis.program)
            wrappers = {n.uid for n in analysis.program.nodes()
                        if n.is_component}
            overlap = sorted(set(borrowed) & wrappers)
            self.assertEqual(overlap, [], f"{name}: {overlap}")


class TestFusionKeepsTheBorrowedWeight(unittest.TestCase):
    """A fused kernel still streams the parameter threaded into one of its
    members. Once a match is absorbed those members are gone from
    `FusedProgram.nodes()`, and the edge survives only on the fused node's
    own `inputs_obj` — so a lookup that consulted origins alone would drop
    the bytes silently, which is the failure mode worth a test rather than
    a comment."""

    MODELS = ("deepseek-ai__DeepSeek-V3", "openai__gpt-oss-20b")

    def test_fusing_does_not_change_the_weight_ledger(self):
        binding = get_binding("vllm")
        framework = Framework.of(binding, FUSION_SCHEMES)
        loader = shared_loader()
        for name in self.MODELS:
            with self.subTest(model=name):
                config = json.loads(
                    (SNAPSHOTS / f"{name}.json").read_text())["config"]
                deployment = Deployment(
                    Program.of(loader.match(
                        config, model_id=name.replace("__", "/", 1)))).at(
                        batch=1, input_tokens=2048)
                def ledger(analysis):
                    # Leaves only. A component's `Traffic` is the sum of its
                    # children, so counting wrappers counts the same bytes
                    # twice — and the two trees do not even have the same
                    # wrappers once a match is absorbed.
                    layer = T.derive(analysis)
                    return sum(
                        float(t.weight_reads) + float(t.borrowed_weight_reads)
                        for uid, t in layer.values.items()
                        if not analysis.program.by_uid(uid).children)
                naive = ledger(deployment.analyze())
                fused = ledger(deployment.fused(framework).analyze())
                self.assertAlmostEqual(
                    naive, fused, delta=max(naive * 1e-9, 1),
                    msg=f"{name}: {naive} unfused vs {fused} fused")


class TestEveryStoredParameterHasAReader(unittest.TestCase):
    """The ratchet. A stored parameter no graph node reads is an edge the IR
    dropped, a parameter the forward never reads, or one the forward reads
    in a step the graph does not model. The last two are the spec's own
    declaration (`stored_without_graph_reader`, with its kind, reason and
    source anchor); the first is a bug.

    Stated as an EQUALITY per model against the spec's declarations, not a
    pass-list here: a superset would stay green after the missing op lands,
    and the commit that wires a reader has to drop the declaration with it.
    """

    def test_unread_parameters_are_exactly_the_declared_ones(self):
        for name, analysis in _analyses():
            with self.subTest(model=name):
                stored = {analysis.program.by_uid(r.uid).uid
                          for r in analysis.tensors()
                          if not (r.is_metadata or r.is_buffer)}
                readers: collections.Counter = collections.Counter()
                for node in analysis.program.nodes():
                    if node.is_component:
                        continue
                    for ref in (node.inputs_obj or ()):
                        if ref.src is not None:
                            readers[ref.src.uid] += 1
                    # A tie reads the tensor it names (a shared `module:`, a
                    # tied head), as the validator's unread check counts.
                    if node.tied_to_uid is not None:
                        readers[node.tied_to_uid] += 1
                unread = {
                    node.name for node in analysis.program.nodes()
                    if node.uid in stored
                    and is_identity_source(node.op)
                    and not readers.get(node.uid)
                }
                declared = {node.name for node in analysis.program.nodes()
                            if node.stored_without_graph_reader is not None}
                self.assertEqual(
                    unread, declared, (
                        f"{name}: the stored parameters with no reader are "
                        "not the ones the spec declares `stored_without_graph_reader`. A "
                        "new name means an edge was dropped or a compute "
                        "node is missing; a declared name with a reader "
                        "means the declaration should go."))

    def test_a_declared_parameter_is_stored_and_reads_nothing(self):
        """`stored_without_graph_reader` keeps the bytes in the stored weights and bills
        no read to anything: the parameter is an identity source with no
        reader, so no node borrows it."""
        for name, analysis in _analyses():
            declared = {node.uid for node in analysis.program.nodes()
                        if node.stored_without_graph_reader is not None}
            if not declared:
                continue
            with self.subTest(model=name):
                stored = {r.uid for r in analysis.tensors()}
                self.assertLessEqual(declared, stored)
                layer = T.derive(analysis)
                for uid in declared:
                    traffic = layer.values.get(uid)
                    if traffic is not None:
                        self.assertEqual(float(traffic.weight_reads), 0.0)
                borrowed = {src for sources in T._borrowed_sources(
                    analysis.program).values() for src in sources}
                self.assertEqual(declared & borrowed, set())


if __name__ == "__main__":
    unittest.main()
