"""How many nodes the op templates actually describe.

`shape_in` has been an einsum signature all along — `signature_for` says so —
and `IndexSignature` labels every axis of every input and the output, which is
what lets a consumer ask whether two input axes are the SAME index rather than
merely the same size. When the templates do not fit a node, that machinery
refuses and the caller falls back to positional dispatch.

It refused for 746 of 3,028 nodes across the corpus. A quarter of the graph
was running on the fallback, and the reason was uniform: one template per op,
applied at more than one rank. The op that lost most was `transpose`, which
declares no template at all while `axes` at its call site already determines
its mapping completely.

Pinned rather than asserted at zero. What is left is the explicit forms the
families in the pin file name, among them `position_table_add` and `neg_exp`.
A selection (`topk`, `dsa_topk_mask`, `sliding_window_kv`) keeps the label of
the axis it selects along, its extent bounded (`IndexSignature.selects`).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import backend.arch.axis_propagation as ax
from tools.corpus_pins.corpus import CorpusModel
from tools.corpus_pins.pins.index_refusals import INDEX_REFUSALS, refusals

pytestmark = pytest.mark.regression  # walks every snapshot

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"

#: The refusals each model brings, by family `op@rank`: pin P7,
#: `tests/golden/pins/index_refusals.json`: the families listed there per
#: model, such as the root `embedding` at rank 3, a rank-4 `linear` and
#: `sample`, and the rank-1 `exp` / `neg_exp` of a linear-attention decay
#: gate. A checkpoint added on an existing architecture brings that
#: architecture's refusals, no template changed.
PINNED = INDEX_REFUSALS.load()


class TestIndexSignatureCoverage(unittest.TestCase):
    # One test walks every snapshot's validator; the regression lane's own
    # 900 s ceiling is for a test, not for this whole-corpus measurement
    # (SOL takes the same allowance).
    @pytest.mark.timeout(3600)
    def test_the_templates_still_describe_what_they_did(self):
        """746 → 74. Every step was checked against the shape-derived flops
        of all 39 snapshots, which is the number a wrong signature moves and
        a tensor-name gate cannot see. One step moved it — declaring
        `permute`'s second input, which its graph really does pass — and was
        reverted for that reason: the arity is wrong, and correcting it makes
        a routing index a shape source. Both have to be true before it
        changes."""
        measured = {path.stem: refusals(CorpusModel(path.stem))
                    for path in sorted(SNAPSHOTS.glob("*.json"))}
        over = {stem: (sum(families.values()), PINNED[stem]["count"])
                for stem, families in measured.items()
                if sum(families.values()) > PINNED[stem]["count"]}
        self.assertEqual(
            over, {},
            "more nodes lost their index signature than pinned (measured, "
            "pinned) — an op is being applied at a rank none of its templates "
            "describes")
        total = sum(sum(families.values()) for families in measured.values())
        pinned, floor = INDEX_REFUSALS.totals(PINNED)
        self.assertLessEqual(total, pinned)
        self.assertGreaterEqual(
            total, floor,
            f"{pinned - total} fewer refusals than pinned, which is good — "
            f"lower the pins so they cannot drift back")


if __name__ == "__main__":
    unittest.main()


class TestSignatureLabelsAreTheAxes(unittest.TestCase):
    """A fitting template is not enough: its labels are the axes' identities,
    so a template of the right rank with the wrong labels passes the count
    above and still says something false. Checked where each rank-4
    pointwise template is actually applied."""

    def _signature(self, model: str, node_name: str):
        import json

        from backend.arch.eval.projection import project_model
        from tests.snapshots import shared_loader
        from backend.arch.validate._shapes import _rank_node

        config = json.loads((SNAPSHOTS / f"{model.replace('/', '__')}.json")
                            .read_text())["config"]
        resolved = shared_loader().match(config, model_id=model)
        report = project_model(resolved, batch=1, input_tokens=16)

        def find(nodes):
            for nr in nodes:
                if nr.name == node_name and nr.op:
                    return nr
                hit = find(nr.children or [])
                if hit is not None:
                    return hit
            return None

        nr = next(hit for _t, _c, br in report.block_results
                  if (hit := find(br.nodes)) is not None)
        ins = nr.shape_in
        ins = [ins] if ins and not isinstance(ins[0], (list, tuple)) else ins
        node = _rank_node(len(nr.shape_out), None, None, [len(s) for s in ins])
        return ax.signature_for(ax._lookup(nr.op), node, len(nr.shape_out))

    def test_the_v4_indexer_relu_reads_the_score_matrix(self):
        sig = self._signature("deepseek-ai/DeepSeek-V4-Pro", "score_relu")
        self.assertEqual(sig.ins, (("B", "n_heads", "s_query", "s_kv"),))
        self.assertEqual(sig.out, ("B", "n_heads", "s_query", "s_kv"))

    def test_the_gemma_v_norm_reads_per_head_features(self):
        sig = self._signature("google/gemma-4-12B-it", "v_norm")
        self.assertEqual(sig.ins, (("B", "n_heads", "s_ctx", "dim"),))
        self.assertEqual(sig.out, ("B", "n_heads", "s_ctx", "dim"))

    def test_the_kda_decay_gate_is_scaled_per_head(self):
        for name in ("g_sigmoid", "g_decay"):
            sig = self._signature("moonshotai/Kimi-K3", name)
            self.assertEqual(sig.ins[0][:3], ("B", "n_heads", "s_ctx"), name)
            self.assertEqual(sig.out[:3], ("B", "n_heads", "s_ctx"), name)
