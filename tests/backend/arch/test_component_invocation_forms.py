"""A component is the same graph wherever it is invoked.

`hc_head` is called twice in DeepSeek-V4: as a node inside the
`dsv4_mtp_block` graph, and once at the model root. Both call sites pass
the same four things — a residual and three `nn.Parameter`s — and the body
reads them by the names the component declares (`x`, `fn`, `scale`,
`base`).

Until the model root could say `op:`, the second form went through
`template:`, which is for layer stacks and carries no `inputs:` into the
IR. The body's refs to `fn` / `scale` / `base` stayed unresolved and the
three parameters they name had no reader at all — invisible, because a
shape does not depend on that edge (`linear_external_weight` takes `in`
and `out` from the node's own `dims:`), and only visible once something
asked what dtype the weight is or who moved its bytes.

So the fact worth pinning is not "the two forms produce identical trees" —
identity strings differ, and should. It is that **an edge to a stored
weight is present in both**. A residual edge is allowed to stay
unresolved: it names an enclosing scope, which is what frame aliases are
for.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import shared_loader
from backend.arch.eval.program import Program
from backend.arch.ops import is_identity_source

SNAP = (PROJECT_ROOT / "backend" / "arch" / "snapshots"
        / "deepseek-ai__DeepSeek-V4-Pro.json")




def _wrappers(program, op_name: str):
    return [n for n in program.nodes()
            if n.is_component and n.op == op_name]


def _leaves(node, prefix=""):
    """(local path, node) for every non-component descendant."""
    for child in node.children or ():
        path = f"{prefix}{child.name}"
        if child.is_component:
            yield from _leaves(child, prefix=f"{path}.")
        else:
            yield path, child


class TestBothInvocationFormsWireTheSameWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        resolved = shared_loader().match(
            json.loads(SNAP.read_text())["config"],
            model_id=SNAP.stem.replace("__", "/", 1))
        cls.program = Program.of(resolved)
        cls.calls = _wrappers(cls.program, "hc_head")

    def test_both_call_sites_are_component_wrappers(self):
        """The model root used to flatten its call — no wrapper node, so
        there was nothing to compare against the block-graph one."""
        self.assertGreaterEqual(
            len(self.calls), 2,
            "expected `hc_head` invoked both inside dsv4_mtp_block and at "
            f"the model root; found {len(self.calls)} wrapper(s)")

    def test_the_bodies_have_the_same_shape(self):
        bodies = [sorted((p, n.op) for p, n in _leaves(call))
                  for call in self.calls]
        for other in bodies[1:]:
            self.assertEqual(bodies[0], other,
                             "the same component expanded to different bodies")

    def test_every_weight_edge_is_wired_at_both_call_sites(self):
        by_path: dict = {}
        for call in self.calls:
            for path, leaf in _leaves(call):
                by_path.setdefault(path, []).append(leaf)
        unwired = []
        for path, leaves in by_path.items():
            if len(leaves) < 2:
                continue
            width = max(len(lf.inputs_obj or ()) for lf in leaves)
            for slot in range(width):
                srcs = []
                for lf in leaves:
                    refs = lf.inputs_obj or ()
                    ref = refs[slot] if slot < len(refs) else None
                    src = ref.src if ref is not None else None
                    srcs.append(src.op if src is not None else None)
                # A weight edge at one call site is a weight edge at both.
                # (Residual edges resolve to None at a call site whose
                # producer is an enclosing scope; those are exempt.)
                identity = [s is not None and is_identity_source(s)
                            for s in srcs]
                if any(identity) and not all(identity):
                    unwired.append((path, slot, srcs))
        self.assertEqual(
            unwired, [],
            "these inputs read a stored weight at one call site and nothing "
            f"at the other: {unwired}")


if __name__ == "__main__":
    unittest.main()
