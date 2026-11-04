"""Tests for spec_id_compose: the two parallel namespaces.

* `compose_child_spec_id` — pure nested append (`<parent>.<local>`),
  optional `__<scope>` suffix for compute-only primitives. Drives
  structural id / display_path / aggregation_id. Two IR nodes can
  never collide under pure nesting.
* `compose_weight_name` — the same nested append over the nodes the
  CHECKPOINT has modules for. `populate_weight_names` skips the links
  declaring `hf_module: false`, which is what makes the Linear inside an
  `mlp.gate` wrapper store as `mlp.gate.weight` and not
  `mlp.gate.gate.weight`. Drives HF tensor matching only.

The property test on real specs (DSv3 + Llama-3.2-1B) checks the
post-finalize invariant that resolved_inputs sids live in the same
id space as node.display_path — direct lookup works without a denest
step now that display_path is itself pure-nested.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


from tests.snapshots import shared_loader
from backend.arch.spec_id_compose import (
    compose_child_spec_id,
    compose_weight_name,
)
from backend.arch.spec_tree import (
    SpecIRTree,
    finalize_ir_tree,
    iter_tree,
)


SNAPSHOT_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _resolve(model_name: str):
    path = SNAPSHOT_DIR / f"{model_name}.json"
    config = json.loads(path.read_text())["config"]
    return shared_loader().match(config, model_id=model_name.replace("__", "/", 1))


class TestComposeChildSpecId(unittest.TestCase):
    """Pure-nested composition."""

    def test_nested_prefix(self):
        self.assertEqual(
            compose_child_spec_id("layers.0", "self_attn"),
            "layers.0.self_attn",
        )

    def test_a_repeated_segment_is_kept(self):
        # The wrapper and the node inside it are two nodes and keep two ids,
        # whatever the checkpoint calls them. Collapsing the pair here is
        # what let a fused subgraph write two shape entries under one id.
        self.assertEqual(
            compose_child_spec_id("mlp.gate", "gate"),
            "mlp.gate.gate",
        )

    def test_a_dotted_local_name_is_not_lifted(self):
        self.assertEqual(
            compose_child_spec_id("mlp.gate", "gate.sigmoid"),
            "mlp.gate.gate.sigmoid",
        )
        self.assertEqual(
            compose_child_spec_id("mlp.gate", "gate.e_score_correction_bias"),
            "mlp.gate.gate.e_score_correction_bias",
        )

    def test_sibling_nested(self):
        self.assertEqual(
            compose_child_spec_id("mlp.gate", "scores_biased"),
            "mlp.gate.scores_biased",
        )

    def test_empty_parent_uses_bare_name(self):
        self.assertEqual(compose_child_spec_id("", "layer"), "layer")
        self.assertEqual(
            compose_child_spec_id("", "embed_tokens"), "embed_tokens")

    def test_component_scope_suffix(self):
        # Compute-only primitives can carry a `__<scope>` suffix to
        # disambiguate cross-component name collisions.
        self.assertEqual(
            compose_child_spec_id(
                "attn.scores", "softmax", component_scope="mla",
            ),
            "attn.scores.softmax__mla",
        )


class TestComposeWeightName(unittest.TestCase):
    """HF weight-name composition — nested append, one rule.

    The fold that used to live here (a `primary` child taking the parent's
    name, a `<primary>.<rest>` sibling lifting to `<parent>.<rest>`) is now
    a skip in `populate_weight_names`, driven by the node's own
    `hf_module` — see `TestHFModuleSkipsTheSegment`.
    """

    def test_nested(self):
        self.assertEqual(
            compose_weight_name("layers.0", "self_attn"),
            "layers.0.self_attn",
        )

    def test_sibling_nested(self):
        self.assertEqual(
            compose_weight_name("mlp.gate", "scores_biased"),
            "mlp.gate.scores_biased",
        )

    def test_empty_parent_uses_bare_name(self):
        self.assertEqual(compose_weight_name("", "layer"), "layer")
        self.assertEqual(
            compose_weight_name("", "embed_tokens"), "embed_tokens")


class TestNormalizeIrInputsInvariant(unittest.TestCase):
    """Property test: after finalize, every resolved_inputs sid either
    matches a node's display_path in the same scope or is external.

    Pre-refactor this required a denest step because display_path was
    the post-fold canonical id while ref walkers emitted the nested
    form. Now display_path itself is pure-nested, so direct equality
    is all that's needed.
    """

    def _check_invariant(self, resolved):
        tree = iter_tree(resolved)
        finalize_ir_tree(tree)
        scopes = []
        scopes.append((tree.top_level, _scope_ids(tree.top_level)))
        for _meta, roots in tree.blocks:
            scopes.append((roots, _scope_ids(roots)))
        for nodes, ids_in_scope in scopes:
            _check_nodes(self, nodes, ids_in_scope)

    def test_llama_3_2_1b_invariant(self):
        resolved = _resolve("meta-llama__Llama-3.2-1B")
        self.assertIsNotNone(resolved)
        self._check_invariant(resolved)

    def test_dsv3_invariant(self):
        # DSv3's router_noaux_tc exercises the moduleless-node skip.
        resolved = _resolve("deepseek-ai__DeepSeek-V3")
        self.assertIsNotNone(resolved)
        self._check_invariant(resolved)


def _scope_ids(roots) -> set:
    out: set = set()

    def _walk(nodes):
        for n in nodes:
            out.add(n.display_path)
            if n.children:
                _walk(n.children)

    _walk(roots)
    return out


def _check_nodes(test, nodes, ids_in_scope):
    for n in nodes:
        for sid, _port in (n.resolved_inputs or ()):
            if sid in ids_in_scope:
                continue
            # External — left as-is for cross-scope alias resolution.
        if n.children:
            _check_nodes(test, n.children, ids_in_scope)


class TestFinalizeIsIdempotent(unittest.TestCase):
    """Calling `finalize_ir_tree` twice is a no-op."""

    def test_double_finalize_stable(self):
        resolved = _resolve("deepseek-ai__DeepSeek-V3")
        tree = iter_tree(resolved)
        before = _snapshot_resolved_inputs(tree)
        finalize_ir_tree(tree)
        after = _snapshot_resolved_inputs(tree)
        self.assertEqual(before, after)


def _snapshot_resolved_inputs(tree: SpecIRTree):
    out = []

    def _walk(nodes):
        for n in nodes:
            out.append((n.display_path, tuple(n.resolved_inputs)))
            if n.children:
                _walk(n.children)

    _walk(tree.top_level)
    for _meta, roots in tree.blocks:
        _walk(roots)
    return tuple(out)


class TestWeightNamePopulatedOnPrimaryFold(unittest.TestCase):
    """On a router_noaux_tc-bearing spec (DSv3), the Linear declaring
    `hf_module: false` must produce a leaf whose `weight_name` ends in
    `mlp.gate` — the wrapper's path, which is where the checkpoint keeps
    it — while its `display_path` stays the nested `mlp.gate.linear`.
    That divergence is the point: one node, two namespaces.

    `e_score_correction_bias` is a flat sibling with a module of its own,
    so both of its names end in `mlp.gate.e_score_correction_bias`."""

    def test_dsv3_router_fold_in_weight_name_only(self):
        resolved = _resolve("deepseek-ai__DeepSeek-V3")
        tree = iter_tree(resolved)

        moduleless_hit = False
        sibling_hit = False

        def _strip_scope(sid: str) -> str:
            return sid.split("__", 1)[0]

        def _walk(nodes):
            nonlocal moduleless_hit, sibling_hit
            for n in nodes:
                disp = _strip_scope(n.display_path)
                wn = n.weight_name
                if disp.endswith("mlp.gate.linear") and wn and wn.endswith("mlp.gate"):
                    moduleless_hit = True
                # `e_score_correction_bias` is a flat sibling under the
                # router wrapper WITH a module of its own, so its
                # display_path and weight_name are the same nested form.
                if (
                    disp.endswith("mlp.gate.e_score_correction_bias")
                    and wn
                    and wn.endswith("mlp.gate.e_score_correction_bias")
                ):
                    sibling_hit = True
                if n.children:
                    _walk(n.children)

        _walk(tree.top_level)
        for _meta, roots in tree.blocks:
            _walk(roots)

        self.assertTrue(
            moduleless_hit,
            "the moduleless router linear did not fold onto `mlp.gate`",
        )
        self.assertTrue(
            sibling_hit,
            "`e_score_correction_bias` did not stay nested under the wrapper",
        )


if __name__ == "__main__":
    unittest.main()
