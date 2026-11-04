"""Phase A consistency tests for the spec_id → object-ref refactor.

`finalize_ir_tree` (invoked at the tail of `iter_tree`) populates
parent / uid / inputs_obj / component_scope on every SpecIRNode in
parallel with the existing spec_id / resolved_inputs strings. This
file asserts the new fields stay consistent with the old ones so
consumers can be migrated incrementally without breaking the
dual-write invariant.

Failing here means the refactor's foundation is wrong — fix before
moving to Phase B.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


from tests.snapshots import shared_loader
from backend.arch.spec_tree import (
    InputRef,
    SpecIRNode,
    SpecIRTree,
    iter_tree,
)


SNAPSHOT_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"

MODELS = (
    "meta-llama__Llama-3.2-1B",
    "deepseek-ai__DeepSeek-V3",
    "deepseek-ai__DeepSeek-V4-Pro",
)


def _resolve(model_name: str):
    path = SNAPSHOT_DIR / f"{model_name}.json"
    config = json.loads(path.read_text())["config"]
    return shared_loader().match(config, model_id=model_name.replace("__", "/", 1))


def _build_finalized_tree(model_name: str) -> SpecIRTree:
    resolved = _resolve(model_name)
    # iter_tree runs finalize_ir_tree at exit, so the returned tree
    # already has parent / uid / inputs_obj populated.
    return iter_tree(resolved)


def _all_nodes(tree: SpecIRTree) -> List[SpecIRNode]:
    out: List[SpecIRNode] = []

    def _walk(nodes):
        for n in nodes:
            out.append(n)
            if n.children:
                _walk(n.children)

    _walk(tree.top_level)
    for _meta, roots in tree.blocks:
        _walk(roots)
    return out


def _all_nodes_with_scope_root(tree: SpecIRTree):
    """Yield (scope_roots, node) for every node — scope_roots is the
    list this node lives under (tree.top_level or one block's roots),
    so callers can verify intra-scope invariants without crossing the
    template boundary."""

    def _walk(roots, nodes):
        for n in nodes:
            yield roots, n
            if n.children:
                yield from _walk(roots, n.children)

    yield from _walk(tree.top_level, tree.top_level)
    for _meta, roots in tree.blocks:
        yield from _walk(roots, roots)


def _root_set(nodes) -> set:
    """Return the id() of every node directly in the given list (a
    scope's root frontier). Used to assert parent==None for these."""
    return {id(n) for n in nodes}


class TestParentPointers(unittest.TestCase):
    """Every non-root node has a parent; that parent's children list
    contains the node by identity (no orphan / no cross-scope leak)."""

    def test_parent_wiring(self):
        for model in MODELS:
            with self.subTest(model=model):
                tree = _build_finalized_tree(model)
                top_roots = _root_set(tree.top_level)
                block_roots = set()
                for _meta, roots in tree.blocks:
                    block_roots |= _root_set(roots)

                for n in _all_nodes(tree):
                    if id(n) in top_roots or id(n) in block_roots:
                        self.assertIsNone(
                            n.parent,
                            f"scope-root node {n.display_path!r} has non-None parent",
                        )
                        continue
                    self.assertIsNotNone(
                        n.parent,
                        f"non-root node {n.display_path!r} has None parent",
                    )
                    self.assertIn(
                        n, n.parent.children,
                        f"node {n.display_path!r} not in its parent's children",
                    )


class TestUidUniqueDeterministic(unittest.TestCase):
    """uid is unique within one build, and equal across two builds of
    the same spec (deterministic — required for any future caching /
    wire-stable handle use)."""

    def test_uid_unique(self):
        for model in MODELS:
            with self.subTest(model=model):
                tree = _build_finalized_tree(model)
                uids = [n.uid for n in _all_nodes(tree)]
                self.assertEqual(
                    len(uids), len(set(uids)),
                    f"{model}: uid collisions: "
                    f"{len(uids) - len(set(uids))} duplicates",
                )

    def test_uid_deterministic_across_builds(self):
        for model in MODELS:
            with self.subTest(model=model):
                # Two independent build paths — fresh ResolvedSpec each
                # time to bypass the iter_tree memoization on the
                # resolved instance.
                t1 = _build_finalized_tree(model)
                t2 = _build_finalized_tree(model)
                ids1 = [(n.display_path, n.uid) for n in _all_nodes(t1)]
                ids2 = [(n.display_path, n.uid) for n in _all_nodes(t2)]
                self.assertEqual(ids1, ids2)


class TestInputsObjAlignedWithResolved(unittest.TestCase):
    """Every entry in `inputs_obj` corresponds positionally to an entry
    in `resolved_inputs`. The `src` pointer is into the SAME scope as
    the consuming node; external refs (cross-scope) have src=None."""

    def test_position_and_scope(self):
        for model in MODELS:
            with self.subTest(model=model):
                tree = _build_finalized_tree(model)
                for scope_roots, n in _all_nodes_with_scope_root(tree):
                    self.assertEqual(
                        len(n.inputs_obj), len(n.resolved_inputs),
                        f"{n.display_path!r}: inputs_obj/resolved_inputs len mismatch",
                    )
                    scope_ids = {id(x) for x in _scope_node_iter(scope_roots)}
                    for (sid, port), ref in zip(n.resolved_inputs, n.inputs_obj,
                                                strict=True):
                        self.assertIsInstance(ref, InputRef)
                        self.assertEqual(ref.port, port)
                        if ref.src is None:
                            continue  # external — accept
                        self.assertEqual(ref.src.display_path, sid)
                        self.assertIn(
                            id(ref.src), scope_ids,
                            f"{n.display_path!r} input {sid!r} resolved to a node "
                            f"outside its own scope",
                        )


def _scope_node_iter(roots):
    for n in roots:
        yield n
        if n.children:
            yield from _scope_node_iter(n.children)


class TestSynthesizedNodesCarryProvenance(unittest.TestCase):
    """Materialized fusion and communication nodes carry provenance on the
    delivered SpecIRNode —
    `"fusion"` or `"comm"`. The marker is the structural barrier that
    distinguishes synthesized nodes from yaml-emitted ones without
    parsing names or op strings.
    """

    def test_fused_leaves_carry_fusion_provenance(self):
        from backend.inference.pipeline import analyze
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.framework_bindings import get_binding

        for model in MODELS:
            with self.subTest(model=model):
                resolved = _resolve(model)
                ar = analyze(
                    resolved,
                    parallel=ParallelConfig(),
                    framework_binding=get_binding("vllm"),
                    batch_size=1, avg_input_tokens=1, avg_output_tokens=0,
                    breakdown=False,
                )
                if not ar.fusion_matches:
                    self.skipTest(f"{model}: no fusion patterns matched")
                nodes = {
                    node.uid: node
                    for _meta, roots in ar.post_tree.blocks
                    for node in _scope_node_iter(roots)
                }
                nodes.update({
                    node.uid: node
                    for node in _scope_node_iter(ar.post_tree.top_level)
                })
                checked = 0
                for match in ar.fusion_matches:
                    fused_sid = match["fused_aggregation_id"]
                    node = nodes.get(match["fused_uid"])
                    if node is None:
                        continue
                    self.assertEqual(
                        node.provenance, "fusion",
                        f"{model}: fused node {fused_sid!r} provenance="
                        f"{node.provenance!r}, expected 'fusion'",
                    )
                    checked += 1
                self.assertGreater(
                    checked, 0,
                    f"{model}: no fused nodes surfaced for provenance check",
                )

    def test_yaml_walker_emits_spec_provenance(self):
        """Every node iter_tree returns has provenance='spec' by
        default — the structural barrier that validate / materialize
        relies on (they run BEFORE any synthesis pass)."""
        for model in MODELS:
            with self.subTest(model=model):
                tree = _build_finalized_tree(model)
                offenders: List[str] = [
                    f"{model}: {node.display_path!r} "
                    f"provenance={node.provenance!r}"
                    for node in _all_nodes(tree)
                    if node.provenance != "spec"
                ]
                self.assertEqual(
                    offenders, [],
                    "iter_tree emitted non-'spec' provenance:\n"
                    + "\n".join(offenders[:20]),
                )


if __name__ == "__main__":
    unittest.main()
