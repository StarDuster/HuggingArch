"""Per-leaf weight CLASS — what a weight is, asked without a deployment.

`backend.inference.sharding` used to answer two questions: which part of the
model a weight belongs to, and what fraction of it a rank holds. The second
answer is gone. A class cannot produce a ratio — a fused QKV's axes enter
additively, and an extent below the degree does not split at all — so per-rank
bytes come from the forward-propagated SHAPE, and `tests/test_shard_invariants.py`
is where that derivation is locked.

What is left here is the classification, and its contract is that it is
*structural*: the same leaf gets the same class at every deployment, because
the question is which knob OWNS the weight and that does not change when the
knob is set to 1. The vocab endpoints are the one place the old code answered
by name — `display_path == "embed_tokens"` — which said `aux` for every model
spelling them otherwise; they now go through `sharding.vocab_kind_of_leaf`,
which asks the leaf's declared `boundary_role`. This is the weight-breakdown
DISPLAY class only; placement is independent and reads `weight_axes`.
"""
from __future__ import annotations

import sys
import unittest

from tests.snapshots import resolve_snapshot
from pathlib import Path
from typing import List, ClassVar


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.parallel_config import ParallelConfig
from backend.inference.sharding import (
    moe_child_class,
    role_class,
    step_class,
)
from backend.inference.weight_breakdown import per_rank_weight_breakdown




def _resolved(model_name):
    return resolve_snapshot(model_name)


def _flatten_leaves(group_or_node, out: List[dict]) -> None:
    """Walk a per_rank_weight_breakdown TreeNode dict-tree, collecting leaves."""
    if not group_or_node.get("is_component", False) and not group_or_node.get("children"):
        out.append(group_or_node)
        return
    for child in group_or_node.get("children", []) or []:
        _flatten_leaves(child, out)


def _classes(model_name: str, parallel: ParallelConfig):
    """`{leaf name: class}` for every weight-bearing leaf of the rendered tree,
    plus the model-level nodes (where the vocab endpoints live)."""
    resolved = _resolved(model_name)
    if resolved is None:
        return None
    # A pipelined tree describes one named stage; any stage's leaves carry
    # the same classes.
    out = per_rank_weight_breakdown(
        resolved=resolved, parallel=parallel, pp_stage=0 if parallel.pp > 1 else None)
    leaves: List[dict] = []
    for group in out["groups"]:
        _flatten_leaves(group, leaves)
    for ml_node in out["model_level"]:
        _flatten_leaves(ml_node, leaves)
    return {leaf["name"]: leaf["class_"] for leaf in leaves
            if leaf["bytes_full"] > 0}


class TestRoleClass(unittest.TestCase):
    """A role's weight class is `roles.yaml`'s: the root of its lineage when
    the registry marks it a weight class, else None (inherit)."""

    def test_known_role_passes_through(self):
        self.assertEqual(role_class("attention"), "attention")
        self.assertEqual(role_class("ffn_dense"), "ffn_dense")
        self.assertEqual(role_class("ffn_moe"), "ffn_moe")

    def test_registered_variant_collapses_to_its_lineage_root(self):
        self.assertEqual(role_class("attention_swa"), "attention")
        self.assertEqual(role_class("ffn_moe_shared"), "ffn_moe")

    def test_unregistered_variant_is_not_attention_by_spelling(self):
        """`attention_sink` used to collapse to attention, and that was the
        bug: membership was a `startswith("attention_")` test, so a role
        nobody had classified was silently granted attention's sharding
        semantics. Now `roles.yaml` answers, and only for roles someone
        declared; `SpecLoader` refuses a spec naming one it does not."""
        self.assertIsNone(role_class("attention_sink"))

    def test_a_role_naming_no_weight_class_inherits(self):
        # Wrappers, norms and the router name no class of their own.
        for role in ("block", "norm", "router", None):
            self.assertIsNone(role_class(role), role)


class TestStepClass(unittest.TestCase):
    """One level down the IR tree: role classes, transparent fallthrough,
    and an MoE layer keeping everything inside it."""

    def test_transparent_wrapper_inherits_prev_class(self):
        self.assertEqual(step_class("attention", None), "attention")
        self.assertEqual(step_class("attention", "norm"), "attention")

    def test_explicit_role_wins_over_prev_class(self):
        self.assertEqual(step_class("attention", "ffn_dense"), "ffn_dense")

    def test_the_invocation_not_the_component_name_is_classified(self):
        """One component called in two roles in one model (DeepSeek-V3's
        `swiglu_ffn`: the first layers' dense FFN, the later layers' shared
        expert) is classified by each call's own role."""
        self.assertEqual(step_class(None, "ffn_dense"), "ffn_dense")
        self.assertEqual(step_class(None, "ffn_moe"), "ffn_moe")

    def test_nothing_inside_an_moe_layer_leaves_it(self):
        self.assertEqual(step_class("ffn_moe", "ffn_dense"), "ffn_moe")
        self.assertEqual(step_class("ffn_moe", "router"), "ffn_moe")

    def test_moe_parts_are_read_off_declarations(self):
        """The gate refines `router`; the bank is what runs inside the
        `permute` -> `combine` zone; whatever is neither is every-token work
        (the shared expert). No name is read."""
        self.assertEqual(moe_child_class("ffn_moe", True), "moe_routed")
        self.assertEqual(moe_child_class("ffn_moe", False), "moe_shared")
        self.assertEqual(moe_child_class("ffn_moe_shared", False), "moe_shared")
        self.assertEqual(moe_child_class("router", False), "moe_router")
        # DeepSeek-V4's hash-routing table refines `router`: a gate part.
        self.assertEqual(moe_child_class("router_hash_table", False), "moe_router")


class TestMoePartsOnTheTree(unittest.TestCase):
    """The parts of an MoE layer, on the rendered tree."""

    def _tree(self, model):
        resolved = _resolved(model)
        out = per_rank_weight_breakdown(resolved=resolved, parallel=ParallelConfig())
        found = {}

        def walk(node, path):
            here = f"{path}/{node['name']}"
            found.setdefault(here, set()).add(node["class_"])
            for child in node.get("children") or ():
                walk(child, here)

        for group in out["groups"]:
            walk(group, "")
        return found

    def _part(self, tree, suffix):
        (cls,) = set().union(*(c for p, c in tree.items() if p.endswith(suffix)))
        return cls

    def test_a_layer_nested_in_an_moe_call_is_not_one_part(self):
        """Hy3 wraps its MoE layer (`ffn`) in another MoE call (`mlp`). The
        inner layer holds the gate, the routed bank and the shared expert;
        each is its own part, and the layer holding all three is the layer
        (it used to take the gate's class for all of it)."""
        tree = self._tree("tencent/Hy3")
        self.assertEqual(self._part(tree, "/mlp/ffn"), "ffn_moe")
        self.assertEqual(self._part(tree, "/mlp/ffn/experts"), "moe_routed")
        self.assertEqual(self._part(tree, "/mlp/ffn/experts/gate_proj"), "moe_routed")
        self.assertEqual(self._part(tree, "/mlp/ffn/shared_mlp"), "moe_shared")
        self.assertEqual(self._part(tree, "/mlp/ffn/expert_bias"), "moe_router")

    def test_the_hash_routing_table_is_the_gate(self):
        tree = self._tree("deepseek-ai/DeepSeek-V4-Flash")
        self.assertEqual(self._part(tree, "tid2eid"), "moe_router")


class TestClassIsDeploymentIndependent(unittest.TestCase):
    """The property that makes it a class rather than a factor.

    A weight belongs to attention whether or not anyone splits attention. Read
    the degrees to decide, and a `tp_attention=1` model reports as entirely
    replicated — which is what the retired divisor table did in the one place
    it was reachable, and why the two mechanisms had to be collapsed into one.
    """

    MODELS = (
        "meta-llama__Llama-3.2-1B",
        "deepseek-ai__DeepSeek-V3",
        "deepseek-ai__DeepSeek-V2-Lite",
        "google__gemma-4-12B-it",
    )

    # Wildly different deployments — including the degenerate one. Classes
    # must be byte-identical across all of them.
    _CONFIGS = (
        ParallelConfig(),
        ParallelConfig(tp_attention=8, tp_dense_ffn=8),
        ParallelConfig(tp_attention=4, tp_dense_ffn=4, tp_moe_expert=4,
                       ep_moe=2, dp_attention=2),
        ParallelConfig(cp=4, pp=2),
    )

    def test_same_leaf_same_class_at_every_deployment(self):
        for m in self.MODELS:
            with self.subTest(model=m):
                base = _classes(m, self._CONFIGS[0])
                for cfg in self._CONFIGS[1:]:
                    self.assertEqual(
                        _classes(m, cfg), base,
                        f"{m}: leaf classes changed with the deployment")

    def test_every_leaf_has_a_class(self):
        for m in self.MODELS:
            with self.subTest(model=m):
                by_name = _classes(m, self._CONFIGS[1])
                unclassed = [n for n, c in by_name.items() if not c]
                self.assertEqual(unclassed, [],
                                 f"{m}: leaves the walker never classified")


class TestVocabEndpointsAskBoundaryRole(unittest.TestCase):
    """The embedding and the LM head are found by their declared boundary_role.

    Every model here spells at least one of them in a way the retired
    `display_path == "embed_tokens"` test did not recognize, and got `aux` —
    the class for a replicated leftover — while the shape layer was
    vocab-sharding the same tensor. `speech_embeddings` is the counterexample
    in the other direction: an embedding whose table is its own, smaller
    vocabulary, which nothing vocab-shards. Its banks are read by the audio
    encoder (vLLM mimo_audio.py `apply_speech_embeddings`), so they sit in
    the audio group, not among the model-level vocab endpoints (None here).
    """

    CASES: ClassVar[dict] = {
        "google__gemma-4-12B-it": {"language_model.embed_tokens": "embedding"},
        "THUDM__glm-4-9b": {
            "transformer.embedding.word_embeddings": "embedding",
            "transformer.output_layer": "lm_head",
        },
        "moonshotai__Kimi-K3": {
            "language_model.model.embed_tokens": "embedding",
            "language_model.lm_head": "lm_head",
        },
        "XiaomiMiMo__MiMo-V2.5": {
            "embed_tokens": "embedding",
            "lm_head": "lm_head",
            "speech_embeddings": None,
        },
    }

    def test_endpoints_are_classified_by_declared_role_not_by_name(self):
        for model, expect in self.CASES.items():
            with self.subTest(model=model):
                by_name = _classes(model, ParallelConfig(tp_attention=4))
                got = {n: by_name.get(n) for n in expect}
                self.assertEqual(got, expect)


if __name__ == "__main__":
    unittest.main()
