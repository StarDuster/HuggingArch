"""Layout-propagated state on the public analysis path.

`Deployment.analyze()` uses `LayoutPropagation`. This file pins the
facts that a count cannot say: a row-parallel exit holds a partial on
the family that contracted, a residual add sees a complete tensor, a
state change has an operation.

Serving-edge counts — `EDGE` plus the two `INTERNAL_*` sites — are
pinned. Endpoint closures (stage-close, cache write, model output) are
extra and are not in that count; they are the walk finishing tensors
that have no downstream consumer.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.eval.transition import RedistributeSite
from tests.snapshots import shared_loader
from backend.arch.ops import OP_REGISTRY
from backend.arch.layout.axis_use import AXIS_USE, CONTRACT
from backend.arch.layout.foldlaw import HOMOMORPHISM_REGISTRY
from backend.arch.layout.layout import Block, Replicated, Unknown
from backend.arch.layout.materializer import (
    materialize_comm_events)
from backend.arch.layout.propagation import EdgeState
from backend.inference.parallel_config import ParallelConfig

pytestmark = pytest.mark.regression  # walks every snapshot

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"
SERVING_SITES = {
    RedistributeSite.EDGE,
    RedistributeSite.INTERNAL_IN,
    RedistributeSite.INTERNAL_OUT,
}

#: `(model, case) → serving-edge collectives`
PINNED: Dict[Tuple[str, str], int] = {
    # Dense TP: one reduction per row-parallel exit, deferred past the
    # residual add to the first consumer that needs the tensor whole.
    # Endpoint closures (embedding stage-close, vocab gather at the head)
    # are extra and not in the count.
    ("meta-llama__Llama-3.2-1B", "tp"): 2,
    ("google__gemma-2-9b", "tp"): 4,
    ("deepseek-ai__DeepSeek-V2-Lite", "tp"): 4,
    ("Qwen__Qwen3-30B-A3B", "tp"): 2,
    # EP: two all_to_all per bank, out and back. The deployment is spelled
    # with BOTH degrees because that is the one that exists: attention is
    # request-sharded, the bank is expert-sharded, and the seam is the
    # exchange. `ep_moe=8` alone is a legal mesh (attention stays at 1) and
    # derives an EP all_reduce instead — not the pair a MoE actually runs.
    ("Qwen__Qwen3-30B-A3B", "ep"): 2,
    ("deepseek-ai__DeepSeek-V2-Lite", "ep"): 2,
    # CP is Ulysses (the pre-10-19 default, which these counts were pinned
    # on — `CASES["cp"]` asks for it by name): attention's sequence cut is
    # repaired by an all_to_all onto heads. GQA pays it in and back; MLA
    # repartitions the query, both decompressed K/V halves and the output,
    # beside the compressed-KV gathers its cache write still needs.
    ("meta-llama__Llama-3.2-1B", "cp"): 4,
    ("google__gemma-2-9b", "cp"): 8,
    ("Qwen__Qwen3-30B-A3B", "cp"): 4,
    ("deepseek-ai__DeepSeek-V2-Lite", "cp"): 12,
}

CASES = {
    "tp": dict(tp_attention=4, tp_dense_ffn=4, tp_moe_expert=4),
    "ep": dict(ep_moe=8, dp_attention=8),
    "cp": dict(cp=2, cp_scheme="ulysses"),
}


def _analysis(resolved: Any, **cfg: int):
    """Public analysis: global extents, layout riding beside them."""
    program = Program.of(resolved)
    config = ParallelConfig(**cfg) if cfg else None
    analysis = Deployment(program, config=config).at(
        batch=1, input_tokens=128).analyze()
    return analysis, program


def _serving_collectives(analysis) -> Tuple[Tuple[Any, str], ...]:
    """`(world, op)` on serving edges, one transfer per SSA tensor."""
    seen: set = set()
    out: list = []
    for edge in analysis.redistributes:
        if edge.site not in SERVING_SITES:
            continue
        key = (edge.producer, edge.producer_port, edge.required)
        if key in seen:
            continue
        seen.add(key)
        out.extend(edge.collectives)
    return tuple(out)


def _closure(analysis) -> Tuple[str, ...]:
    """A state change without an operation is not closed."""
    return tuple(
        f"uid {edge.node} in[{edge.input_index}]: state changes "
        f"with no operation — layout has no such move"
        for edge in analysis.redistributes
        if edge.arrived != edge.required and not edge.collectives)


def _axis_family(axis) -> str | None:
    return axis.group.id if isinstance(axis, Block) else None


def _families(state: EdgeState) -> set:
    return {axis.group.id for axis in state.layout if isinstance(axis, Block)}


class TestServingEdgeCounts(unittest.TestCase):
    def test_counts_have_not_moved(self):
        for (name, case), expected in PINNED.items():
            with self.subTest(model=name, case=case):
                payload = json.loads((SNAPSHOTS / f"{name}.json").read_text())
                resolved = shared_loader().match(
                    payload["config"], model_id=name.replace("__", "/", 1))
                analysis, _program = _analysis(resolved, **CASES[case])
                self.assertEqual(len(_serving_collectives(analysis)), expected)


class TestClosure(unittest.TestCase):
    """The equation, checked — not the injector, and not a report.

    A state change must correspond to an operation. The walk records a
    redistribute exactly when what arrived is not what was required, so the
    bridging is by construction — but a bridge with no collective in it is
    a state that changed on paper while nothing moved.
    """

    def test_the_pinned_models_close(self):
        for name, case in PINNED:
            with self.subTest(model=name, case=case):
                payload = json.loads((SNAPSHOTS / f"{name}.json").read_text())
                analysis, _program = _analysis(
                    shared_loader().match(
                           payload["config"], model_id=name.replace("__", "/", 1)),
                    **CASES[case])
                self.assertEqual(_closure(analysis), ())

    def test_the_whole_corpus_closes_when_layout_runs(self):
        """Unsupported meshes refuse by name; a walk that ran must still close.

        Layout does not yet cover every snapshot on every axis (linear-
        attention CP is one refusal). A mesh the walk refuses is another
        test's question. A mesh that runs with an empty collective is this
        one's.
        """
        for path in sorted(glob.glob(str(SNAPSHOTS / "*.json"))):
            name = os.path.basename(path)[:-5]
            try:
                config = json.loads(Path(path).read_text())["config"]
                resolved = shared_loader().match(
                    config, model_id=name.replace("__", "/", 1))
            except Exception:
                continue
            for case, cfg in CASES.items():
                with self.subTest(model=name, case=case):
                    try:
                        analysis, _program = _analysis(resolved, **cfg)
                    except ValueError:
                        continue
                    self.assertEqual(_closure(analysis), ())

    def test_the_check_can_fail(self):
        """A gate that cannot go red is decoration."""
        payload = json.loads(
            (SNAPSHOTS / "meta-llama__Llama-3.2-1B.json").read_text())
        analysis, _program = _analysis(
            shared_loader().match(
                   payload["config"], model_id="meta-llama/Llama-3.2-1B"),
            **CASES["tp"])
        self.assertEqual(_closure(analysis), ())

        edge = analysis.redistributes[0]
        fictional = replace(edge, collectives=())
        doctored = replace(
            analysis, redistributes=(*analysis.redistributes, fictional))
        self.assertTrue(any("no operation" in v for v in _closure(doctored)))

        with self.assertRaises(ValueError):
            EdgeState((Unknown("no world"),))


class TestTheStateItselfIsRight(unittest.TestCase):
    """Counts agreeing is worth little if the states behind them are wrong."""

    def _llama_tp(self):
        payload = json.loads(
            (SNAPSHOTS / "meta-llama__Llama-3.2-1B.json").read_text())
        return _analysis(
            shared_loader().match(
                   payload["config"], model_id="meta-llama/Llama-3.2-1B"),
            **CASES["tp"])

    def test_a_row_parallel_exit_holds_a_partial(self):
        """The state a shape difference cannot represent: `o_proj` and
        `down_proj` contract along the sharded axis, so every rank holds a
        full-shaped INCOMPLETE tensor. The family is the one that owns the
        contracted axis, so the two exits answer differently even though
        one knob set both.
        """
        analysis, program = self._llama_tp()
        worlds = {"o_proj": "tp_attention", "down_proj": "tp_dense_ffn"}
        exits = [n for n in program.nodes() if n.name in worlds]
        self.assertTrue(exits, "no row-parallel exits found")
        for node in exits:
            with self.subTest(node.name):
                state = analysis.states[node.uid]
                self.assertTrue(state.carries_partial(), node.name)
                self.assertEqual(state.value.group.id, worlds[node.name])
                self.assertTrue(
                    all(isinstance(axis, Replicated) for axis in state.layout),
                    f"{node.name} a partial is full-shaped")

    def test_a_column_parallel_projection_holds_a_shard(self):
        analysis, program = self._llama_tp()
        for name, world in (("q_proj", "tp_attention"),
                            ("gate_proj", "tp_dense_ffn")):
            node = next(n for n in program.nodes() if n.name == name)
            with self.subTest(name):
                state = analysis.states[node.uid]
                self.assertFalse(state.carries_partial(), name)
                self.assertEqual(_axis_family(state.layout[-1]), world)

    def test_the_head_shard_survives_the_whole_attention_body(self):
        """Q is split on heads, and every node from the projection to the
        merge computes on its own heads. Gathering in the middle would be
        doing TP for no benefit.
        """
        analysis, program = self._llama_tp()
        interior = [n for n in program.nodes()
                    if n.name in ("q_rope", "qk_score", "softmax",
                                  "attn_apply", "o_merge_heads")]
        self.assertTrue(interior)
        for node in interior:
            with self.subTest(node.name):
                self.assertIn(
                    "tp_attention", _families(analysis.states[node.uid]),
                    f"{node.name} lost the head shard")

    def test_every_reduction_resolves_a_partial(self):
        """An all_reduce is the finalizer of a pending sum. A gather is a
        layout repair, not a reduction, and under tensor parallelism the
        residual edges should not need one.
        """
        analysis, _program = self._llama_tp()
        for edge in analysis.redistributes:
            ops = [op for _w, op in edge.collectives]
            if "all_reduce" not in ops:
                continue
            with self.subTest(node=edge.node):
                self.assertTrue(edge.resolves_a_partial)

    def test_a_mode_is_laid_over_the_resting_cut_not_substituted_for_it(self):
        """Under DP-attention the residual stream is token-sharded, and a
        linear contracts the FEATURE axis — so the token shard passes
        through it untouched. The only communication the pair needs is the
        routing exchange, so a GATHER appearing here is the regression.
        """
        payload = json.loads(
            (SNAPSHOTS / "Qwen__Qwen3-30B-A3B.json").read_text())
        analysis, program = _analysis(
            shared_loader().match(
                   payload["config"], model_id="Qwen/Qwen3-30B-A3B"),
            **CASES["ep"])
        q = next(n for n in program.nodes() if n.name == "q_proj")
        token_family = _axis_family(analysis.states[q.uid].layout[0])
        self.assertEqual(token_family, "dp_attention",
                         "DP-attention should shard the stream")
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            node = next(n for n in program.nodes() if n.name == name)
            with self.subTest(name):
                self.assertEqual(
                    _axis_family(analysis.states[node.uid].layout[0]),
                    token_family,
                    f"{name} lost the token shard the request cut carries")
        self.assertEqual(
            {op for _w, op in _serving_collectives(analysis)},
            {"all_to_all"},
            "expert parallelism moves tokens; it gathers nothing")

    def test_a_collective_costs_what_it_moves(self):
        """Both figures below are derived from the model's own dims rather
        than read off the implementation.
        """
        payload = json.loads(
            (SNAPSHOTS / "meta-llama__Llama-3.2-1B.json").read_text())
        config = payload["config"]
        analysis, _program = _analysis(
            shared_loader().match(config, model_id="meta-llama/Llama-3.2-1B"),
            **CASES["tp"])
        # Every layer reduces a `[B, S, d_model]` BF16 residual twice —
        # once out of attention, once out of the FFN — and a ring
        # all_reduce moves `2(N-1)/N` of it. Endpoint closures are not
        # this bill.
        residual = 1 * 128 * config["hidden_size"] * 2
        expected = 2 * (2 * 3 / 4) * residual * config["num_hidden_layers"]
        events = [
            event for event in materialize_comm_events(analysis)
            if event.op == "all_reduce" and event.site == RedistributeSite.EDGE]
        self.assertEqual(len(events), 2)
        self.assertEqual(
            int(sum(event.bytes_per_rank for event in events)), int(expected))

    def test_a_routing_exchange_costs_the_same_both_ways(self):
        """Tokens go out to their experts and the results come back, so the
        two legs move one tensor twice — and the arithmetic is checkable
        without this package at all.

        `B` is one attention stream. Each of its tokens routes to
        `n_experts_active` experts; a ring all_to_all moves `(N-1)/N` of
        that per layer.
        """
        payload = json.loads(
            (SNAPSHOTS / "Qwen__Qwen3-30B-A3B.json").read_text())
        config = payload["config"]
        analysis, _program = _analysis(
            shared_loader().match(config, model_id="Qwen/Qwen3-30B-A3B"),
            **CASES["ep"])
        legs = [
            event for event in materialize_comm_events(analysis)
            if event.op == "all_to_all"]
        self.assertEqual([event.op for event in legs], ["all_to_all"] * 2)
        self.assertEqual(legs[0].bytes_per_rank, legs[1].bytes_per_rank,
                         "out and back move the same tokens")
        per_layer = (1 * 128 * config["num_experts_per_tok"]
                     * config["hidden_size"] * 2)
        expected = 2 * (7 / 8) * per_layer * config["num_hidden_layers"]
        self.assertEqual(
            int(sum(event.bytes_per_rank for event in legs)), int(expected))

    def test_a_dense_model_under_tp_assumes_nothing(self):
        """On a dense model under tensor parallelism the assumption list is
        EMPTY: every state is derived from a declaration.
        """
        for name in ("meta-llama__Llama-3.2-1B", "google__gemma-2-9b"):
            with self.subTest(name):
                payload = json.loads((SNAPSHOTS / f"{name}.json").read_text())
                analysis, _program = _analysis(
                    shared_loader().match(
                           payload["config"], model_id=name.replace("__", "/", 1)),
                    **CASES["tp"])
                self.assertEqual(analysis.assumptions, ())

    def test_the_reduction_sits_at_the_residual_add(self):
        """WHERE, not how many — and the counts could not tell the difference.

        Standard tensor parallelism reduces at the row-parallel exit, before
        the residual: `o_proj` and `down_proj` produce a partial, the add
        resolves it, and every norm downstream sees a whole tensor.
        """
        analysis, program = self._llama_tp()
        reduced_at = {
            program.by_uid(edge.node).name
            for edge in analysis.redistributes
            if edge.site == RedistributeSite.EDGE and any(
                op == "all_reduce" for _w, op in edge.collectives)}
        self.assertEqual(reduced_at, {"post_attn_add", "out"})
        for name in ("post_attn_add", "post_attention_layernorm", "out"):
            node = next(n for n in program.nodes() if n.name == name)
            with self.subTest(name):
                state = analysis.states[node.uid]
                self.assertFalse(
                    state.carries_partial(),
                    f"{name} should see a whole tensor")
                self.assertTrue(
                    all(isinstance(axis, Replicated) for axis in state.layout),
                    f"{name} should see a whole tensor")

    def test_a_partial_is_never_carried_past_a_non_linear_op(self):
        """A partial is not a slice of the answer — it is a full-shaped WRONG
        answer on every rank, correct only after the reduction. So only an
        op that is a registered homomorphism, or that produces the partial
        by contracting a sharded axis, can hold one.
        """
        payload = json.loads(
            (SNAPSHOTS / "meta-llama__Llama-3.2-1B.json").read_text())
        analysis, program = _analysis(
            shared_loader().match(
                   payload["config"], model_id="meta-llama/Llama-3.2-1B"),
            **CASES["tp"])
        for node in program.nodes():
            if node.children:
                continue
            state = analysis.states.values.get(node.uid)
            if state is None or not state.carries_partial():
                continue
            if node.op in HOMOMORPHISM_REGISTRY:
                continue
            uses = AXIS_USE.get(node.op or "") or {}
            produces = any(
                getattr(use, "category", None) == CONTRACT
                for use in uses.values())
            self.assertTrue(
                produces,
                f"{node.name} ({node.op}) holds a partial and neither a "
                f"homomorphism nor a contract produces one")

    def test_the_assumption_list_only_shrinks(self):
        """Layout derives the CP ring's successor (Ulysses) and the routing
        seam from declarations. The leftover todo against `ops.py` is
        empty on this path; pin that so adding a model cannot quietly
        grow it.
        """
        seen: Counter = Counter()
        for name in ("meta-llama__Llama-3.2-1B", "google__gemma-2-9b",
                     "Qwen__Qwen3-30B-A3B", "deepseek-ai__DeepSeek-V2-Lite"):
            payload = json.loads((SNAPSHOTS / f"{name}.json").read_text())
            resolved = shared_loader().match(
                payload["config"], model_id=name.replace("__", "/", 1))
            for cfg in (CASES["tp"], CASES["ep"],
                        dict(ep_moe=8), CASES["cp"]):
                analysis, _program = _analysis(resolved, **cfg)
                for entry in analysis.assumptions:
                    seen[entry.op] += 1
        self.assertEqual(dict(seen), {})

    def test_context_parallelism_emits_the_ulysses_exchange(self):
        """Context parallelism moves data. Matching Ulysses emits
        all_to_all; a zero here would be a claim that it is free.
        """
        for name in ("meta-llama__Llama-3.2-1B", "google__gemma-2-9b"):
            with self.subTest(name):
                payload = json.loads((SNAPSHOTS / f"{name}.json").read_text())
                analysis, _program = _analysis(
                    shared_loader().match(
                           payload["config"], model_id=name.replace("__", "/", 1)),
                    **CASES["cp"])
                ops = {op for _w, op in _serving_collectives(analysis)}
                self.assertIn("all_to_all", ops)

    def test_without_parallelism_nothing_is_distributed(self):
        payload = json.loads(
            (SNAPSHOTS / "meta-llama__Llama-3.2-1B.json").read_text())
        analysis, _program = _analysis(shared_loader().match(
            payload["config"], model_id="meta-llama/Llama-3.2-1B"))
        self.assertEqual(analysis.redistributes, ())
        self.assertTrue(all(
            not state.carries_partial()
            and all(isinstance(axis, Replicated) for axis in state.layout)
            for state in analysis.states.values.values()))


class TestCouplingDeclaration(unittest.TestCase):
    """`couples_on` is a catalog completeness gate. Layout's equivalent is
    axis_use, tested next to the registry. This class does not re-run
    the public walk.
    """

    def test_every_registered_op_says_what_it_couples_along(self):
        """`None` is the absence of a claim, and the walk raises on it. So an
        op added without the field works until the day someone shards a model
        that uses it — a failure whose distance from its cause is the whole
        registry.

        `()` is a fine answer and the commonest one. What is not fine is
        silence: an elementwise op that has simply never been thought about is
        indistinguishable, at the call site, from one that was.
        """
        silent = sorted(name for name, spec in OP_REGISTRY.items()
                        if getattr(spec, "couples_on", None) is None)
        self.assertEqual(silent, [], "registered op declares no `couples_on`")

        # Slot names, not symbols: they are looked up in the op's own shape
        # signature and its call site's `dims:`, so a repeat is a copy-paste
        # and a non-identifier is a typo that would resolve to nothing.
        malformed = sorted(
            f"{name}: {spec.couples_on}" for name, spec in OP_REGISTRY.items()
            if len(set(spec.couples_on)) != len(spec.couples_on)
            or not all(isinstance(s, str) and s.isidentifier()
                       for s in spec.couples_on))
        self.assertEqual(malformed, [], "`couples_on` entry that is not a "
                                        "distinct slot name")

    def test_a_declaration_no_call_site_can_read_is_named_or_absent(self):
        """The third way to be silent, and the one the two tests above miss.

        A slot resolves through either of two channels: the op's own
        `shape_in`, whose labels the arriving tensors bind, or the call
        site's `dims:`. An entry in NEITHER is written down, well-formed,
        and unreadable — it names an axis no comparison will ever mention,
        so every shard clears it. That fails closed today, because a named
        axis resolving to nothing clears nothing; it is still an op whose
        declaration has no effect, which is a fact that should have a name
        rather than a comment.

        The question asked is whether a declaration CAN be read, not what it
        resolves to — so the channels are checked statically, without
        standing up a deployment. `head_repeat` is absent from the
        assertion entirely: it has no call site in the corpus, so it is
        untested, not inert, and the two deserve different words.
        """
        declared = {name: set(spec.couples_on)
                    for name, spec in OP_REGISTRY.items() if spec.couples_on}
        readable = {name: set() for name in declared}
        seen = Counter()
        for name, spec in OP_REGISTRY.items():
            if name not in declared:
                continue
            signature = spec.shape_in or ()
            groups = (signature if signature and isinstance(signature[0], (list, tuple))
                      else (signature,))
            for group in groups:
                readable[name] |= {l for l in group if isinstance(l, str)}

        for path in sorted(SNAPSHOTS.glob("*.json")):
            try:
                program = Program.of(
                    shared_loader().match(
                           json.loads(path.read_text())["config"],
                           model_id=path.stem.replace("__", "/", 1)))
            except Exception:      # a spec that will not load is another test's
                continue
            for node in program.nodes():
                name = node.op or ""
                if name not in declared:
                    continue
                seen[name] += 1
                readable[name] |= {slot for slot, expr
                                   in (node.dims_raw or {}).items() if expr}

        unreadable = sorted(name for name, slots in declared.items()
                            if seen[name] and not (slots & readable[name]))
        self.assertEqual(unreadable, [],
                         "op whose `couples_on` no call site can resolve — "
                         "give it a `shape_in`, or retire the entry")


if __name__ == "__main__":
    unittest.main()
