"""Independent aux stacks, persistent endpoints, fusion rebase."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval import cache as cache_mod
from backend.arch.eval.dim import value_of
from backend.arch.eval.fusion import Framework, rebase_analysis
from tests.snapshots import shared_loader
from backend.arch.layout import propagation, weight_endpoints
from backend.arch.layout.layout import Block, Replicated, physical_local_extent
from backend.arch.layout.physical import physical_shapes
from backend.inference.framework_bindings import get_binding
from backend.inference.fusion_plans import FUSION_SCHEMES
from backend.inference.parallel_config import ParallelConfig
from backend.inference.step_eval import evaluate_step

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"
GEMMA = "google__gemma-4-12B-it.json"
LLAMA = "meta-llama__Llama-3.2-3B.json"


def _resolved(snapshot: str):
    payload = json.loads((SNAPSHOTS / snapshot).read_text())
    resolved = shared_loader().match(
        payload["config"], model_id=Path(snapshot).stem.replace("__", "/", 1))
    if resolved is None:
        raise AssertionError(f"snapshot {snapshot} did not resolve")
    return resolved


def _workload(phase: str):
    return dict(batch=1, input_tokens=32, output_tokens=0)


class TestVisionAndAuxStacks(unittest.TestCase):
    """The encoders run in their own encode phase, not in the language
    model's step, so their entries, states and weights are read off the
    whole-checkpoint walk (`Deployment.checkpoint()`): the step's walk with
    every encoder walked beside it."""
    cases = None

    @classmethod
    def setUpClass(cls):
        if TestVisionAndAuxStacks.cases is None:
            resolved = _resolved(GEMMA)
            parallel = ParallelConfig(tp_attention=8, tp_dense_ffn=8)
            TestVisionAndAuxStacks.cases = {}
            for phase in ("prefill", "decode"):
                step = evaluate_step(
                    resolved, parallel, phase=phase, **_workload(phase))
                TestVisionAndAuxStacks.cases[phase] = (
                    step, step.as_deployment().checkpoint())
        cls.cases = TestVisionAndAuxStacks.cases

    def test_independent_vision_and_audio_entries_are_replicated(self):
        for phase, (_step, analysis) in self.cases.items():
            with self.subTest(phase=phase):
                stacks = {stack.name: stack for stack in analysis.program.stacks}
                # `video` is the vision tower called again on the video
                # frames (`call: vision`): its own encoder entry.
                self.assertEqual(set(stacks), {"", "vision", "video", "audio"})
                self.assertEqual(stacks["vision"].aux_kind, "vision")
                self.assertEqual(stacks["video"].aux_kind, "vision")
                self.assertEqual(stacks["audio"].aux_kind, "audio")
                for name in ("vision", "video", "audio"):
                    first = stacks[name].stages[0].roots[0]
                    state = analysis.states.values[first.uid]
                    self.assertTrue(all(
                        isinstance(axis, Replicated) for axis in state.layout))
                    self.assertIsNone(state.value)
                    self.assertIsNone(state.workload)

    def test_aux_entries_inherit_the_request_owner_endpoint(self):
        analysis = evaluate_step(
            _resolved(GEMMA),
            ParallelConfig(
                tp_attention=8, tp_dense_ffn=8, dp_attention=2),
            phase="prefill",
            **_workload("prefill")).as_deployment().checkpoint()
        for stack in analysis.program.stacks:
            if stack.name not in {"vision", "video", "audio"}:
                continue
            first = stack.stages[0].roots[0]
            state = analysis.states.values[first.uid]
            self.assertIsInstance(state.layout[0], Block)
            self.assertEqual(state.layout[0].group.id, "dp_attention")
            self.assertTrue(all(isinstance(axis, Replicated)
                                for axis in state.layout[1:]))

    def test_every_gemma_axis_projects_to_the_local_shape_face(self):
        for phase, (_step, analysis) in self.cases.items():
            local_face = physical_shapes(analysis)
            compared = 0
            for uid, state in analysis.states.values.items():
                global_shape = analysis.shapes.values[uid].out or ()
                local_shape = local_face.values[uid].out or ()
                self.assertEqual(len(state.layout), len(global_shape))
                self.assertEqual(len(local_shape), len(global_shape))
                for layout, whole, local in zip(
                        state.layout, global_shape, local_shape, strict=True):
                    self.assertEqual(
                        physical_local_extent(layout, int(value_of(whole))),
                        int(value_of(local)))
                    compared += 1
            self.assertGreater(compared, 150, phase)

    def test_all_gemma_weight_endpoints_are_declared(self):
        for phase, (_step, analysis) in self.cases.items():
            rows = weight_endpoints.instantiate_analysis(
                analysis, weight_endpoints.placement_context(analysis))
            with self.subTest(phase=phase):
                self.assertEqual(len(rows), 40)

    def test_model_outputs_are_complete_and_non_routed(self):
        for phase, (_step, analysis) in self.cases.items():
            outputs = [
                node for node in analysis.program.nodes()
                if node.boundary_role == "output_head"]
            self.assertEqual(len(outputs), 1, phase)
            state = analysis.states.values[outputs[0].uid]
            self.assertIsNone(state.value)
            self.assertIsNone(state.workload)
            (endpoint,) = [
                record for record in analysis.redistributes
                if record.input_index == -3]
            self.assertEqual(endpoint.node, outputs[0].uid)
            self.assertEqual(endpoint.collectives,
                             (("tp_attention", "all_gather"),))
            self.assertTrue(all(isinstance(axis, Replicated)
                                for axis in endpoint.required.layout))


class TestPersistentCacheEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolved = _resolved(LLAMA)

    def _propagate(self, parallel):
        return evaluate_step(
            self.resolved, parallel, phase="prefill",
            **_workload("prefill")).analysis

    def test_cp_cache_write_is_two_explicit_all_gathers(self):
        analysis = self._propagate(ParallelConfig(
            tp_attention=4, tp_dense_ffn=4, cp=2))
        cached = cache_mod.cached_node_uids(analysis.program)
        from backend.arch.eval.transition import RedistributeSite
        records = [record for record in analysis.redistributes
                   if record.site == RedistributeSite.INTERNAL_OUT
                   and record.collectives == (("cp", "all_gather"),)]
        # Gather fires on the primitive that holds the axes. Identity is
        # the declared child (`v_split_heads`); the permute inside is
        # where the CP cut is published whole.
        leaves = set()
        for uid in cached:
            node = analysis.program.by_uid(uid)
            leaves.add(
                analysis.program.output_leaf(node)[0].uid
                if node.children else uid)
        self.assertEqual({record.node for record in records}, leaves)
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record.producer, record.node)
            self.assertIsInstance(record.arrived.layout[2], Block)
            self.assertIsInstance(record.required.layout[2], Replicated)
            self.assertIsNone(record.required.value)
        self.assertFalse(any(
            record.input_index == -2 for record in analysis.redistributes))

    def test_pcp_same_owner_map_needs_no_endpoint_redistribute(self):
        analysis = self._propagate(ParallelConfig(
            tp_attention=4, tp_dense_ffn=4, cp=2, dcp=2))
        self.assertFalse(any(
            record.input_index == -2 for record in analysis.redistributes))
        for uid in cache_mod.cached_node_uids(analysis.program):
            state = analysis.states.values[uid]
            _required, collectives = analysis.rule.cache_write_endpoint(
                analysis.program.by_uid(uid), state,
                analysis.shapes.values[uid])
            self.assertEqual(collectives, ())
            self.assertIsInstance(state.layout[2], Replicated)

    def test_whole_to_dcp_is_a_zero_network_local_slice(self):
        analysis = self._propagate(ParallelConfig(
            tp_attention=4, tp_dense_ffn=4, cp=2, dcp=2))
        uid = next(iter(cache_mod.cached_node_uids(analysis.program)))
        original = analysis.states.values[uid]
        whole = Replicated(scope=analysis.rule.universe)
        synthetic = propagation.EdgeState(tuple(
            whole for _axis in original.layout))
        required, collectives = analysis.rule.cache_write_endpoint(
            analysis.program.by_uid(uid), synthetic,
            analysis.shapes.values[uid])
        self.assertIsInstance(required.layout[2], Block)
        self.assertEqual(required.layout[2].group.id, "dcp")
        self.assertEqual(collectives, ())


class TestFusionHandoff(unittest.TestCase):
    def test_rebase_does_not_migrate_layout_state_or_redistributes(self):
        analysis = evaluate_step(
            _resolved(GEMMA),
            ParallelConfig(tp_attention=8, tp_dense_ffn=8),
            phase="prefill", **_workload("prefill")).analysis
        fused, rewrite = Framework.of(
            get_binding("vllm"), FUSION_SCHEMES).rewrite(
            analysis.program, analysis)
        self.assertFalse(rewrite.is_identity())
        rebased = rebase_analysis(analysis, fused, rewrite)

        self.assertIs(rebased.states, analysis.states)
        self.assertIs(rebased.port_states, analysis.port_states)
        self.assertIs(rebased.redistributes, analysis.redistributes)
        self.assertFalse(set(rewrite.origins) & set(rebased.states.values))
        self.assertTrue(set(rewrite.dropped) <= set(rebased.states.values))


if __name__ == "__main__":
    unittest.main()
