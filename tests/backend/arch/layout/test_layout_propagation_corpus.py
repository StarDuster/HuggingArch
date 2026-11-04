"""MLA and MoE routed-zone propagation gates.

Assertions use explicit process-group membership, never only a world label:
DP-attention and EP intentionally use different names for the two ends of the
same routing exchange.
"""

from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.dim import Dim, value_of
from backend.arch.eval.program import BlockStage
from tests.snapshots import shared_loader
from backend.arch.layout import propagation, weight_endpoints
from backend.arch.layout.layout import (
    Block,
    Replicated,
    Segmented,
    owner_ranks,
    physical_local_extent,
    sharded,
)
from backend.arch.layout.groups import ProcessGroupFamily
from backend.arch.layout.physical import (
    physical_kernel_shapes,
    physical_shapes,
)
from backend.arch.layout.physical_cost import physical_flops
from backend.arch.layout.value import Partial, SUM
from backend.arch.eval.storage import is_expert_axis
from backend.inference.parallel_config import ParallelConfig
from backend.inference.step_eval import evaluate_step
from backend.arch.spec_tree.nodes import SpecIRNode


def _probe(*, op, uid, display_path, axes=None, weight_axes=None):
    """A real `SpecIRNode` stand-in for the layout rule (which the walk always
    hands a real node). `display_path` is derived from `name` when the node has
    no parent, so it round-trips; `inputs_obj` / `weight_copies` / `parent`
    default to the empty/None values the rule reads."""
    return SpecIRNode(name=display_path, op=op, is_component=False, uid=uid,
                      axes=axes, weight_axes=weight_axes)


SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"
PHASES = ("prefill", "decode")

QWEN_OPS = {
    # The two residual adds. With no shared expert the MoE's output is its
    # `combine`: `moe_out` would add nothing to it, and is not built.
    "add": 2, "attention_apply": 1, "attention_score": 1, "combine": 1,
    "embedding": 1, "gather": 1, "gqa_qknorm_attention": 1,
    "head_merge": 1, "head_split": 3, "linear": 9, "moe_ffn": 1,
    "mul": 1, "permute": 1, "rmsnorm": 5, "rope": 2,
    "router_softmax_topk": 1, "silu": 1, "softmax": 1,
    "softmax_act": 1, "gated_experts": 1, "topk": 1, "transpose": 4,
    "view_merge": 1, "view_split": 3,
    # Qwen3-30B-A3B's config sets norm_topk_prob, so the router renormalizes
    # its gathered top-k weights (Qwen3MoeTopKRouter.forward).
    "normalize_sum_one": 1,
    # The sampled-head splice (transforms.splice_sampled_head) inserts one
    # zero-FLOP `sample` view before the output head — prefill's lm_head runs
    # on the last position only. Present in every phase.
    "sample": 1,
}

DEEPSEEK_PREFILL_OPS = {
    "add": 10, "attention_apply": 3, "attention_score": 3,
    "broadcast": 3, "combine": 2, "concat": 7, "embedding": 2,
    "gather": 2, "group_mask_apply": 2, "group_top2_sum": 2,
    "head_broadcast": 3, "head_merge": 3, "head_split": 6,
    "linear": 35, "mla_attention": 3,
    "moe_ffn_gate_routing_bias": 2, "mul": 5,
    "normalize_sum_one": 2, "parameter_vector": 2, "permute": 2,
    "pre_norm__attn-mla_attention_ffn-moe_ffn_gate_routing_bias_norm-rmsnorm": 1,
    "rmsnorm": 16, "rope": 3, "rope_emb": 3, "router_noaux_tc": 2,
    "scale_const": 2, "sigmoid": 2, "silu": 5, "slice": 18,
    "softmax": 3, "swiglu_ffn": 3, "gated_experts": 2, "topk": 4,
    "transpose": 12,
    "view_merge": 3, "view_split": 6,
    # The non-absorbed MLA reads its two cached streams (the KV latent and
    # k_pe) through explicit `kv_cache_view` nodes: two per MLA attention.
    "kv_cache_view": 6,
    # One zero-FLOP `sample` view before each output head (splice_sampled_head):
    # the lm_head and the MTP layer's private `shared_head.head`.
    # DEEPSEEK_DECODE_OPS inherits them via the spread below.
    "sample": 2,
}

DEEPSEEK_DECODE_OPS = {
    **DEEPSEEK_PREFILL_OPS,
    "broadcast": 6,
    "head_broadcast": 6,
    "head_split": 3,
    "linear": 32,
    "per_head_linear": 6,
    "rope": 3,
    "rope_emb": 3,
    "slice": 12,
    "transpose": 12,
    "view_split": 3,
    # The absorbed MLA decode component reads the cache without a view node.
    "kv_cache_view": 0,
    # Absorbed MLA runs kv_b_proj as two per-head GEMMs (q and v absorption)
    # that read one stored tensor: the component declares it once at the
    # file's shape (`kv_b_proj: parameter_tensor`) and the build ties both
    # readers to it. One per MLA attention: dense layer, MoE layer, MTP.
    "parameter_tensor": 3,
}

MODELS = {
    "qwen": {
        "snapshot": "Qwen__Qwen3-30B-A3B.json",
        "ops": {phase: QWEN_OPS for phase in PHASES},
        "route_zones": 1,
        "d_model": 2048,
        "weight_rows": {phase: 15 for phase in PHASES},
    },
    "deepseek": {
        "snapshot": "deepseek-ai__DeepSeek-V3.json",
        "ops": {"prefill": DEEPSEEK_PREFILL_OPS,
                "decode": DEEPSEEK_DECODE_OPS},
        # Main backbone plus the explicitly accounted MTP auxiliary stack.
        "route_zones": 2,
        "d_model": 7168,
        # One row per stored tensor. Decode's absorbed kv_b_proj is one
        # tensor read by two per-head GEMMs, as prefill's single GEMM reads
        # it, so both phases store the same 55.
        "weight_rows": {"prefill": 55, "decode": 55},
    },
}

CASES = {
    "dp_ep": ParallelConfig(ep_moe=8, dp_attention=8),
    "ep": ParallelConfig(ep_moe=8),
    "all": ParallelConfig(
        ep_moe=8, tp_moe_expert=4, tp_attention=4,
        tp_dense_ffn=4, dp_attention=8),
}


def _resolved(snapshot: str):
    payload = json.loads((SNAPSHOTS / snapshot).read_text())
    resolved = shared_loader().match(
        payload["config"], model_id=Path(snapshot).stem.replace("__", "/", 1))
    if resolved is None:
        raise AssertionError(f"snapshot {snapshot} did not resolve")
    return resolved


def _workload(phase: str):
    return dict(batch=1, input_tokens=128, output_tokens=0)


def _family_signature(family):
    return tuple(sorted(tuple(group) for group in family.groups))


def _fully_replicated(axis) -> bool:
    if isinstance(axis, Replicated):
        return True
    return isinstance(axis, Segmented) and all(
        _fully_replicated(child)
        for _offset, _extent, child in axis.segments)


class _Corpus(unittest.TestCase):
    cases = None

    @classmethod
    def setUpClass(cls):
        if _Corpus.cases is not None:
            cls.cases = _Corpus.cases
            return
        _Corpus.cases = {}
        for model, meta in MODELS.items():
            resolved = _resolved(meta["snapshot"])
            for label, parallel in CASES.items():
                for phase in PHASES:
                    workload = _workload(phase)
                    step = evaluate_step(
                        resolved, parallel, phase=phase, **workload)
                    propagated = propagation.propagate_step(step)
                    _Corpus.cases[(model, label, phase)] = (
                        step, propagated)
            if model == "deepseek":
                parallel = ParallelConfig(
                    tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8)
                for phase in PHASES:
                    workload = _workload(phase)
                    step = evaluate_step(
                        resolved, parallel, phase=phase, **workload)
                    _Corpus.cases[(model, "tp", phase)] = (
                        step, propagation.propagate_step(step))
        cls.cases = _Corpus.cases

    def _each(self, labels=("dp_ep", "ep", "all")):
        for (model, label, phase), (step, propagated) in self.cases.items():
            if label not in labels:
                continue
            with self.subTest(model=model, case=label, phase=phase):
                yield model, label, phase, step, propagated


class TestCoverage(_Corpus):
    def test_real_graph_multisets_states_and_weight_endpoints_are_closed(self):
        for model, _label, phase, step, propagated in self._each(("dp_ep",)):
            meta = MODELS[model]
            self.assertIs(step.analysis.program.ir, propagated.program.ir)
            self.assertEqual(
                Counter(propagated.program.by_uid(uid).op
                        for uid in propagated.shapes.values),
                Counter(meta["ops"][phase]))
            self.assertEqual(set(propagated.states.values),
                             set(propagated.shapes.values))
            self.assertEqual(
                set(propagated.port_states),
                {(uid, "out") for uid in propagated.shapes.values})

            endpoint_rows = weight_endpoints.instantiate(step)
            self.assertEqual(len(endpoint_rows), meta["weight_rows"][phase])


class TestPhysicalShapes(_Corpus):
    def test_every_static_axis_and_every_estimated_route_axis_reconciles(self):
        for _model, _label, _phase, _step, propagated in self._each(
                ("dp_ep", "ep", "all", "tp")):
            local_shapes = physical_shapes(propagated)
            compared = estimated = 0
            for uid, shape in propagated.shapes.values.items():
                state = propagated.states.values[uid]
                local = local_shapes.values[uid].out or ()
                global_axes = shape.out or ()
                self.assertEqual(len(state.layout), len(global_axes))
                self.assertEqual(len(local), len(global_axes))
                for axis, (whole, layout, expected) in enumerate(zip(
                        global_axes, state.layout, local, strict=True)):
                    # The rows axis of a routed tensor — the one after its
                    # expert axis (`n_expert_copies`, asked of identity) — is
                    # an exact expected load, the MEAN rows per expert, not
                    # an integer static extent. This branch is also what
                    # keeps small-batch decode honest (e.g. M=1/4 rather
                    # than ceil(M)=1). The two sides of each equality below
                    # come from different places — the walk's shape against
                    # the edge's `ExpectedRoutedLoad` — which is what makes
                    # this a check rather than a restatement.
                    if (state.workload is not None and axis > 0
                            and is_expert_axis(global_axes[axis - 1])):
                        if isinstance(layout, Replicated):
                            shards = 1
                        elif isinstance(layout, Block):
                            shards = layout.num_shards
                        else:
                            self.assertIsInstance(layout, propagation.Product)
                            shards = 1
                            for cut in layout.cuts:
                                shards *= cut.num_shards
                        localized = Fraction(value_of(whole), shards)
                        mean = Fraction(value_of(
                            state.workload.expected_rows_per_expert))
                        self.assertEqual(mean, localized)
                        self.assertEqual(Fraction(value_of(expected)), mean)
                        estimated += 1
                        continue

                    global_extent = Fraction(value_of(whole))
                    expected_extent = Fraction(value_of(expected))
                    if global_extent.denominator != 1:
                        self.assertTrue(_fully_replicated(layout))
                        self.assertEqual(global_extent, expected_extent)
                        compared += 1
                        continue

                    physical_source = int(global_extent)
                    names = frozenset(getattr(whole, "names", ()) or ())
                    if (names == {"B"}
                            and propagated.rule.request_family is not None):
                        # Phase 4 still binds B per attention stream. The
                        # layout already carries the logical DP owner cut, so
                        # the reconciliation globalizes B exactly once.
                        physical_source *= (
                            propagated.rule.request_family.group_size)
                    self.assertEqual(
                        physical_local_extent(layout, physical_source),
                        int(expected_extent))
                    compared += 1
            self.assertGreater(compared, 100)
            if propagated.rule.route_family is not None:
                self.assertGreater(estimated, 0)


class TestRoutedWorkload(_Corpus):
    def test_measure_capacity_and_seam_communication_are_exact(self):
        for model, label, _phase, step, propagated in self._each(
                ("dp_ep", "ep")):
            meta = MODELS[model]
            route = propagated.rule.route_family
            self.assertIsNotNone(route)
            expected_cell_rows = Fraction(
                step.batch * int(value_of(propagated.binding.raw("s_query")))
                * int(value_of(propagated.binding.raw("n_experts_active")))
                * (step.dp_attention if label == "dp_ep" else 1))
            dispatches = [
                uid for uid, state in propagated.states.values.items()
                if propagated.program.by_uid(uid).op == "permute"
                and state.workload is not None]
            self.assertEqual(len(dispatches), meta["route_zones"])
            for uid in dispatches:
                workload = propagated.states.values[uid].workload
                self.assertEqual(workload.load_assumption_id,
                                 "moe_routing_balance")
                self.assertEqual(workload.routed_rows_in_cell,
                                 expected_cell_rows)
                self.assertTrue(workload.is_measure_preserving())
                local_experts = (workload.num_experts
                                 // workload.group.group_size)
                self.assertEqual(
                    workload.expected_rows_on_rank(local_experts),
                    expected_cell_rows / route.group_size)
                self.assertEqual(
                    workload.routed_row_capacity_per_rank,
                    expected_cell_rows / route.group_size)

            route_records = [
                record for record in propagated.redistributes
                if propagated.program.by_uid(record.node).op
                in {"permute", "combine"}]
            if label == "dp_ep":
                self.assertEqual(len(route_records), 2 * meta["route_zones"])
                self.assertTrue(all(
                    record.collectives == (("ep_moe", "all_to_all"),)
                    for record in route_records))
                for record in route_records:
                    evidence = (record.arrived.workload
                                or record.required.workload)
                    self.assertIsNotNone(evidence)
                    # Capacity is the bytes-facing row count. At factor 1 it
                    # is the exact closed-form routed payload.
                    payload_elements = (
                        evidence.routed_row_capacity_per_rank
                        * meta["d_model"])
                    self.assertEqual(
                        payload_elements,
                        expected_cell_rows * meta["d_model"]
                        / route.group_size)
            else:
                self.assertEqual(route_records, [])
                ep_reduces = [
                    record for record in propagated.redistributes
                    if record.collectives == (("ep_moe", "all_reduce"),)]
                self.assertEqual(len(ep_reduces), meta["route_zones"])
                self.assertTrue(all(record.arrived.carries_partial()
                                    for record in ep_reduces))

    def test_hot_expert_factor_scales_only_route_capacity(self):
        resolved = _resolved(MODELS["qwen"]["snapshot"])
        base = evaluate_step(
            resolved, ParallelConfig(ep_moe=8, dp_attention=8),
            phase="prefill", **_workload("prefill")).analysis
        hot = evaluate_step(
            resolved, ParallelConfig(ep_moe=8, dp_attention=8),
            phase="prefill", ep_load_factor=1.5,
            **_workload("prefill")).analysis
        base_load = next(
            state.workload for uid, state in base.states.values.items()
            if base.program.by_uid(uid).op == "permute")
        hot_load = next(
            state.workload for uid, state in hot.states.values.items()
            if hot.program.by_uid(uid).op == "permute")
        self.assertEqual(hot_load.routed_rows_in_cell,
                         base_load.routed_rows_in_cell)
        self.assertEqual(hot_load.expected_rows_per_expert,
                         base_load.expected_rows_per_expert)
        # 1.5 is the hottest EXPERT's load; the rank holding it also holds
        # k - 1 experts that share what is left of the rows.
        e, k = hot_load.num_experts, hot_load.local_experts
        rho = Fraction(3, 2)
        rank = (rho + (k - 1) * (e - rho) / (e - 1)) / k
        self.assertEqual(hot_load.busiest_rank_load_factor, rank)
        self.assertLess(rank, rho)
        self.assertEqual(
            hot_load.routed_row_capacity_per_rank,
            base_load.routed_row_capacity_per_rank * rank)

    def test_orthogonal_cp_cut_survives_and_scales_the_route_cell(self):
        """CP labels route-group copies while DP still forces the exchange."""
        parallel = ParallelConfig(
            cp=2, dp_attention=2, ep_moe=2, tp_moe_expert=2)
        self.assertEqual(parallel.validate(), [])
        base_rule = propagation.LayoutPropagation.of(parallel, "prefill")
        uid = -205
        probe = _probe(
            op="permute", uid=uid, display_path="probe.permute",
            weight_axes=None)
        rule = replace(
            base_rule,
            # The walk's binding on the permute: M = B·s_ctx·k·dp_attention
            # / n_route_targets = 1·128·8·2/128.
            env={uid: {"n_experts": 128, "n_expert_copies": 128,
                       "n_experts_active": 8, "n_route_targets": 128,
                       "M": 16}},
        )
        whole = Replicated(scope=rule.universe)
        token = propagation.EdgeState((
            sharded(rule.request_family),
            sharded(rule.families["cp"]),
            whole,
        ))
        routing = propagation.EdgeState((
            token.layout[0], token.layout[1], whole,
        ))
        required, dispatch = rule.transition(
            probe, [token, routing], False,
            [[1, 128, 2048], [1, 128, 8]])

        # `[n_expert_copies, M, d]`: the experts cut over the route group,
        # the CP token cut preserved on the per-expert rows.
        self.assertEqual(dispatch.layout[0],
                         dispatch.workload.expert_shard_axis)
        self.assertIsInstance(dispatch.layout[1], Block)
        self.assertEqual(dispatch.layout[1].group.id, "cp")
        self.assertEqual(dispatch.workload.routed_rows_in_cell,
                         Fraction(1024))
        self.assertEqual(dispatch.workload.expected_rows_per_expert,
                         Fraction(8))
        self.assertEqual(dispatch.workload.routed_row_capacity_per_rank,
                         Fraction(512))

        # The synthetic invocation's M is global across CP. The retained cut
        # turns it into this route cell's exact physical expected extent.
        old_m = 16
        self.assertEqual(old_m, 16)
        self.assertEqual(
            physical_local_extent(dispatch.layout[1], old_m), 8)

        self.assertEqual(
            rule.internal(required, dispatch, node=probe),
            [("out", token, (("ep_moe", "all_to_all"),))])

        # Carry that localized M through a complete routed zone with no
        # expert-TP cut. This is the successful counterpart to the diagonal
        # rejection below: every expert op sees physical M=8 while phase 4's
        # pre-localized node binding remains the explicitly registered
        # M=16 difference.
        linear_uid = -209
        linear = _probe(
            op="linear", uid=linear_uid,
            display_path="probe.expert_linear", weight_axes=None)
        expert_rule = replace(
            rule,
            weights={**rule.weights,
                     linear_uid: propagation._WeightPlacement(
                         axes=(whole, whole),
                         slots={"in": whole, "out": whole})},
            env={**rule.env, linear_uid: {"bias": False}},
        )
        experts = Dim.symbol("n_expert_copies", 128)
        expert_required, expert = expert_rule.transition(
            linear, [dispatch], False, [[experts, old_m, 2048]])
        self.assertEqual(expert_required, [dispatch])
        self.assertEqual(expert.layout[:2], dispatch.layout[:2])
        self.assertEqual(expert.workload, dispatch.workload)
        self.assertEqual(
            physical_local_extent(expert.layout[1], old_m),
            expert.workload.expected_rows_per_expert)

        combine = _probe(
            op="combine", uid=-213, display_path="probe.combine",
            weight_axes=None)
        combine_required, combined = expert_rule.transition(
            combine, [expert, routing], False,
            [[experts, old_m, 2048], [1, 128, 8]])
        self.assertIsNone(combined.workload)
        self.assertEqual(combined.layout, token.layout)
        self.assertEqual(
            expert_rule.internal(combine_required, combined, node=combine),
            [("in", expert, (("ep_moe", "all_to_all"),))])

    def test_full_cp_dp_ep_expert_tp_gathers_routed_rows(self):
        resolved = _resolved(MODELS["qwen"]["snapshot"])
        parallel = ParallelConfig(
            cp=2, dp_attention=2, ep_moe=2, tp_moe_expert=2)
        analysis = evaluate_step(
            resolved,
            parallel,
            phase="prefill",
            **_workload("prefill"),
        ).analysis
        gathers = [
            record for record in analysis.redistributes
            if ("tp_moe_expert", "all_gather") in record.collectives
        ]
        self.assertTrue(gathers)
        for record in gathers:
            self.assertIsNotNone(record.arrived.workload)
            self.assertIsNotNone(record.required.workload)
            self.assertEqual(record.arrived.workload.row_replication, 1)
            self.assertEqual(record.required.workload.row_replication, 2)
            # The per-expert rows, after the expert axis, are gathered.
            self.assertIsInstance(record.arrived.layout[1], Block)
            self.assertIsInstance(record.required.layout[1], Replicated)

    def test_ulysses_gathers_kv_when_cp_exceeds_kv_heads(self):
        resolved = _resolved(MODELS["qwen"]["snapshot"])
        analysis = evaluate_step(
            resolved,
            ParallelConfig(cp=8, ep_moe=8, cp_scheme="ulysses"),
            phase="prefill",
            **_workload("prefill"),
        ).analysis
        cp_ops = [
            collective
            for record in analysis.redistributes
            for world, collective in record.collectives
            if world == "cp"
        ]
        self.assertIn("all_to_all", cp_ops)
        self.assertIn("all_gather", cp_ops)

    def test_dcp_subgroup_uses_kv_replica_cells(self):
        resolved = _resolved(MODELS["qwen"]["snapshot"])
        analysis = evaluate_step(
            resolved,
            ParallelConfig(
                tp_attention=8,
                tp_dense_ffn=8,
                tp_moe_expert=8,
                dcp=2,
            ),
            phase="decode",
            **_workload("decode"),
        ).analysis
        dcp_ops = [
            collective
            for record in analysis.redistributes
            for world, collective in record.collectives
            if world == "dcp"
        ]
        self.assertIn("all_gather_lse", dcp_ops)
        self.assertIn("all_reduce", dcp_ops)

    def test_owner_inclusion_reads_replica_cells_not_group_names(self):
        rule = self.cases[("qwen", "dp_ep", "prefill")][1].rule
        parent = ProcessGroupFamily("parent", ((0, 1, 2, 3),))
        route = ProcessGroupFamily("route", ((0, 1), (2, 3)))
        universe = frozenset(range(4))
        whole = Replicated(scope=universe)
        replicated_halves = Block(parent, ((0, 1), (2, 3)))
        cut_singletons = sharded(parent)
        probe_rule = replace(
            rule, universe=universe, route_family=route)

        self.assertTrue(probe_rule.route_owner_inclusion(
            propagation.EdgeState((replicated_halves, whole))))
        self.assertFalse(probe_rule.route_owner_inclusion(
            propagation.EdgeState((cut_singletons, whole))))

    def test_replica_cell_cut_survives_dispatch_and_guards_routed_axis(self):
        rule = self.cases[("qwen", "ep", "prefill")][1].rule
        parent = ProcessGroupFamily("parent", ((0, 1, 2, 3),))
        route = ProcessGroupFamily("route", ((0, 1), (2, 3)))
        universe = frozenset(range(4))
        whole = Replicated(scope=universe)
        replicated_halves = Block(parent, ((0, 1), (2, 3)))
        permute = _probe(
            op="permute", uid=-206, display_path="probe.permute",
            weight_axes=None)
        probe_rule = replace(
            rule, universe=universe, route_family=route,
            request_family=None,
            # M = B·s_ctx·k/n_route_targets = 4·8·1/4, as the walk binds it.
            env={-206: {"n_experts": 4, "n_expert_copies": 4,
                        "n_experts_active": 1, "n_route_targets": 4,
                        "M": 8}},
        )
        token = propagation.EdgeState(
            (replicated_halves, whole, whole))
        routing = propagation.EdgeState(
            (replicated_halves, whole, whole))
        required, dispatch = probe_rule.transition(
            permute, [token, routing], False,
            [[4, 8, 16], [4, 8, 1]])
        experts = Dim.symbol("n_expert_copies", 4)

        self.assertEqual(dispatch.layout[0], sharded(route))
        self.assertEqual(dispatch.layout[1], replicated_halves)
        self.assertEqual(dispatch.workload.routed_rows_in_cell, Fraction(16))
        self.assertEqual(dispatch.workload.expected_rows_per_expert,
                         Fraction(4))
        self.assertEqual(probe_rule.internal(required, dispatch, node=permute),
                         [])

        transpose = _probe(
            op="transpose", uid=-207, display_path="probe.transpose",
            axes=(1, 0, 2), weight_axes=None)
        with self.assertRaisesRegex(ValueError,
                                    "routed tensors keep semantic axes"):
            probe_rule.transition(
                transpose, [dispatch], False, [[experts, 8, 16]])
        transpose_tail = _probe(
            op="transpose", uid=-210, display_path="probe.transpose_tail",
            axes=(0, 2, 1), weight_axes=None)
        with self.assertRaisesRegex(ValueError,
                                    "routed tensors keep semantic axes"):
            probe_rule.transition(
                transpose_tail, [dispatch], False, [[experts, 8, 16]])

        # The incoming expert debt is over the same family as the EP sum.
        # It must finalize first instead of trying to form a non-orthogonal
        # joined Partial.
        combine = _probe(
            op="combine", uid=-208, display_path="probe.combine",
            weight_axes=None)
        expert = propagation.EdgeState(
            dispatch.layout, Partial(SUM, route), dispatch.workload)
        combine_required, combined = probe_rule.transition(
            combine, [expert, routing], False,
            [[experts, 8, 16], [4, 8, 1]])
        self.assertIsNone(combine_required[0].value)
        self.assertEqual(combined.value, Partial(SUM, route))
        self.assertEqual(
            probe_rule.lower_diff(expert, combine_required[0]),
            (("route", "all_reduce"),))

    def test_concat_localizes_a_replicated_leading_axis(self):
        rule = self.cases[("qwen", "all", "prefill")][1].rule
        family = rule.families["tp_attention"]
        whole = Replicated(scope=rule.universe)
        cut = sharded(family)
        concat = _probe(
            op="concat", uid=-212, display_path="probe.concat",
            weight_axes=None)
        replicated_input = propagation.EdgeState((whole, whole, whole))
        sharded_input = propagation.EdgeState((cut, whole, whole))

        required, produced = rule.transition(
            concat, [replicated_input, sharded_input], False,
            [[8, 4, 2], [8, 4, 3]])

        self.assertEqual(required[0].layout[0], cut)
        self.assertEqual(required[1], sharded_input)
        self.assertEqual(produced.layout[0], cut)
        self.assertEqual(rule.lower_diff(replicated_input, required[0]), ())


class TestMlaAndFoldLaw(_Corpus):
    def test_mla_splits_real_heads_before_per_head_concat(self):
        prefill = self.cases[("deepseek", "tp", "prefill")][1]
        q_b = [
            (uid, state) for uid, state in prefill.states.values.items()
            if prefill.program.by_uid(uid).display_path.endswith("q_b_proj")]
        self.assertEqual(len(q_b), 3)
        for uid, state in q_b:
            self.assertIsInstance(state.layout[-1], Block)
            prefix = prefill.program.by_uid(uid).display_path.rsplit(".", 1)[0]
            split_uid, split = next(
                (other_uid, other)
                for other_uid, other in prefill.states.values.items()
                if prefill.program.by_uid(other_uid).display_path
                == f"{prefix}.q_split_heads")
            self.assertEqual(state.layout[-1], split.layout[1])
            flat_shape = prefill.shapes.values[uid].out
            split_shape = prefill.shapes.values[split_uid].out
            flat_extent = int(value_of(flat_shape[-1]))
            heads = int(value_of(split_shape[1]))
            per_head = int(value_of(split_shape[-1]))
            for head in (0, heads - 1):
                self.assertEqual(
                    owner_ranks(state.layout[-1], head * per_head,
                                flat_extent, prefill.rule.universe),
                    owner_ranks(split.layout[1], head, heads,
                                prefill.rule.universe))
        q_slices = [
            state for uid, state in prefill.states.values.items()
            if prefill.program.by_uid(uid).display_path.split(
                "__", 1)[0].endswith("q_nope_slice")]
        self.assertEqual(len(q_slices), 3)
        self.assertTrue(all(isinstance(state.layout[1], Block)
                            and isinstance(state.layout[-1], Replicated)
                            for state in q_slices))
        query_concats = [
            state for uid, state in prefill.states.values.items()
            if prefill.program.by_uid(uid).display_path.split(
                "__", 1)[0].endswith("query_concat")]
        self.assertEqual(len(query_concats), 3)
        self.assertTrue(all(isinstance(state.layout[1], Block)
                            and isinstance(state.layout[-1], Segmented)
                            for state in query_concats))

        decode = self.cases[("deepseek", "tp", "decode")][1]
        per_head = [
            state for uid, state in decode.states.values.items()
            if decode.program.by_uid(uid).op == "per_head_linear"]
        self.assertEqual(len(per_head), 6)
        self.assertTrue(all(isinstance(state.layout[1], Block)
                            for state in per_head))
        # Concat keeps its formal segment injections; the attention pairing
        # accepts an all-replicated Segmented axis without globally erasing it.
        latent_concats = [
            state for uid, state in decode.states.values.items()
            if decode.program.by_uid(uid).display_path.split(
                "__", 1)[0].endswith(("q_concat", "kv_concat"))]
        self.assertEqual(len(latent_concats), 6)
        self.assertTrue(all(isinstance(state.layout[-1], Segmented)
                            for state in latent_concats))

    def test_rank4_rope_cost_includes_heads_and_tp_scales(self):
        for phase in PHASES:
            step, propagated = self.cases[("deepseek", "tp", phase)]
            global_analysis = step.analysis
            local_shapes = physical_kernel_shapes(propagated)
            local_flops_layer = physical_flops(propagated, local_shapes)
            found = 0
            for uid, cost in global_analysis.cost.values.items():
                node = global_analysis.program.by_uid(uid)
                if not node.display_path.split("__", 1)[0].endswith(
                        "q_pe_rope"):
                    continue
                global_shape = global_analysis.shapes.values[uid].ins[0]
                local_shape = local_shapes.values[uid].ins[0]
                global_elements = 1
                local_elements = 1
                for extent in global_shape:
                    global_elements *= int(value_of(extent))
                for extent in local_shape:
                    local_elements *= int(value_of(extent))
                global_flops = int(value_of(cost.flops))
                local_flops = int(value_of(local_flops_layer.values[uid]))
                # 3 per rotated element (4 mul + 2 add per pair).
                self.assertEqual(global_flops, 3 * global_elements)
                self.assertEqual(local_flops, 3 * local_elements)
                self.assertEqual(global_flops, 8 * local_flops)
                found += 1
            self.assertEqual(found, 3)

    def test_orthogonal_contraction_debts_join_and_conflicts_settle(self):
        rule = self.cases[("qwen", "dp_ep", "prefill")][1].rule
        left = ProcessGroupFamily("left", ((0, 1), (2, 3)))
        right = ProcessGroupFamily("right", ((0, 2), (1, 3)))
        universe = frozenset(range(4))
        whole = Replicated(scope=universe)
        row_cut = sharded(left)
        uid = -211
        probe = _probe(
            op="linear", uid=uid, display_path="probe.linear",
            weight_axes=None)
        probe_rule = replace(
            rule,
            universe=universe,
            weights={uid: propagation._WeightPlacement(
                axes=(whole, row_cut),
                slots={"in": row_cut, "out": whole})},
            env={uid: {"bias": False}},
        )

        incoming = Partial(SUM, right)
        state = propagation.EdgeState(
            (whole, whole, row_cut), incoming)
        required, produced = probe_rule.transition(
            probe, [state], False, [[1, 1, 4]])
        self.assertEqual(required[0].value, incoming)
        self.assertEqual(produced.value.group.group_size, 4)
        self.assertIn("join(right,left)", produced.value.group.id)

        conflicting = propagation.EdgeState(
            (whole, whole, row_cut), Partial(SUM, left))
        required, produced = probe_rule.transition(
            probe, [conflicting], False, [[1, 1, 4]])
        self.assertIsNone(required[0].value)
        self.assertEqual(produced.value, Partial(SUM, left))

        # The same conflict starting whole finalizes directly into the
        # row-parallel contraction layout with one reduce-scatter.
        conflicting_whole = propagation.EdgeState(
            (whole, whole, whole), Partial(SUM, left))
        required, produced = probe_rule.transition(
            probe, [conflicting_whole], False, [[1, 1, 4]])
        self.assertEqual(required[0].layout[-1], row_cut)
        self.assertIsNone(required[0].value)
        self.assertEqual(produced.value, Partial(SUM, left))
        self.assertEqual(
            probe_rule.lower_diff(conflicting_whole, required[0]),
            (("left", "reduce_scatter"),))

    def test_singleton_aux_block_skips_fixed_point_but_closes_a_partial(self):
        propagated = self.cases[("deepseek", "all", "prefill")][1]
        mtp_stack = next(
            stack for stack in propagated.program.stacks
            if stack.aux_kind == "mtp")
        self.assertEqual(mtp_stack.upstream, ("norm",))
        self.assertEqual(len(mtp_stack.stages), 1)
        mtp_stage = mtp_stack.stages[0]
        self.assertIsInstance(mtp_stage, BlockStage)
        self.assertEqual(mtp_stage.template.count, 1)

        backbone = next(
            stack for stack in propagated.program.stacks
            if stack.aux_kind is None)
        norm_stage = next(stage for stage in backbone.stages
                          if stage.name == "norm")
        inbound = propagated.states.values[norm_stage.roots[-1].uid]
        exit_state = propagated.states.values[mtp_stage.roots[-1].uid]
        self.assertNotEqual(exit_state.finalized(), inbound)

        # Count one has no repeated-region interface to compare, but it is
        # still a stage boundary: a debt leaving it is finalized exactly once.
        pending = propagation.EdgeState(
            inbound.layout,
            Partial(SUM, propagated.rule.families["tp_moe_expert"]),
        )
        settled, collectives = propagated.rule.close_stage(
            pending, interface=None, stage_uid=mtp_stage.roots[0].uid)
        self.assertIsNone(settled.value)
        self.assertEqual(collectives,
                         (("tp_moe_expert", "all_reduce"),))

    def test_tp_partial_survives_both_inverse_route_exchanges(self):
        for model in MODELS:
            propagated = self.cases[(model, "all", "prefill")][1]
            combines = [
                record for record in propagated.redistributes
                if (propagated.program.by_uid(record.node).op == "combine"
                    and record.collectives
                    == (("ep_moe", "all_to_all"),))]
            self.assertEqual(len(combines), MODELS[model]["route_zones"])
            for record in combines:
                self.assertTrue(record.arrived.carries_partial())
                self.assertTrue(record.required.carries_partial())
                self.assertEqual(record.arrived.value,
                                 record.required.value)
                self.assertEqual(record.arrived.value.group.id,
                                 "tp_moe_expert")


class TestIllegalRoutingRefuses(_Corpus):
    def test_nondivisible_expert_collection_and_lost_workload_refuse(self):
        rule = self.cases[("qwen", "dp_ep", "prefill")][1].rule
        whole = Replicated(scope=rule.universe)
        token = propagation.EdgeState(
            (Block(rule.request_family,
                   tuple((i,) for i in range(rule.request_family.group_size))),
             whole, whole))
        routing = propagation.EdgeState(
            (token.layout[0], whole, whole))
        probe = _probe(
            op="permute", uid=-301, display_path="probe.permute",
            weight_axes=None)
        bad_rule = replace(
            rule, env={-301: {
                "n_experts": 10, "n_expert_copies": 10,
                "n_experts_active": 1, "n_route_targets": 10,
                "M": Fraction(128 * rule.request_family.group_size, 10),
            }})
        with self.assertRaisesRegex(ValueError, "experts must divide"):
            bad_rule.transition(
                probe, [token, routing], False,
                [[1, 128, 2048], [1, 128, 8]])

        combine = _probe(
            op="combine", uid=-302, display_path="probe.combine",
            weight_axes=None)
        expert_without_evidence = propagation.EdgeState(
            (whole, whole, whole))
        with self.assertRaisesRegex(ValueError, "lost its ExpectedRoutedLoad"):
            rule.transition(
                combine, [expert_without_evidence, routing], False,
                [[Dim.symbol("n_expert_copies", 64), 1, 2048],
                 [1, 128, 8]])


if __name__ == "__main__":
    unittest.main()
