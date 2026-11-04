"""Small composition oracles for layout's cross-layer contracts.

The snapshot lane proves that known graphs run end to end.  These tests pin
the model-independent compositions that snapshots exposed only incidentally:
SSA fanout, trailing-axis broadcasting, semantic weight-axis identity,
declared block outputs, and factored row-major ownership.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

from backend.arch.eval.dim import Dim
from backend.arch.eval.program import BlockTemplate
from backend.arch.eval.transition import RedistributeSite
from backend.arch.layout.groups import ProcessGroupFamily
from backend.arch.layout.layout import (
    flattened,
    local_index,
    owner_ranks,
    physical_local_extent,
    replicated,
    sharded,
    slice_axis,
)
from backend.arch.layout.materialize_comm import (
    FlowRef,
    PayloadRole,
    PhysicalCollective,
    Placement as EventPlacement,
    RewriteTarget,
    _coalesce_events,
)
from backend.arch.layout.materializer import (
    consumer_of, consumer_roots, family_geometry,
)
from backend.arch.layout.linear import elementwise
from backend.arch.layout.value import SUM, Partial
from backend.arch.layout.propagation import (
    EdgeState,
    LayoutPropagation,
    _WeightPlacement,
)
from backend.arch.spec_tree import BlockMeta


SCOPE8 = frozenset(range(8))
REP8 = replicated(SCOPE8)
X2 = ProcessGroupFamily("x", ((0, 1), (2, 3), (4, 5), (6, 7)))
Y2 = ProcessGroupFamily("y", ((0, 2), (1, 3), (4, 6), (5, 7)))
BX = sharded(X2)
BY = sharded(Y2)


def _rule(*, weights=None, request_family=None):
    return LayoutPropagation(
        parallel=None,
        phase="prefill",
        families={},
        universe=SCOPE8,
        weights=weights or {},
        env={},
        request_family=request_family,
    )


class TestPendingFanoutMaterialization(unittest.TestCase):
    @staticmethod
    def _event(event_id, target, key, *, producer=10, port="out", count=1,
               placement=EventPlacement.BEFORE, anchor=None, rewrites="auto"):
        anchor = target if anchor is None else anchor
        if rewrites == "auto":
            rewrites = (RewriteTarget(target, 0),)
        return PhysicalCollective(
            event_id=event_id,
            site=RedistributeSite.EDGE,
            op="all_reduce",
            source=FlowRef(producer, port),
            source_event=None,
            rewrites=rewrites,
            placement_uid=anchor,
            placement=placement,
            attribution_uid=anchor,
            report_anchor_uid=anchor,
            tree_anchor_event=None,
            count_uid=anchor,
            count=count,
            world="tp",
            group_size=2,
            base_stride=1,
            flow_in_shape=(8,),
            collective_in_shape=(8,),
            collective_out_shape=(8,),
            flow_out_shape=(8,),
            payload_shape=(8,),
            payload_role=PayloadRole.FLOW_BUFFER,
            payload_dtype_bytes=2.0,
            payload_bytes=16.0,
            bytes_per_rank=16.0 * count,
            dedup_key=key,
        )

    @staticmethod
    def _by_name(event_id, key, *, count):
        """A by-name reader: `_record_topology` settles it after the
        producer with NO rewrite target."""
        return TestPendingFanoutMaterialization._event(
            event_id, 0, key, count=count, anchor=10,
            placement=EventPlacement.AFTER, rewrites=(),
        )

    def test_one_pending_ssa_transition_is_charged_once_across_consumers(self):
        events = (
            self._event("first", 20, (10, "out", "sum")),
            self._event("second", 21, (10, "out", "sum")),
        )
        merged = _coalesce_events(events, {20: 1, 21: 0})

        self.assertEqual(len(merged), 1)
        self.assertEqual(
            merged[0].rewrites,
            (RewriteTarget(20, 0), RewriteTarget(21, 0)),
        )
        self.assertEqual(merged[0].placement_uid, 21)

    def test_a_by_node_and_a_by_name_reader_settle_once_before_the_reader(self):
        """One value read by a ref (rewritten) and by a stage's name (no
        ref): both lowered the same collective at the same count, so the
        merge keeps the ref's rewrite, grafts before that reader, and names
        the producer."""
        events = (
            self._event("by-node", 20, (10, "out", "sum"), count=4),
            self._by_name("by-name", (10, "out", "sum"), count=4),
        )
        (event,) = _coalesce_events(events, {10: 0, 20: 1})
        self.assertEqual(event.count, 4)
        self.assertIs(event.placement, EventPlacement.BEFORE)
        self.assertEqual(event.placement_uid, 20)
        self.assertEqual(event.attribution_uid, 10)
        self.assertEqual(event.count_uid, 10)
        self.assertEqual(event.rewrites, (RewriteTarget(20, 0),))

    def test_the_same_merge_in_either_record_order(self):
        forward = _coalesce_events(
            (self._event("by-node", 20, (10, "out", "sum"), count=4),
             self._by_name("by-name", (10, "out", "sum"), count=4)),
            {10: 0, 20: 1})
        backward = _coalesce_events(
            (self._by_name("by-name", (10, "out", "sum"), count=4),
             self._event("by-node", 20, (10, "out", "sum"), count=4)),
            {10: 0, 20: 1})
        stripped = [
            replace(event, event_id="") for event in (forward + backward)
        ]
        self.assertEqual(stripped[0], stripped[1])

    def test_two_by_name_readers_settle_after_the_producer(self):
        events = (
            self._by_name("a", (10, "out", "sum"), count=4),
            self._by_name("b", (10, "out", "sum"), count=4),
        )
        (event,) = _coalesce_events(events, {10: 0})
        self.assertIs(event.placement, EventPlacement.AFTER)
        self.assertEqual(event.placement_uid, 10)
        self.assertEqual(event.count, 4)
        self.assertEqual(event.rewrites, ())

    def test_readers_billing_one_value_differently_raise(self):
        """The count is the value's, settled before the merge; two readers
        disagreeing on it is a lowering bug, not something to reconcile."""
        events = (
            self._event("by-node", 20, (10, "out", "sum"), count=4),
            self._by_name("by-name", (10, "out", "sum"), count=2),
        )
        with self.assertRaises(ValueError):
            _coalesce_events(events, {10: 0, 20: 1})

    def test_a_real_inconsistency_still_raises(self):
        left = self._event("left", 20, (10, "out", "sum"))
        right = self._event("right", 21, (10, "out", "sum"))
        right = replace(right, payload_bytes=32.0)
        with self.assertRaises(ValueError):
            _coalesce_events((left, right), {20: 0, 21: 1})

    def test_distinct_values_and_nonfinalizing_edges_are_not_coalesced(self):
        events = (
            self._event("x", 20, (10, "out", "x")),
            self._event("other-producer", 21, (11, "out", "x"), producer=11),
            self._event("y", 22, (10, "out", "y")),
            self._event("gather-a", 23, None),
            self._event("gather-b", 24, None),
        )

        merged = _coalesce_events(events, {uid: uid for uid in range(20, 25)})
        self.assertEqual([event.event_id for event in merged], [
            "x", "other-producer", "y", "gather-a", "gather-b"
        ])

    def test_named_outputs_from_one_producer_are_distinct_ssa_values(self):
        events = (
            self._event("out", 20, (10, "out", "sum")),
            self._event(
                "stats", 21, (10, "stats", "sum"), port="stats"
            ),
        )

        merged = _coalesce_events(events, {20: 0, 21: 1})
        self.assertEqual([event.event_id for event in merged], ["out", "stats"])


class TestPhysicalEventTopology(unittest.TestCase):
    def test_internal_output_consumer_is_selected_by_named_port(self):
        source = SimpleNamespace(
            uid=1, display_path="source", inputs_obj=(), children=()
        )

        def consumer(uid, port):
            return SimpleNamespace(
                uid=uid,
                display_path=f"consumer_{port}",
                inputs_obj=(SimpleNamespace(src=source, port=port),),
                children=(),
            )

        primary = consumer(2, "")
        stats = consumer(3, "stats")

        class Program:
            def nodes(self):
                return (source, primary, stats)

            def by_uid(self, uid):
                return {node.uid: node for node in self.nodes()}[uid]

        program = Program()
        self.assertEqual(consumer_of(program, source.uid, "out"), 2)
        self.assertEqual(consumer_of(program, source.uid, "stats"), 3)

    def test_sibling_readers_anchor_at_their_shared_parent(self):
        source = SimpleNamespace(
            uid=1, display_path="permute", inputs_obj=(), children=(),
            parent=None,
        )
        ffn = SimpleNamespace(
            uid=2, display_path="ffn", inputs_obj=(), children=(),
            parent=None,
        )
        left = SimpleNamespace(
            uid=3, display_path="gate_proj", children=(), parent=ffn,
            inputs_obj=(SimpleNamespace(src=source, port="out"),),
        )
        right = SimpleNamespace(
            uid=4, display_path="up_proj", children=(), parent=ffn,
            inputs_obj=(SimpleNamespace(src=source, port="out"),),
        )
        ffn.children = (left, right)

        class Program:
            def nodes(self):
                return (source, ffn, left, right)

            def by_uid(self, uid):
                return {node.uid: node for node in self.nodes()}[uid]

        self.assertEqual(consumer_of(Program(), source.uid, "out"), 2)

    def test_flat_sibling_readers_are_one_region(self):
        source = SimpleNamespace(
            uid=1, display_path="permute", inputs_obj=(), children=(),
            parent=None,
        )
        left = SimpleNamespace(
            uid=2, display_path="w1", children=(), parent=None,
            inputs_obj=(SimpleNamespace(src=source, port="out"),),
        )
        right = SimpleNamespace(
            uid=3, display_path="w3", children=(), parent=None,
            inputs_obj=(SimpleNamespace(src=source, port="out"),),
        )

        class Program:
            def nodes(self):
                return (source, left, right)

            def by_uid(self, uid):
                return {node.uid: node for node in self.nodes()}[uid]

        program = Program()
        self.assertEqual(consumer_of(program, source.uid, "out"), 2)
        self.assertEqual(consumer_roots(program, source.uid, "out"), (2, 3))

    def test_readers_in_two_regions_still_fail(self):
        source = SimpleNamespace(
            uid=1, display_path="permute", inputs_obj=(), children=(),
            parent=None,
        )
        a = SimpleNamespace(
            uid=10, display_path="block_a", children=(), parent=None,
            inputs_obj=(),
        )
        b = SimpleNamespace(
            uid=20, display_path="block_b", children=(), parent=None,
            inputs_obj=(),
        )
        left = SimpleNamespace(
            uid=11, display_path="a.gate", children=(), parent=a,
            inputs_obj=(SimpleNamespace(src=source, port="out"),),
        )
        right = SimpleNamespace(
            uid=21, display_path="b.up", children=(), parent=b,
            inputs_obj=(SimpleNamespace(src=source, port="out"),),
        )

        class Program:
            def nodes(self):
                return (source, a, b, left, right)

            def by_uid(self, uid):
                return {node.uid: node for node in self.nodes()}[uid]

        with self.assertRaisesRegex(ValueError, "exactly one region"):
            consumer_of(Program(), source.uid, "out")

    def test_wrapper_consumer_collapses_nested_reader(self):
        source = SimpleNamespace(
            uid=1, display_path="permute", inputs_obj=(), children=()
        )
        inner = SimpleNamespace(
            uid=3, display_path="expert.gate", children=(),
            inputs_obj=(SimpleNamespace(src=source, port="out"),),
        )
        wrapper = SimpleNamespace(
            uid=2, display_path="expert", children=(inner,),
            inputs_obj=(SimpleNamespace(src=source, port="out"),),
        )

        class Program:
            def nodes(self):
                return (source, wrapper, inner)

            def by_uid(self, uid):
                return {node.uid: node for node in self.nodes()}[uid]

        program = Program()
        self.assertEqual(consumer_of(program, source.uid, "out"), 2)

    def test_physical_stride_uses_rank_membership_not_layout_coordinate_order(self):
        joined = ProcessGroupFamily(
            "join", ((0, 2, 4, 6, 1, 3, 5, 7),)
        )
        self.assertEqual(family_geometry(joined), (8, 1))


class TestTrailingBroadcastComposition(unittest.TestCase):
    node = SimpleNamespace(op="mul", uid=1, display_path="probe.mul")

    def test_equal_extents_align_by_trailing_position_not_size(self):
        # H and S are both extent 8 but carry different owner maps.  A rank-2
        # operand aligns with S,D; searching for an axis by extent would pick
        # H and silently exchange the two semantic dimensions.
        full = EdgeState((REP8, BX, BY, REP8))
        by_sequence = EdgeState((BY, REP8))
        required, produced = elementwise(_rule(), 
            self.node,
            [full, by_sequence],
            ((2, 8, 8, 16), (8, 16)),
        )
        self.assertEqual(required, [full, by_sequence])
        self.assertEqual(produced, full)

        by_heads = EdgeState((BX, REP8))
        with self.assertRaisesRegex(ValueError, "alignment law"):
            elementwise(_rule(), 
                self.node,
                [full, by_heads],
                ((2, 8, 8, 16), (8, 16)),
            )

    def test_a_pending_size_one_factor_operand_rides_the_broadcast(self):
        # mul's law folds a pending sum by an invariant factor in either
        # operand; broadcasting the pending one along a size-one axis copies
        # values, which commutes with the sum, so the product stays pending.
        pending = Partial(reducer=SUM, group=X2)
        gate = EdgeState((REP8, REP8, REP8), pending)
        full = EdgeState((REP8, REP8, REP8))
        _required, produced = elementwise(_rule(),
            self.node,
            [gate, full],
            ((2, 8, 1), (2, 8, 16)),
        )
        self.assertEqual(produced.value, pending)
        self.assertEqual(produced.layout, full.layout)

    def test_add_refuses_a_pending_sum_beside_a_complete_operand(self):
        # add folds only matching pending sums; a complete operand next to a
        # pending one means settling went wrong, and a silent pending output
        # would drop a reduction.
        add = SimpleNamespace(op="add", uid=2, display_path="probe.add")
        pending = EdgeState((REP8, REP8), Partial(reducer=SUM, group=X2))
        whole = EdgeState((REP8, REP8))
        with self.assertRaisesRegex(ValueError, "pending sum beside"):
            elementwise(_rule(), add, [pending, whole], ((8, 16), (8, 16)))
        # Its law has no factor either, so a pending operand stays barred
        # from the broadcast side even when both sums match.
        both = EdgeState((REP8, REP8, REP8), pending.value)
        with self.assertRaisesRegex(ValueError, "broadcast parameter"):
            elementwise(_rule(), add, [both, both],
                        ((2, 8, 16), (2, 8, 1)))

    def test_rank3_parameter_broadcasts_over_an_inner_size_one_axis(self):
        full = EdgeState((REP8, BX, BY, REP8))
        parameter = EdgeState((BX, REP8, REP8))
        required, produced = elementwise(_rule(), 
            self.node,
            [full, parameter],
            ((2, 8, 8, 16), (8, 1, 16)),
        )
        self.assertEqual(required, [full, parameter])
        self.assertEqual(produced, full)

        sharded_singleton = EdgeState((BX, BY, REP8))
        with self.assertRaisesRegex(ValueError, "size-one broadcast.*sharded"):
            elementwise(_rule(), 
                self.node,
                [full, sharded_singleton],
                ((2, 8, 8, 16), (8, 1, 16)),
            )

    def test_same_rank_unit_axis_takes_the_nonunit_owners(self):
        parameter = EdgeState((REP8, BY, REP8))
        activation = EdgeState((BX, BY, REP8))

        required, produced = elementwise(_rule(), 
            self.node,
            [parameter, activation],
            ((1, 8, 16), (8, 8, 16)),
        )

        self.assertEqual(required, [parameter, activation])
        self.assertEqual(produced.layout, activation.layout)

    def test_output_layout_is_composed_axis_by_axis(self):
        rows = EdgeState((BX, REP8))
        columns = EdgeState((REP8, BY))

        _required, produced = elementwise(_rule(), 
            self.node,
            [rows, columns],
            ((8, 1), (1, 8)),
        )

        self.assertEqual(produced.layout, (BX, BY))

    def test_same_rank_sharded_singleton_still_refuses(self):
        sharded_unit = EdgeState((BX, BY, REP8))
        activation = EdgeState((BX, BY, REP8))
        with self.assertRaisesRegex(ValueError, "size-one broadcast.*sharded"):
            elementwise(_rule(), 
                self.node,
                [sharded_unit, activation],
                ((8, 1, 16), (8, 8, 16)),
            )

    def test_all_unit_axis_cannot_preserve_a_shared_shard(self):
        sharded_unit = EdgeState((BX,))
        with self.assertRaisesRegex(
                ValueError, "size-one broadcast.*unrepresentable shard"):
            elementwise(_rule(), 
                self.node,
                [sharded_unit, sharded_unit],
                ((1,), (1,)),
            )

    def test_all_unit_axis_can_preserve_an_explicit_padded_owner(self):
        padded_unit = EdgeState((sharded(X2, pad_to=1),))

        _required, produced = elementwise(_rule(), 
            self.node,
            [padded_unit, padded_unit],
            ((1,), (1,)),
        )

        self.assertEqual(produced, padded_unit)

    def test_all_unit_axis_can_preserve_a_declared_logical_owner(self):
        logical_owner = EdgeState((BX,))
        batch = (Dim.symbol("B", 1),)

        _required, produced = elementwise(_rule(request_family=X2), 
            self.node,
            [logical_owner, logical_owner],
            (batch, batch),
        )

        self.assertEqual(produced, logical_owner)

    def test_all_unit_logical_owner_must_be_unambiguous_across_peers(self):
        sharded_unit = EdgeState((BX,))
        batch = (Dim.symbol("B", 1),)
        for other in ((Dim.symbol("gate", 1),), (1,)):
            for shapes in ((batch, other), (other, batch)):
                with self.subTest(shapes=str(shapes)), self.assertRaisesRegex(
                        ValueError,
                        "size-one broadcast.*unrepresentable shard"):
                    elementwise(_rule(request_family=X2), 
                        self.node,
                        [sharded_unit, sharded_unit],
                        shapes,
                    )

    def test_derived_batch_expression_is_not_a_logical_owner_axis(self):
        sharded_unit = EdgeState((BX,))
        derived_batch = (Dim.symbol("B", 1) ** 2,)

        with self.assertRaisesRegex(
                ValueError, "size-one broadcast.*unrepresentable shard"):
            elementwise(_rule(request_family=X2), 
                self.node,
                [sharded_unit, sharded_unit],
                (derived_batch, derived_batch),
            )


class TestSemanticWeightAxisIdentity(unittest.TestCase):
    def test_same_sized_physical_axes_remain_addressable_by_slot(self):
        node = SimpleNamespace(uid=7, display_path="probe.weight")
        placement = _WeightPlacement(
            axes=(BX, BY),
            slots={"head": BX, "sequence": BY},
        )
        rule = _rule(weights={node.uid: placement})

        self.assertEqual(rule.weight(node, "head"), BX)
        self.assertEqual(rule.weight(node, "sequence"), BY)
        self.assertEqual(placement.axes, (BX, BY))

    def test_repeated_semantic_slot_refuses_instead_of_guessing(self):
        node = SimpleNamespace(uid=8, display_path="probe.ambiguous")
        placement = _WeightPlacement(
            axes=(BX, BY),
            slots={},
            ambiguous_slots=frozenset({"width"}),
        )
        rule = _rule(weights={node.uid: placement})

        with self.assertRaisesRegex(ValueError, "occurs on multiple axes"):
            rule.weight(node, "width")


class TestDeclaredBlockOutput(unittest.TestCase):
    def test_declared_forward_root_wins_over_a_later_side_root(self):
        forward = SimpleNamespace(name="forward")
        metrics = SimpleNamespace(name="metrics")
        template = BlockTemplate(
            meta=BlockMeta(template_name="probe", count=1, stack_name="layers",
                           outputs=("forward",)),
            roots=(forward, metrics),
        )

        self.assertEqual(template.output_roots(), (forward,))


class TestFactoredRowMajorOwnership(unittest.TestCase):
    def test_flattened_owner_and_local_index_are_factor_compositions(self):
        axis = flattened((BX, BY), (4, 4))
        for index in range(16):
            outer, inner = divmod(index, 4)
            expected_owners = owner_ranks(BX, outer, 4, SCOPE8) & owner_ranks(
                BY, inner, 4, SCOPE8
            )
            self.assertEqual(owner_ranks(axis, index, 16, SCOPE8), expected_owners)
            for rank in expected_owners:
                expected_local = (
                    local_index(BX, rank, outer, 4, SCOPE8)
                    * physical_local_extent(BY, 4)
                    + local_index(BY, rank, inner, 4, SCOPE8)
                )
                self.assertEqual(
                    local_index(axis, rank, index, 16, SCOPE8), expected_local
                )

    def test_factored_axis_rejects_extent_drift_and_implicit_subslicing(self):
        axis = flattened((BX, BY), (4, 4))
        with self.assertRaisesRegex(ValueError, "multiply to 16"):
            physical_local_extent(axis, 8)
        with self.assertRaisesRegex(ValueError, "factor-aware index map"):
            slice_axis(axis, 0, 8, 16)


if __name__ == "__main__":
    unittest.main()
