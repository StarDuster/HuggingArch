"""Physical geometry derived from one layout-carrying analysis."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.analysis import Analysis
from backend.arch.eval.dim import Dim, as_number, value_of
from backend.arch.eval.evaluator import Evaluator, Walk
from backend.arch.eval.layer import Layer
from backend.arch.eval.op_ref import OpRef
from backend.arch.eval.scope import Carried
from backend.arch.eval.transition import InputRequirements
from backend.arch.eval.values import Cost, Shape
from tests.snapshots import shared_loader
from backend.arch.axis_propagation import compute_shape_out
from backend.arch.ops import OP_REGISTRY
from backend.arch.layout.groups import ProcessGroupFamily
from backend.arch.layout.layout import replicated, sharded
from backend.arch.layout.materializer import materialize_comm_events
from backend.arch.layout.physical import (
    local_extent,
    local_node_dims,
    pairs_share,
    physical_kernel_shapes,
    physical_shapes,
)
from backend.arch.layout.physical_cost import (
    _primitive_flops,
    physical_flops,
)
from backend.arch.layout.propagation import EdgeState, LayoutPropagation
from backend.inference.parallel_config import ParallelConfig
from backend.inference.step_eval import evaluate_step


SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"
LANE_COHORT = tuple(json.loads(
    (PROJECT_ROOT / "tests/golden/shardv3_lane_snapshots.json").read_text()
))
TP8 = ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8)


def _resolved(snapshot: str):
    payload = json.loads((SNAPSHOTS / snapshot).read_text())
    resolved = shared_loader().match(
        payload["config"], model_id=Path(snapshot).stem.replace("__", "/", 1))
    assert resolved is not None
    return resolved


def _step(snapshot: str, phase: str):
    return evaluate_step(
        _resolved(snapshot),
        TP8,
        phase=phase,
        batch=1,
        input_tokens=128, output_tokens=0,
    )


def _ints(dims):
    return tuple(int(extent) for extent in dims)


def test_elementwise_shape_uses_standard_trailing_broadcast():
    mul = OP_REGISTRY["mul"]
    channel = Dim.symbol("gate", 32)
    activation = (
        Dim.symbol("B", 2),
        Dim.symbol("s_ctx", 16),
        Dim.symbol("d_model", 32),
    )

    out = compute_shape_out(mul, [(channel,), activation], {})

    assert out == activation
    assert out[-1].expr == activation[-1].expr
    assert compute_shape_out(mul, [activation, (channel,)], {}) == activation
    with pytest.raises(ValueError, match="incompatible extents"):
        compute_shape_out(mul, [(31,), activation], {})


def test_all_unit_broadcast_identity_comes_from_the_widest_operand():
    gate = (Dim.symbol("gate", 1),)
    activation = (
        Dim.symbol("B", 1),
        Dim.symbol("s_ctx", 16),
        Dim.symbol("d_model", 1),
    )

    out = compute_shape_out(OP_REGISTRY["add"], [gate, activation], {})

    assert tuple(dim.expr for dim in out) == tuple(
        dim.expr for dim in activation
    )
    assert compute_shape_out(
        OP_REGISTRY["add"], [activation, gate], {}
    ) == activation


def test_primitive_input_shapes_keep_missing_port_positions():
    refs = tuple(
        SimpleNamespace(src=SimpleNamespace(), port="") for _ in range(2)
    )
    node = SimpleNamespace(inputs_obj=refs)
    live = Shape(out=(Dim.symbol("d_model", 8),))

    ports = Evaluator._primitive_input_shapes(
        node,
        [Carried(), Carried(shape=live)],
    )

    assert ports == [None, list(live.out)]


def test_physical_shapes_use_original_port_index_after_a_missing_input():
    ranks = frozenset(range(2))
    family = ProcessGroupFamily("x", ((0, 1),))
    rule = LayoutPropagation(
        parallel=None,
        phase="prefill",
        families={"x": family},
        universe=ranks,
        weights={},
        env={},
    )
    node = SimpleNamespace(uid=1, display_path="probe.add", children=(),
                           declaration=None)

    class Program:
        @staticmethod
        def by_uid(uid):
            return node if uid == node.uid else None

    program = Program()
    output_state = EdgeState((replicated(ranks),))
    input_state = EdgeState((sharded(family),))
    shape = Shape(out=(8,), ins=((), (8,)))
    analysis = Analysis(
        program=program,
        cost=Layer(program, {node.uid: Cost()}),
        shapes=Layer(program, {node.uid: shape}),
        states=Layer(program, {node.uid: output_state}),
        port_states={(node.uid, "out"): output_state},
        input_states={(node.uid, 0): None, (node.uid, 1): input_state},
        rule=rule,
    )

    local = physical_shapes(analysis).values[node.uid]

    assert local.ins == ((), (4,))


def test_missing_same_rank_input_follows_the_live_output_layout():
    ranks = frozenset(range(2))
    family = ProcessGroupFamily("x", ((0, 1),))
    rule = LayoutPropagation(
        parallel=None,
        phase="prefill",
        families={"x": family},
        universe=ranks,
        weights={},
        env={},
    )
    node = SimpleNamespace(uid=1, display_path="probe.add", children=(),
                           declaration=None)

    class Program:
        @staticmethod
        def by_uid(uid):
            return node if uid == node.uid else None

    program = Program()
    output_state = EdgeState((sharded(family),))
    shape = Shape(out=(8,), ins=((8,), (8,)))
    analysis = Analysis(
        program=program,
        cost=Layer(program, {node.uid: Cost()}),
        shapes=Layer(program, {node.uid: shape}),
        states=Layer(program, {node.uid: output_state}),
        port_states={(node.uid, "out"): output_state},
        input_states={(node.uid, 0): output_state, (node.uid, 1): None},
        rule=rule,
    )

    local = physical_shapes(analysis).values[node.uid]

    assert local.ins == ((4,), (4,))


def test_physical_shapes_project_secondary_ports_and_empty_extents():
    ranks = frozenset(range(2))
    family = ProcessGroupFamily("x", ((0, 1),))
    rule = LayoutPropagation(
        parallel=None,
        phase="prefill",
        families={"x": family},
        universe=ranks,
        weights={},
        env={},
    )
    node = SimpleNamespace(uid=1, display_path="probe.split", children=(),
                           declaration=None)

    class Program:
        @staticmethod
        def by_uid(uid):
            return node if uid == node.uid else None

    program = Program()
    cut = EdgeState((sharded(family),))
    shape = Shape(out=(0,), ports={"live": (8,)})
    analysis = Analysis(
        program=program,
        cost=Layer(program, {node.uid: Cost()}),
        shapes=Layer(program, {node.uid: shape}),
        states=Layer(program, {node.uid: cut}),
        port_states={(node.uid, "out"): cut, (node.uid, "live"): cut},
        rule=rule,
    )

    local = physical_shapes(analysis).values[node.uid]

    assert local.out == (0,)
    assert local.ports["live"] == (4,)


def test_only_the_exact_declared_request_axis_projects_logically():
    ranks = frozenset(range(2))
    family = ProcessGroupFamily("request", ((0, 1),))
    rule = LayoutPropagation(
        parallel=None,
        phase="prefill",
        families={"request": family},
        universe=ranks,
        weights={},
        env={},
        resident_sequence_family=family,
        request_family=family,
    )
    layout = sharded(family)

    singleton = Dim(sp.Symbol("B"), 1)
    local_batch = Dim(sp.Symbol("B"), 6)
    empty_batch = Dim(sp.Symbol("B"), 0)
    assert int(local_extent(singleton, layout, rule)) == 1
    assert int(local_extent(local_batch, layout, rule)) == 6
    assert int(local_extent(empty_batch, layout, rule)) == 0
    assert int(
        local_extent(Dim(sp.Symbol("s_sampled"), 0), layout, rule)
    ) == 0
    for invalid in (
        Dim(sp.Symbol("B"), sp.Rational(3, 2)),
        Dim(sp.Symbol("B"), -1),
    ):
        with pytest.raises(ValueError, match="logical axis owner"):
            local_extent(invalid, layout, rule)
    for ambiguous in (1, Dim(sp.Symbol("B") ** 2, 1)):
        with pytest.raises(ValueError, match="not divisible"):
            local_extent(ambiguous, layout, rule)


def test_consumer_input_state_is_recorded_after_redistribution():
    analysis = _step("meta-llama__Llama-3.2-3B.json", "prefill").analysis
    edge = next(
        edge for edge in analysis.redistributes
        if edge.input_index >= 0
        and analysis.program.by_uid(edge.node).display_path == "post_attn_add__block"
    )

    assert edge.arrived != edge.required
    assert analysis.input_states[(edge.node, edge.input_index)] == edge.required


def test_kernel_requirements_do_not_create_ssa_repairs():
    edge_state = object()
    kernel_state = object()
    output_state = object()
    observed = {}

    class Rule:
        is_trivial = False

        @staticmethod
        def transition(
            node, arrived, uses_weight, in_shapes, **_node_facts
        ):
            assert arrived == [edge_state]
            return InputRequirements(
                edges=(edge_state,), kernels=(kernel_state,)
            ), output_state

        @staticmethod
        def internal(required, produced, *, node):
            observed["required"] = required
            observed["produced"] = produced
            return ()

        @staticmethod
        def lower_diff(*args, **kwargs):
            raise AssertionError("equal SSA edges must not be lowered")

    evaluator = object.__new__(Evaluator)
    evaluator.rule = Rule()
    node = SimpleNamespace(uid=7, inputs_obj=(), display_path="probe.kernel")
    walk = Walk()

    evaluator._place(
        node,
        [Carried(state=edge_state)],
        False,
        walk,
        [(8,)],
        node_dims={},
        stored_tensors=(),
        weight_shape=(),
    )

    assert walk.redistributes == []
    assert walk.input_states[(node.uid, 0)] is kernel_state
    assert observed == {
        "required": (kernel_state,),
        "produced": output_state,
    }


def test_typed_transition_requirements_preserve_consumer_arity():
    requirements = InputRequirements(edges=(), kernels=())

    class Rule:
        is_trivial = False

        @staticmethod
        def transition(
            node, arrived, uses_weight, in_shapes, **_node_facts
        ):
            return requirements, object()

    evaluator = object.__new__(Evaluator)
    evaluator.rule = Rule()
    node = SimpleNamespace(uid=7, inputs_obj=(), display_path="probe.arity")

    with pytest.raises(ValueError, match="0 input states for 1 consumer ports"):
        evaluator._place(
            node,
            [Carried(state=object())],
            False,
            Walk(),
            [(8,)],
            node_dims={},
            stored_tensors=(),
            weight_shape=(),
        )


def test_tp8_kernel_inputs_and_outputs_use_the_same_local_face():
    analysis = _step("meta-llama__Llama-3.2-3B.json", "prefill").analysis
    local = physical_shapes(analysis)
    by_path = {node.display_path: node for node in analysis.program.nodes()}

    logits = local.values[by_path["lm_head"].uid]
    query = local.values[by_path["self_attn.q_proj"].uid]
    output = local.values[by_path["self_attn.o_proj"].uid]

    assert _ints(logits.out) == (1, 1, 16032)
    assert _ints(query.out) == (1, 128, 384)
    assert _ints(output.ins[0]) == (1, 128, 384)
    assert _ints(output.out) == (1, 128, 3072)


def test_cp_sampled_position_is_bound_before_the_output_endpoint():
    step = evaluate_step(
        _resolved("meta-llama__Llama-3.2-3B.json"),
        ParallelConfig(tp_attention=4, tp_dense_ffn=4, cp=2),
        phase="prefill",
        batch=1,
        input_tokens=128, output_tokens=0,
    )
    analysis = step.analysis
    local = physical_shapes(analysis)
    logits = next(
        node for node in analysis.program.nodes() if node.display_path == "lm_head"
    )

    assert analysis.shapes.values[logits.uid].out[1].is_("s_sampled")
    assert _ints(local.values[logits.uid].out) == (1, 1, 32064)
    events = [
        event for event in materialize_comm_events(analysis)
        if event.report_anchor_uid == logits.uid
    ]
    assert all(event.world != "cp" for event in events)
    assert [event.world for event in events] == ["tp_attention"]


def test_tp8_flops_bind_projection_width_from_the_local_output():
    step = _step("meta-llama__Llama-3.2-3B.json", "prefill")
    analysis = step.analysis
    flops = physical_flops(analysis)
    query = next(
        node for node in analysis.program.nodes()
        if node.display_path == "self_attn.q_proj"
    )

    assert int(flops.values[query.uid]) * 8 == int(
        analysis.cost.values[query.uid].flops
    )
    assert set(flops.values) == set(analysis.cost.values)


def test_tp8_flops_record_binds_only_what_the_evaluation_read():
    """q_b_proj's formula reads the merged head axis `n_heads*(…)`, never the
    bare `n_heads`. The record must hold the composite at its per-rank value
    and must NOT carry the stale global `n_heads` the projection once kept."""
    from backend.arch.layout.physical_cost import (
        _physical_flops_and_bindings,
    )

    step = _step("deepseek-ai__DeepSeek-V3.json", "prefill")
    analysis = step.analysis
    values, bindings = _physical_flops_and_bindings(analysis)
    node = next(
        node for node in analysis.program.nodes()
        if node.display_path == "self_attn.q_b_proj"
    )

    record = bindings[node.uid]
    n_heads = sp.Symbol("n_heads")
    composite = n_heads * (sp.Symbol("d_qk_nope") + sp.Symbol("d_qk_rope"))
    assert n_heads not in record
    # 128 heads * 192 / TP8 — the value the per-rank evaluation substituted.
    assert record[composite] == 128 * 192 // 8
    assert int(values[node.uid]) * 8 == int(analysis.cost.values[node.uid].flops)


def test_local_node_dims_answers_bare_factor_symbols_from_facts():
    """`n_heads` exists only as a factor of the merged axis `n_heads*d_head`
    and matches no shape axis. Without facts the global value stands; the
    weight-axis contracts' proven per-rank extent answers it when given."""
    node = SimpleNamespace(display_path="probe")
    n_heads, d_head = sp.symbols("n_heads d_head")
    dims = {
        "out": Dim(n_heads * d_head, 128 * 128),
        "n_heads": Dim(n_heads, 128),
    }
    global_shape = Shape(out=(Dim(n_heads * d_head, 128 * 128),))
    local_shape = Shape(out=(Dim(n_heads * d_head, 16 * 128),))

    projected = local_node_dims(node, dims, global_shape, local_shape,
                                pairs_share=None)
    assert int(projected["out"]) == 16 * 128
    assert int(projected["n_heads"]) == 128       # no facts: global stands

    projected = local_node_dims(
        node, dims, global_shape, local_shape,
        local_symbols={"n_heads": 16}, pairs_share=None)
    assert int(projected["n_heads"]) == 16        # the proven per-rank value


def test_one_symbol_sizes_both_ends_of_a_projection_split_on_one():
    """Qwen3-VL's merger fc1 is `d_vis_merge -> d_vis_merge`, column-split
    (vLLM ColumnParallelLinear): its input stays whole and its output is one
    rank's columns. Each axis keeps its own extent, the bare symbol is no
    per-rank fact of the node, and the linear's `in` follows its input."""
    from backend.arch.eval.transforms import Deployment
    from backend.arch.layout.weight_endpoints import local_symbol_extents
    from tools.corpus_pins.pins.encode_surface import reference_media
    from backend.arch.eval.program import Program
    program = Program.of(_resolved("Qwen__Qwen3-VL-235B-A22B-Instruct.json"))
    analysis = Deployment(program, config=TP8).at(
        batch=1, media=reference_media(program, "one")).checkpoint()
    fc1 = next(node for node in analysis.program.nodes()
               if node.display_path == "visual.merger.linear_fc1")
    global_shape = analysis.shapes.values[fc1.uid]
    local_shape = physical_shapes(analysis).values[fc1.uid]
    assert _ints(local_shape.ins[0])[-1] == 4608
    assert _ints(local_shape.out)[-1] == 4608 // 8
    assert local_symbol_extents(analysis)[fc1.uid]["d_vis_merge"] is None
    projected = local_node_dims(
        fc1, analysis.node_dims.values[fc1.uid], global_shape, local_shape,
        program=analysis.program, pairs_share=None)
    assert (int(projected["in"]), int(projected["out"])) == (4608, 4608 // 8)


def test_a_binding_on_an_ambiguous_symbol_is_refused_not_kept_global():
    """`d` sizes two axes of the node split differently, so the facts hold
    no per-rank value for it. A binding that needs `d` and matches no shape
    axis is refused; keeping the global value would export a whole width as
    one rank's."""
    node = SimpleNamespace(display_path="probe")
    d, n = sp.symbols("d n")
    dims = {"out": Dim(n, 64), "width": Dim(2 * d, 2 * 4608)}
    global_shape = Shape(out=(Dim(n, 64),))
    local_shape = Shape(out=(Dim(n, 8),))
    with pytest.raises(ValueError, match="no one per-rank value"):
        local_node_dims(node, dims, global_shape, local_shape,
                        local_symbols={"d": None}, pairs_share=None)
    projected = local_node_dims(node, {"out": dims["out"]}, global_shape,
                                local_shape, local_symbols={"d": None},
                                pairs_share=None)
    assert int(projected["out"]) == 8


def test_a_ranks_share_of_the_score_pairs_is_exact_in_both_domains():
    """`pairs_share` divides one count by another, so it is a `Rational` —
    an uncut rank's is `1`, a third of the keys is `1/3` — whichever domain
    the two counts arrive in.

    They arrive in both. A decode step's query axis is the literal
    `n_verify`, so its pair counts are plain Python ints and a bare `/` on
    them gave the float `1.0`; prefill's are expressions in `s_input`, where
    sympy's `/` gave an exact `1`. The float then rode into the decode law
    as a coefficient, the walk's own `s_pairs` came out `Float(4224)`, and
    `as_number` published the same count as the string "4224.00000000000"
    on one reader and the integer 4224 on another.
    """
    n_verify, s_kv, s_input = sp.symbols("n_verify s_kv s_input")

    def matrix(queries, keys):
        """A score matrix `(B, heads, queries, keys)`, as the op's output."""
        return Shape(out=(Dim(sp.Symbol("B"), 1), Dim(sp.Symbol("n_heads"), 8),
                          Dim(n_verify, queries), Dim(s_kv, keys)))

    # Decode: one query row, counted as a number, so the pairs are ints.
    assert pairs_share("attention_score", matrix(1, 4224),
                       matrix(1, 4224)) == sp.Integer(1)
    assert pairs_share("attention_score", matrix(1, 4224),
                       matrix(1, 2112)) == sp.Rational(1, 2)
    # An odd cut: the share has a denominator to keep, and keeps it.
    assert pairs_share("attention_score", matrix(1, 4224),
                       matrix(1, 1408)) == sp.Rational(1, 3)
    # Prefill: the same counts as laws in the workload's symbols.
    assert pairs_share("attention_score", matrix(s_input, s_input),
                       matrix(s_input, s_input)) == sp.Integer(1)
    assert pairs_share("attention_score", matrix(s_input, s_input),
                       matrix(s_input, s_input / 3)) == sp.Rational(1, 3)
    # Exact, not merely equal: a float is equal to 1 and cannot say whether
    # the count it scales stayed whole (`dim.as_number`).
    for share in (pairs_share("attention_score", matrix(1, 4224),
                              matrix(1, 4224)),
                  pairs_share("attention_score", matrix(s_input, s_input),
                              matrix(s_input, s_input))):
        assert isinstance(share, sp.Rational), repr(share)
        assert as_number(Dim(sp.Symbol("s_pairs"), 4224 * share)) == 4224


def test_nondivisible_heads_follow_the_propagated_local_width():
    step = _step("Qwen__Qwen2.5-7B.json", "prefill")
    analysis = step.analysis
    shapes = physical_kernel_shapes(analysis)
    flops = physical_flops(analysis, shapes)
    query = next(
        node for node in analysis.program.nodes()
        if node.display_path == "self_attn.q_proj"
    )

    assert _ints(shapes.values[query.uid].out) == (1, 128, 896)
    assert int(flops.values[query.uid]) * 4 == int(
        analysis.cost.values[query.uid].flops
    )


def test_consumer_gather_changes_the_kernel_flops_without_a_model_rule():
    step = _step("MiniMaxAI__MiniMax-M2.7.json", "prefill")
    analysis = step.analysis
    flops = physical_flops(analysis)
    norm = next(
        node for node in analysis.program.nodes()
        if node.display_path == "self_attn.q_norm"
    )

    assert any(
        record.node == norm.uid and record.input_index >= 0
        for record in analysis.redistributes
    )
    assert int(flops.values[norm.uid]) == int(
        analysis.cost.values[norm.uid].flops
    )


def test_physical_flops_reject_conflicting_kernel_input_labels(monkeypatch):
    identity = sp.Symbol("s_kv")
    fake = OpRef(
        name="probe",
        _spec=SimpleNamespace(
            shape_in=(("s_kv",), ("s_kv",)),
            shape_kind="explicit",
            flops_fn=lambda dims: dims["s_kv"],
            arity=lambda: 2,
            repeats_last_input=False,
        ),
        _custom=False,
    )
    monkeypatch.setattr(
        OpRef,
        "resolve",
        classmethod(lambda cls, op, program, **_: fake),
    )
    analysis = SimpleNamespace(program=object())
    node = SimpleNamespace(op="probe", display_path="probe.contract",
                           axis=None)
    shape = Shape(
        ins=((Dim(identity, 64),), (Dim(identity, 128),)),
    )

    with pytest.raises(ValueError, match=r"input label 's_kv'.*conflicting"):
        _primitive_flops(
            analysis,
            node,
            shape,
            {"s_kv": Dim(identity, 128)},
        )


def test_physical_flops_preserve_missing_input_port_coordinates(monkeypatch):
    left = sp.Symbol("left")
    right = sp.Symbol("right")
    fake = OpRef(
        name="probe",
        _spec=SimpleNamespace(
            shape_in=(("left",), ("right",)),
            shape_kind="explicit",
            flops_fn=lambda dims: dims["right"],
            arity=lambda: 2,
            repeats_last_input=False,
        ),
        _custom=False,
    )
    monkeypatch.setattr(
        OpRef,
        "resolve",
        classmethod(lambda cls, op, program, **_: fake),
    )
    analysis = SimpleNamespace(program=object())
    node = SimpleNamespace(op="probe", display_path="probe.ports",
                           axis=None)
    shape = Shape(ins=((), (Dim(right, 7),)))

    assert int(_primitive_flops(
        analysis,
        node,
        shape,
        {"left": Dim(left, 2), "right": Dim(right, 3)},
    )) == 7


@pytest.mark.regression  # every model of the lane cohort
@pytest.mark.parametrize("snapshot", LANE_COHORT)
def test_tp8_lane_corpus_has_total_local_geometry(snapshot):
    for phase in ("prefill", "decode"):
        analysis = _step(snapshot, phase).analysis
        shapes = physical_kernel_shapes(analysis)
        flops = physical_flops(analysis, shapes)
        assert shapes.values
        assert set(flops.values) == set(analysis.cost.values)
        assert all(value_of(value) >= 0 for value in flops.values.values())


def test_ep_dispatch_deployment_projects_declared_execution_copies():
    step = evaluate_step(
        _resolved("Qwen__Qwen3-30B-A3B.json"),
        ParallelConfig(ep_moe=8),
        phase="prefill",
        batch=1,
        input_tokens=128, output_tokens=0,
    )
    analysis = step.analysis
    flops = physical_flops(analysis)

    assert set(flops.values) == set(analysis.cost.values)
    for value in flops.values.values():
        grounded = value_of(value)
        assert not getattr(grounded, "free_symbols", None)
        assert grounded >= 0


def test_ep_dispatch_weightless_leaves_run_on_the_ranks_experts():
    """A weightless leaf of a packed bank (a slice, the activation) runs on
    the `[n_expert_copies, M, d]` buffer, so its per-rank tensor holds the
    rank's experts — the expert axis cut by EP like any other shape."""
    step = evaluate_step(
        _resolved("openai__gpt-oss-20b.json"),
        ParallelConfig(ep_moe=4),
        phase="prefill",
        batch=1,
        input_tokens=128, output_tokens=0,
    )
    analysis = step.analysis
    flops = physical_flops(analysis)
    shapes = physical_kernel_shapes(analysis, physical_shapes(analysis))
    weightless = [
        node
        for node in analysis.program.nodes()
        if (
            node.in_dispatch_zone
            and not node.children
            and not analysis.storage.values.get(node.uid)
        )
    ]

    assert weightless
    for node in weightless:
        assert int(value_of(analysis.shapes.values[node.uid].out[0])) == 32
        assert int(value_of(shapes.values[node.uid].out[0])) == 8
    assert set(flops.values) == set(analysis.cost.values)


def test_ep_dispatch_zone_divides_the_bank_by_its_expert_axis():
    # MiniMax-M2's expert bank is bracketed by permute/combine, so the w1/w3/
    # act/gate_mul/w2 leaves of its `gated_experts` sit INSIDE the dispatch
    # zone. At ep_moe=8 the bank's 256 experts are cut 32 to a rank along
    # their axis, and each leaf's per-rank FLOPs are its global FLOPs over 8.
    step = evaluate_step(
        _resolved("MiniMaxAI__MiniMax-M2.7.json"),
        ParallelConfig(ep_moe=8),
        phase="prefill",
        batch=1,
        input_tokens=128, output_tokens=0,
    )
    analysis = step.analysis
    flops = physical_flops(analysis)
    compared = 0
    for node in analysis.program.nodes():
        if node.children or not analysis.program.in_zone(node):
            continue
        compared += 1
        assert (value_of(flops.values[node.uid]) * 8
                == value_of(analysis.cost.values[node.uid].flops))
    assert compared == 5


def test_dcp_kernel_geometry_connects_compute_and_collectives():
    step = evaluate_step(
        _resolved("meta-llama__Llama-3.2-3B.json"),
        ParallelConfig(tp_attention=4, tp_dense_ffn=4, cp=2, dcp=2),
        phase="decode",
        batch=1,
        input_tokens=128, output_tokens=0,
    )
    analysis = step.analysis
    semantic = physical_shapes(analysis)
    shapes = physical_kernel_shapes(analysis, semantic)
    flops = physical_flops(analysis, shapes)
    by_path = {node.display_path: node for node in analysis.program.nodes()}
    score = by_path["self_attn.qk_score__gqa_attention"]
    softmax = by_path["self_attn.softmax__gqa_attention"]
    apply = by_path["self_attn.attn_apply__gqa_attention"]

    score_shape = shapes.values[score.uid]
    softmax_shape = shapes.values[softmax.uid]
    apply_shape = shapes.values[apply.uid]
    assert _ints(score_shape.ins[0]) == (1, 12, 1, 128)
    assert _ints(score_shape.ins[1]) == (1, 2, 64, 128)
    assert _ints(score_shape.out) == (1, 12, 1, 64)
    assert _ints(softmax_shape.ins[0]) == _ints(score_shape.out)
    assert _ints(softmax_shape.out) == _ints(score_shape.out)
    assert _ints(apply_shape.ins[0]) == _ints(score_shape.out)
    assert _ints(apply_shape.ins[1]) == (1, 2, 64, 128)
    assert _ints(apply_shape.out) == (1, 12, 1, 128)
    # The semantic analysis remains the RS-completed edge, not the invocation's
    # gathered-head buffer.
    assert _ints(semantic.values[apply.uid].out) == (1, 6, 1, 128)

    assert all(int(flops.values[node.uid]) > 0
               for node in (score, softmax, apply))

    events = materialize_comm_events(analysis)
    query = next(
        event for event in events
        if event.world == "dcp"
        and event.op == "all_gather"
        and event.report_anchor_uid == score.uid
    )
    output = next(
        event for event in events
        if event.world == "dcp"
        and event.op == "reduce_scatter"
        and event.count_uid == apply.uid
    )
    assert query.collective_out_shape == _ints(score_shape.ins[0])
    assert output.collective_in_shape == _ints(apply_shape.out)
    assert output.collective_out_shape == _ints(
        semantic.values[apply.uid].out
    )


def test_dcp_dense_mla_kernel_and_collective_chains_are_closed():
    analysis = evaluate_step(
        _resolved("deepseek-ai__DeepSeek-V3.json"),
        ParallelConfig(
            tp_attention=8,
            tp_dense_ffn=8,
            tp_moe_expert=8,
            ep_moe=2,
            cp=2,
            dcp=2,
        ),
        phase="decode",
        batch=1,
        input_tokens=4096, output_tokens=0,
    ).analysis
    shapes = physical_kernel_shapes(analysis)
    flops = physical_flops(analysis, shapes)
    score = next(
        node for node in analysis.program.nodes()
        if node.op == "attention_score"
    )
    softmax = next(
        node for node in analysis.program.nodes() if node.op == "softmax"
    )
    apply = next(
        node for node in analysis.program.nodes()
        if node.op == "attention_apply"
    )
    score_shape = shapes.values[score.uid]
    softmax_shape = shapes.values[softmax.uid]
    apply_shape = shapes.values[apply.uid]

    assert int(value_of(score_shape.ins[0][1])) == int(
        value_of(score_shape.ins[1][1])
    )
    assert _ints(score_shape.out) == _ints(softmax_shape.ins[0])
    assert _ints(softmax_shape.out) == _ints(apply_shape.ins[0])
    assert value_of(flops.values[score.uid]) > 0
    events = materialize_comm_events(analysis)
    query = next(
        event for event in events
        if event.world == "dcp"
        and event.op == "all_gather"
        and event.report_anchor_uid == score.uid
    )
    output = next(
        event for event in events
        if event.world == "dcp"
        and event.op == "reduce_scatter"
        and event.count_uid == apply.uid
    )
    assert query.collective_out_shape == _ints(score_shape.ins[0])
    assert output.collective_in_shape == _ints(apply_shape.out)


def test_dcp_dsa_without_sparse_position_ownership_fails_closed():
    with pytest.raises(
        NotImplementedError, match="no typed cache-owner workload contract"
    ):
        _ = evaluate_step(
            _resolved("deepseek-ai__DeepSeek-V3.2.json"),
            ParallelConfig(
                tp_attention=8,
                tp_dense_ffn=8,
                tp_moe_expert=8,
                ep_moe=2,
                cp=2,
                dcp=2,
            ),
            phase="decode",
            batch=1,
            input_tokens=4096, output_tokens=0,
        ).analysis


def test_dcp_sink_without_a_physical_side_flow_fails_closed():
    with pytest.raises(
        NotImplementedError, match="no physical sink replication or side-flow"
    ):
        _ = evaluate_step(
            _resolved("openai__gpt-oss-20b.json"),
            ParallelConfig(
                tp_attention=4,
                tp_dense_ffn=4,
                tp_moe_expert=8,
                cp=2,
                dcp=2,
            ),
            phase="decode",
            batch=1,
            input_tokens=128, output_tokens=0,
        ).analysis
