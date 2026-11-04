"""DeepSeek-V4's compressor pools complete windows along a real window axis.

Source (`inference/model.py::Compressor.forward`, prefill branch):

    remainder = seqlen % ratio; cutoff = seqlen - remainder
    kv, score = kv[:, :cutoff], score[:, :cutoff]        # tail -> state
    score = score.unflatten(1, (-1, ratio)) + ape
    kv = (kv.unflatten(1, (-1, ratio)) * score.softmax(dim=2)).sum(dim=2)

So the compute frame holds `S // m` COMPLETE windows; the tail tokens go to
the decode-time state. The cache's `ceil(s_kv / m)` is a capacity law on the
`token_pool_compressor` marker, not a shape. The pins below are the facts that
reading implies: element counts of each node in the window frame, the ragged
tail and the one-token decode step, the marker's identity shape, and the
softmax's normalized axis in the layout law.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import sympy as sp

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch import layout
from backend.arch.eval.analysis import Point
from backend.arch.eval.dim import value_of
from backend.arch.layout.edge import EdgeState
from backend.arch.layout.groups import ProcessGroupFamily
from backend.arch.layout.linear import normalize_axis

MODEL = "deepseek-ai/DeepSeek-V4-Flash"


def _step(phase: str, sq: int, lq: int):
    from tests.snapshots import shared_loader
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.step_eval import evaluate_step
    from tests.snapshots import resolve_snapshot

    resolved = resolve_snapshot(MODEL, shared_loader())
    # A prefill of `sq` new tokens over `lq` positions caches the rest; a
    # decode step's one query reads the `lq` positions written.
    assert phase == "prefill" or sq == 1
    return evaluate_step(resolved, ParallelConfig(), phase=phase, batch=1,
                         input_tokens=lq, output_tokens=0,
                         s_prefix_cached=lq - sq if phase == "prefill" else 0
                         ).analysis


def _compressors(analysis):
    """{(variant, path, scope): {leaf: (node, out, flops, in[0], out Dims)}}."""
    out: dict = {}
    for uid in analysis.shapes:
        node = analysis.program.by_uid(uid)
        comp = node.component_path or ""
        if "compressor" not in node.display_path or node.children:
            continue
        # The component the leaf sits in, which owns its weight-bearing
        # leaves (`norm`) as well as its suffixed ones.
        owner = node.component_path[-1][1] if node.component_path else ""
        variant = "overlap" if owner == "dsv4_compressor_overlap" \
            else "no_overlap" if owner == "dsv4_compressor_no_overlap" \
            else None
        if variant is None:
            continue
        key = (variant, node.display_path.rsplit(".", 1)[0], comp)
        cost = analysis.cost.get(uid)
        shape = analysis.shapes[uid]
        out.setdefault(key, {})[node.name] = (
            node, tuple(value_of(d) for d in shape.out),
            value_of(cost.flops) if cost is not None else 0,
            tuple(value_of(d) for d in shape.ins[0]) if shape.ins else (),
            tuple(shape.out))
    return out


def _elements(shape):
    total = 1
    for d in shape:
        total *= d
    return total


@pytest.fixture(scope="module")
def prefill_ragged():
    # 4100 = 32 * 128 + 4 = 1025 * 4: the HCA compressor (m=128) has a
    # 4-token tail, the CSA/indexer compressors (m=4) have none.
    return _compressors(_step("prefill", 4100, 4100))


@pytest.fixture(scope="module")
def prefill_1000():
    # 1000 = 7 * 128 + 104 = 250 * 4: seven HCA windows, 104 tokens to state.
    return _compressors(_step("prefill", 1000, 1000))


@pytest.fixture(scope="module")
def decode():
    return _compressors(_step("decode", 1, 4864))


def test_window_frame_holds_complete_windows_only(prefill_ragged):
    seen = set()
    for (_variant, _path, _comp), nodes in prefill_ragged.items():
        wgate = nodes["gate_usable"][3]
        fold = nodes["gate_win"][1]
        m = fold[2]
        seen.add(m)
        # [B, S, D] -> [B, S // m, m, D]: the tail is sliced off first, so
        # the fold is an exact product and no window is padded.
        assert nodes["gate_usable"][1][1] == (wgate[1] // m) * m
        assert fold == (wgate[0], wgate[1] // m, m, wgate[2])
        assert nodes["kv_win"][1] == fold
        if _variant == "overlap":
            # The overlap pool of a slot is 2m rows of D = fold[3] / 2: the
            # previous slot's Ca half (shifted one slot along the slot axis,
            # same shape) beside this slot's Cb half, joined on the window
            # axis. Still only complete windows: the slot count is the fold's.
            d = fold[3] // 2
            assert nodes["kv_a"][1] == nodes["kv_b"][1] == (*fold[:3], d)
            assert nodes["kv_prev"][1] == nodes["kv_a"][1]
            assert nodes["score_prev"][1] == nodes["score_a"][1]
            assert nodes["kv_pair"][1] == (fold[0], fold[1], 2 * m, d)
            assert nodes["score_pair"][1] == nodes["kv_pair"][1]
            # The slots survive the window sum; the 2m rows collapse.
            assert nodes["pooled"][1] == (fold[0], fold[1], d)
        else:
            # The slots survive the window sum; the window axis collapses.
            assert nodes["pooled"][1] == (fold[0], fold[1], fold[3])
    assert seen == {4, 128}


def test_window_ops_cost_their_elements(prefill_ragged):
    """Per-element ops in the rank-4 frame cost exactly their elements —
    the fold's `B * (S//m) * m * D`, however the op's rank-3 template
    lines up against the rank-4 tensor."""
    per_element = {"scores": 1, "smax": 5, "weighted": 1, "pooled": 1}
    for nodes in prefill_ragged.values():
        n = _elements(nodes["gate_win"][1])
        for name, k in per_element.items():
            assert nodes[name][2] == k * n, name


def test_non_divisible_prefill_keeps_complete_windows_and_names_them(prefill_1000):
    """S=1000, m=128: 7 complete windows, the 104-token tail goes to the
    state — and the slot axis keeps its identity from the fold through the
    window sum to the norm and RoPE after pooling."""
    hca = [n for (v, _p, _c), n in prefill_1000.items() if v == "no_overlap"]
    assert hca
    for nodes in hca:
        assert nodes["gate_usable"][3][1] == 1000
        assert nodes["gate_usable"][1][1] == 7 * 128
        assert nodes["gate_win"][1][1:3] == (7, 128)
        slot = nodes["gate_win"][4][1]
        # The walk binds the query tokens as `s_query` (s_ctx is only a
        # lookup alias of it); the slot is named by the tokens it folds.
        assert "s_query" in slot.names
        # sum over the window axis yields the slot axis, the same Dim.
        assert nodes["pooled"][4][1] == slot
        assert nodes["pooled"][4][1].names == slot.names
        # pe_slice / nope_slice read the norm output.
        for name in ("pe_slice", "nope_slice", "pe_rope", "roped", "pooled_out"):
            assert nodes[name][4][1] == slot, name
            assert nodes[name][1][:2] == (1, 7), name


def test_decode_pools_once_every_m_tokens(decode):
    """Decode is the other branch of Compressor.forward: each token's row
    joins the ring, and once every m tokens the m state rows (2m in the
    overlap variant) are pooled into one row, which is normalized, roped and
    cached. A step completes 1/m of a window, so the window ops cost one
    token's rows and the pooled row's norm and RoPE cost 1/m of a row."""
    seen = set()
    for (variant, _path, _comp), nodes in decode.items():
        assert "gate_usable" not in nodes
        wgate = nodes["gate_win"][3]
        fold = nodes["gate_win"][1]
        m = fold[2]
        seen.add(m)
        assert wgate[1] == 1                       # one token per step
        assert fold == (wgate[0], sp.Rational(1, m), m, wgate[2])
        assert nodes["kv_win"][1] == fold
        # The window ops read the ring: m rows of the projected width per
        # window, 2m rows of D in the overlap variant — one token's row of
        # the projection per step either way.
        rows = 2 * m if variant == "overlap" else m
        width = fold[3] // 2 if variant == "overlap" else fold[3]
        assert nodes["smax"][1] == (fold[0], fold[1], rows, width)
        assert nodes["weighted"][1] == nodes["smax"][1]
        n = _elements(nodes["smax"][1])
        assert n == _elements(wgate)
        assert nodes["scores"][2] == n
        assert nodes["smax"][2] == 5 * n
        assert nodes["weighted"][2] == n
        assert nodes["pooled"][2] == n
        assert nodes["smax"][0].axis == 2 and nodes["pooled"][0].axis == 2
        # One pooled row per window: 1/m of a row per step through the norm,
        # RoPE and the cached latent.
        for name in ("pooled", "norm", "pe_rope", "roped", "pooled_out"):
            assert nodes[name][1][:2] == (fold[0], fold[1]), name
        assert nodes["pooled_out"][1][2] == width
        # RMSNorm is 2 FLOPs per element: one D-wide row per m tokens.
        assert nodes["norm"][2] == 2 * fold[0] * sp.Rational(1, m) * width
    assert seen == {4, 128}


def test_the_average_slot_reaches_no_kernel_geometry():
    """The `s_ctx / m` slot is an average rate, not a geometry. It lives in
    the compressor's decode pool and in the KV rows that pool appends to the
    attention's union (one compressed row per m steps); the kernels that
    read the cache see whole extents."""
    from backend.arch.eval.dim import expr_of
    analysis = _step("decode", 1, 4864)
    carriers = set()
    for uid, shape in analysis.shapes.values.items():
        node = analysis.program.by_uid(uid)
        axes = [*(shape.out or ())]
        for ins in shape.ins or ():
            axes.extend(ins)
        if any(not sp.sympify(value_of(a)).is_integer
               and {"Cape", "compress_ratio"} & {
                   str(f) for f in sp.sympify(expr_of(a)).free_symbols}
               for a in axes):
            owner = node.component_path[-1][1] if node.component_path else ""
            carriers.add("compressor-internal"
                         if owner.startswith("dsv4_compressor") else node.op)
    # `head_broadcast` is a component: its `broadcast` and `transpose` are
    # the views it is made of.
    assert carriers == {"compressor-internal", "dsv4_compressor_overlap",
                        "dsv4_compressor_no_overlap", "kv_stream_concat",
                        "head_broadcast", "broadcast", "transpose"}, carriers


def test_marker_is_identity_and_carries_the_ceil_law(prefill_ragged):
    for nodes in prefill_ragged.values():
        node, shape, flops, _in, _dims = nodes["pooled_out"]
        assert node.op == "token_pool_compressor"
        assert flops == 0
        roped = nodes["roped"][1]
        assert shape == roped
    from backend.arch.ops import get_op
    spec = get_op("token_pool_compressor")
    assert spec.shape_kind == "preserve" and not spec.couples_on
    stored = spec.kv_storage_positions_per_query_fn(
        {"s_kv": 4100, "compress_ratio": 128})
    assert int(value_of(stored)) == 33     # ceil: the tail owns a slot


def test_axes_are_declared_on_the_nodes(prefill_ragged):
    for nodes in prefill_ragged.values():
        assert nodes["smax"][0].axis == 2
        assert nodes["pooled"][0].axis == 2
        assert nodes["pooled"][0].op == "sum_reduce"
        # `in` is the reduced width (the window), as for every sum_reduce:
        # m positions, or the overlap variant's 2m rows.
        overlap = "kv_pair" in nodes
        assert nodes["pooled"][0].dims_raw["in"] == (
            "2 * Cape" if overlap else "Cape")
        if overlap:
            # The one-slot shift runs along the slot axis, and the pair joins
            # on the window axis the softmax and the sum reduce.
            assert nodes["kv_prev"][0].op == nodes["score_prev"][0].op == "shift"
            assert nodes["kv_prev"][0].axis == nodes["score_prev"][0].axis == 1
            assert nodes["kv_pair"][0].axis == nodes["score_pair"][0].axis == 2


def _normalize(axis, layout_in):
    rep = layout.replicated(frozenset(range(8)))
    rule = SimpleNamespace(sole=lambda node, states: (0, states[0]),
                           replicated=lambda: rep)
    node = SimpleNamespace(op="softmax_act", axis=axis, display_path="smax",
                           inputs_obj=(), shape=None, dims_raw={})
    state = EdgeState(tuple(layout_in), None, None)
    required, out = normalize_axis(rule, node, [state], [()])
    return rep, required[0], out


def test_softmax_needs_only_its_normalized_axis_whole():
    rep = layout.replicated(frozenset(range(8)))
    cut = layout.sharded(ProcessGroupFamily(
        "x", ((0, 1), (2, 3), (4, 5), (6, 7))))
    # [B, slots, window, D] with the feature cut: orthogonal to a softmax
    # over the window axis, so it survives.
    _rep, need, out = _normalize(2, (rep, rep, rep, cut))
    assert need.layout[3] == cut and out.layout[3] == cut
    # The same cut on the window axis is inside the normalization.
    _rep, need, out = _normalize(2, (rep, rep, cut, rep))
    assert need.layout[2] == rep and out.layout[2] == rep
    # A router's rank-3 form normalizes its last axis.
    _rep, need, _out = _normalize(None, (rep, rep, cut))
    assert need.layout[2] == rep
    # A declared axis that is not the signature's normalized axis is refused,
    # and so is a rank-4 node that leaves its non-last normalized axis unsaid.
    for axis in (3, None):
        with pytest.raises(ValueError, match="normalizes axis 2"):
            _normalize(axis, (rep, rep, rep, rep))


def test_softmax_metadata_names_the_window_axis(prefill_ragged, decode):
    """Registry, AxisUse and the node state one fact: the coupled label
    `dim` sits at the window axis of every DSv4 rank-4 softmax, which is the
    node's declared `axis: 2`."""
    from backend.arch.axis_propagation import signature_for
    from backend.arch.layout.axis_use import AXIS_USE
    from backend.arch.ops import get_op
    spec = get_op("softmax_act")
    assert tuple(spec.couples_on) == ("dim",)
    assert set(AXIS_USE["softmax_act"]) == {"dim"}
    assert AXIS_USE["softmax_act"]["dim"].category == "normalize"
    for frame in (prefill_ragged, decode):
        for nodes in frame.values():
            node = nodes["smax"][0]
            sig = signature_for(spec, node, 4)
            assert sig is not None
            assert sig.ins[0].index("dim") == node.axis == 2
            assert len(nodes["smax"][1]) == 4


def test_cp_gathers_the_sequence_before_the_fold():
    """Under CP the ranks' chunks end wherever the cut falls, not on window
    boundaries, so the folded sequence arrives whole: every rank pools every
    window (the gather is a priced collective), divisible or not."""
    from tests.snapshots import shared_loader
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.pipeline import analyze
    from tests.snapshots import resolve_snapshot

    resolved = resolve_snapshot(MODEL, shared_loader())

    def window_flops(cp, seq):
        rows = analyze(resolved, parallel=ParallelConfig(cp=cp), batch_size=1,
                       avg_input_tokens=seq, avg_output_tokens=0, phase="prefill", breakdown=False).per_op_timing
        return sum(r["flops_per_rank"] for r in rows
                   if ".compressor." in r["aggregation_id"]
                   and r["op"] in ("position_table_add", "softmax_act",
                                   "sum_reduce"))

    for seq in (4096, 1000):
        assert window_flops(2, seq) == window_flops(1, seq), seq


class _Rule:
    """The four calls the geometry laws make of a propagation rule."""

    def __init__(self, env):
        self.env = env

    def sole(self, node, states):
        return 0, states[0]

    def replicated(self):
        return layout.replicated(frozenset(range(8)))

    def cell(self, node, what, result):
        return result.require(what, node.display_path)

    def whole_axis(self, node, state, index, use=None, what=""):
        fixed = list(state.layout)
        fixed[index] = self.replicated()
        return EdgeState(tuple(fixed), state.value, state.workload)


def _split(axis, layout_in, shape, factors):
    from backend.arch.eval.dim import Dim
    from backend.arch.layout.geometry import view_split
    names = list(factors)
    node = SimpleNamespace(
        op="view_split", axis=axis, uid=1, display_path="split", shape=None, out_shape=None,
        dims_raw={k: k for k in names}, inputs_obj=())
    rule = _Rule({1: {k: Dim.symbol(k, v) for k, v in factors.items()}})
    state = EdgeState(tuple(layout_in), None, None)
    return view_split(rule, node, [state], [tuple(shape)])


def test_view_split_layout_folds_a_whole_sequence_and_keeps_head_cuts():
    rep = layout.replicated(frozenset(range(8)))
    cut = layout.sharded(ProcessGroupFamily(
        "x", ((0, 1), (2, 3), (4, 5), (6, 7))))
    seq = layout.sharded(ProcessGroupFamily(
        "y", ((0, 2), (1, 3), (4, 6), (5, 7))))
    # A sequence fold: the cut sequence is required whole, the feature cut
    # (an axis the fold does not touch) survives on its own axis.
    required, out = _split(1, (rep, seq, cut), (1, 1024, 512),
                           {"s_slots": 8, "window": 128})
    assert required[0].layout == (rep, rep, cut)
    assert out.layout == (rep, rep, rep, cut)
    # rank_out = rank_in + 1, and a negative axis names the same position.
    _r, neg = _split(-2, (rep, seq, cut), (1, 1024, 512),
                     {"s_slots": 8, "window": 128})
    assert neg.layout == out.layout and len(out.layout) == 4
    # A head split of the last axis keeps the head cut on the outer factor.
    required, out = _split(-1, (rep, rep, cut), (1, 16, 512),
                           {"n_heads": 8, "d_head": 64})
    assert required[0].layout == (rep, rep, cut)
    assert out.layout == (rep, rep, cut, rep)
    # No declared axis is refused, never guessed.
    with pytest.raises(ValueError, match="no `axis:`"):
        _split(None, (rep, rep, rep), (1, 16, 512),
               {"n_heads": 8, "d_head": 64})


def test_a_view_split_without_an_axis_is_a_repairable_schema_error():
    from backend.arch.schema import BlockGraph, GraphNode
    with pytest.raises(ValueError, match=r"without `axis:`.*axis: -1"):
        BlockGraph(inputs=["x"], graph={
            "q": GraphNode(op="view_split", inputs=["x"],
                           dims={"n_heads": "2", "d_head": "4"})},
            outputs=["q"])


def test_a_sequence_cut_gathers_the_projected_tokens_before_the_windows():
    """Under context parallelism each rank holds a block of the prompt, and
    a window may straddle two ranks. The compressor's topology does not
    depend on the prompt's length: the projected tokens (`wkv`, `wgate`) are
    gathered across the sequence group before the complete windows are
    sliced off and pooled — sglang's `Compressor.compute_kv_score` gathers
    `kv_score` in global token order first
    (`layers/attention/dsv4/compressor.py:425-433`) — at a length shorter
    than one window, at one that is no multiple of it, and at one that is.
    One walk serves all three, and its gathers read the length there."""
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.pipeline import serving_layers
    from backend.inference.step_eval import evaluate_step
    from tests.snapshots import resolve_snapshot

    step = evaluate_step(resolve_snapshot(MODEL),
                         ParallelConfig.from_dict({"cp": 8}),
                         phase="prefill", batch=1, input_tokens=4096,
                         output_tokens=0)
    layers = serving_layers(step, framework_binding=None, stage=None,
                            timed=step.walk.program.included_stack_names(
                                include_aux=False))
    gathered = {node.uid: node.name for node in layers.pre_program.nodes()
                if node.provenance == "comm"
                and node.name.endswith(("__on__gate_usable",
                                        "__on__kv_usable"))}
    assert gathered and all("__all_gather__" in name
                            for name in gathered.values())
    for tokens in (104, 1032, 4096):
        at = layers.at(Point({**step.point, "s_input": tokens}))
        for uid in gathered:
            # The whole prompt's projected tokens arrive on every rank.
            assert at.comm_meta.values[uid]["flow_out_shape"][1] == tokens


def test_glm_kpool_compressor_gathers_its_projections_before_the_windows():
    """GLM-5.3-Flash's pooled-key indexer (`glm_kpool_compressor`) slices
    its projected keys and gate into whole pools (`keys_usable`,
    `gate_usable`) before it windows them. Under context parallelism the
    sequence a rank holds is a block of the prompt, so the same rule as the
    DSv4 compressor's places the gathers: on the projections' way into the
    slice, before any pool is cut. No framework runs this model with prefill
    CP today; the placement is the model's own, and this pins it."""
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.pipeline import serving_layers
    from backend.inference.step_eval import evaluate_step
    from tests.snapshots import resolve_snapshot

    step = evaluate_step(resolve_snapshot("zai-org/GLM-5.3-Flash"),
                         ParallelConfig.from_dict({"cp": 8}),
                         phase="prefill", batch=1, input_tokens=4096,
                         output_tokens=0)
    layers = serving_layers(step, framework_binding=None, stage=None,
                            timed=step.walk.program.included_stack_names(
                                include_aux=False))
    gathers = {node.uid: node.name for node in layers.pre_program.nodes()
               if node.provenance == "comm"
               and node.display_path.split(".")[-2] == "index_kpool"}
    assert sorted(name.split("__on__")[1] for name in gathers.values()) == [
        "gate_usable", "gate_usable", "keys_usable", "keys_usable"]
    assert all("__all_gather__" in name for name in gathers.values())
    for tokens in (104, 1032, 4096):
        at = layers.at(Point({**step.point, "s_input": tokens}))
        for uid in gathers:
            assert at.comm_meta.values[uid]["flow_out_shape"][1] == tokens
