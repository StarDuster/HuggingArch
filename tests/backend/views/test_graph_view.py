"""Shared overlay + L1 entry points (`backend.views.graph_view`)."""
from __future__ import annotations

import pytest

from backend.views.abstraction import Sums, abstract
from tests.backend.views import declared_roles
from backend.views.graph_view import (
    diff_of_graphs, diff_representative_blocks, entry_from_graph,
    l1_of_nodes,
)


def _n(nid, op, inputs=(), flops=0, params=0, **kw):
    return {"id": nid, "op": op, "tensor_name": nid, "inputs": list(inputs),
            "flops": flops, "params": params, "weight_bytes": 2 * params,
            **declared_roles(op), **kw}


def _attn():
    return [
        _n("x", "input"),
        _n("q", "linear", ["x"], 10, 5),
        _n("kv", "linear", ["x"], 4, 2),
        _n("k", "slice", ["kv"]),
        _n("v", "slice", ["kv"]),
        _n("score", "attention_score", ["q", "k"], 3),
        _n("sm", "softmax", ["score"], 1),
        _n("apply", "attention_apply", ["sm", "v"], 3),
        _n("o", "linear", ["apply"], 10, 5),
    ]


def test_l1_of_nodes_requires_totals():
    with pytest.raises(TypeError):
        l1_of_nodes(_attn())  # type: ignore[misc]


def test_l1_of_nodes_matches_abstract_with_the_same_totals():
    nodes = _attn()
    totals = Sums(sum(n["flops"] for n in nodes),
                  sum(n["params"] for n in nodes),
                  sum(n["weight_bytes"] for n in nodes))
    assert l1_of_nodes(nodes, totals) == abstract(nodes, totals).to_dict()


def test_diff_of_graphs_overlays_two_copies():
    graph = {"nodes": _attn(), "edges": [], "inputs": ["x"], "outputs": ["o"]}
    out = diff_of_graphs([
        {"key": "a", "graph": graph},
        {"key": "b", "graph": graph},
    ])
    assert out["available"] is True
    assert out["entries"] == ["a", "b"]


def test_overlay_without_l1_omits_numeric_totals():
    """Block/serving graphs have no L1; overlay must not invent 0==0 totals."""
    graph = {"nodes": _attn(), "edges": [], "inputs": ["x"], "outputs": ["o"]}
    out = diff_of_graphs([
        {"key": "a", "graph": graph},
        {"key": "b", "graph": graph},
    ])
    assert out["available"] is True
    totals = out.get("totals")
    assert totals is None
    assert not isinstance((totals or {}).get("delta"), list)


def test_diff_of_graphs_refuses_a_single_entry():
    out = diff_of_graphs([{"key": "a", "graph": {"nodes": _attn(), "edges": []}}])
    assert out["available"] is False


def test_stream_in_is_dropped_before_entry():
    """`declared()` refuses stream_in; overlay strips it so the walk can run."""
    nodes = [*_attn(), _n("st", "stream_in")]
    graph = {"nodes": nodes, "edges": [], "inputs": ["x"], "outputs": ["o"],
             "reads": ["cache"]}
    entry = entry_from_graph(key="a", graph=graph)
    assert "st" not in entry.nodes
    assert entry.reads == ["cache"]
    out = diff_of_graphs([
        {"key": "a", "graph": graph},
        {"key": "b", "graph": graph},
    ])
    assert out["available"] is True


def _block(name):
    """A block whose frame binds `d_ffn` to a value of its own, so a
    winner's `dims` says which block it was read off."""
    from backend.arch.eval.projection.result import BlockReport, NodeReport
    node = NodeReport(name="up", op="linear", flops=0, params=0)
    return BlockReport(template_name=name, nodes=[node], total_flops=0,
                       total_params=0,
                       dims={"d_ffn": {"dense": 3, "moe": 58}.get(name, 1)})


def test_representative_template_picks_the_largest_main_path_count():
    from types import SimpleNamespace
    from backend.analyzer import _representative_template

    report = SimpleNamespace(
        block_results=[("dense", 3, _block("dense")),
                       ("moe", 58, _block("moe")),
                       ("mtp", 1, _block("mtp"))],
        block_aux_kinds=[None, None, "mtp"],
        block_derived_groups=[None, None, None],
    )
    got = _representative_template(report)
    assert got == {"name": "moe", "count": 58, "n_layers": 61, "bucket_id": 1,
                   "call": "1", "dims": {"d_ffn": 58}, "basis": "most_layers"}


def test_diff_representative_reads_each_published_dims():
    """Compare explains a block diff in the block's own dim frame — the
    values Analyzer published with its winner, not the model's dim table."""
    from types import SimpleNamespace

    def payload(value):
        graph = {
            "nodes": [_n("x", "input"),
                      _n("up", "linear", ["x"], params=value,
                         params_symbolic="d_ffn*d_model")],
            "edges": [], "inputs": ["x"], "outputs": ["up"],
        }
        return {"block_graph": {"block": graph},
                "representative_template": {"name": "block",
                                            "dims": {"d_ffn": value}}}

    resolved = SimpleNamespace(
        spec=SimpleNamespace(custom_ops={}), dim_values={"d_ffn": 999})
    out = diff_representative_blocks(
        [payload(3), payload(58)], [resolved, resolved], ["a", "b"])
    assert out["available"] is True
    changed = next(c for c in out["classes"] if c["diff"])
    assert changed["members"] == ["up", "up"]
    assert changed["because"]["dims"] == {"d_ffn": [3, 58]}


def test_representative_template_a_tie_names_no_block():
    from types import SimpleNamespace
    from backend.analyzer import _representative_template

    report = SimpleNamespace(
        block_results=[("a", 16, _block("a")), ("b", 16, _block("b"))],
        block_aux_kinds=[None, None],
        block_derived_groups=[None, None],
    )
    got = _representative_template(report)
    assert got["name"] is None and got["basis"] == "tie"
    assert [b["name"] for b in got["tied"]] == ["a", "b"]
    assert (got["count"], got["n_layers"]) == (16, 32)


def test_representative_template_counts_a_declaration_not_its_groups():
    """The groups a `read: all` stack splits one declaration into are one
    block: Kimi-K3's 61 `block_kda_moe` layers in eight history lengths are
    its representative, not a tie of eight 8-layer groups. The bucket is the
    declaration's largest group."""
    from types import SimpleNamespace
    from backend.analyzer import _representative_template

    def group(m):
        return {"declaration": {"layer_path": [3], "assignment_index": 1,
                                "base_template": "kda"}, "visible": m}

    report = SimpleNamespace(
        block_results=[("mla", 24, _block("mla")),
                       ("kda", 8, _block("kda")),
                       ("kda", 10, _block("kda")),
                       ("kda", 9, _block("kda"))],
        block_aux_kinds=[None] * 4,
        block_derived_groups=[None, group(1), group(2), group(3)],
    )
    got = _representative_template(report)
    assert (got["name"], got["count"], got["bucket_id"], got["basis"]) == (
        "kda", 27, 2, "most_layers")


def test_stamp_l1_covers_the_component_not_the_block():
    from backend.views.graph_view import stamp_l1
    kids = _attn()
    attn = {
        "id": "self_attn", "op": "mla_attention", "children": kids,
        "flops": sum(n["flops"] for n in kids),
        "params": sum(n["params"] for n in kids),
        "weight_bytes": sum(n["weight_bytes"] for n in kids),
    }
    down = _n("down", "linear", ["x"], 4, 2)
    block = [_n("x", "input"), attn, down]
    stamp_l1(block)
    assert "l1" not in block[0] and "l1" not in down
    assert attn["l1"]["available"] is True, attn["l1"].get("reason")
    members = {m for b in attn["l1"]["boxes"] for m in b["members"]}
    assert "down" not in members
    stamp_l1(block)
    assert attn["l1"]["available"] is True


def test_stamp_l1_groups_a_fused_attention_without_cost_fields():
    """The serving graph has no per-op FLOPs; L1 still boxes q / k+v / core."""
    from backend.rendering._compute_graph import _lifted_input_roles
    from backend.views.graph_view import stamp_l1
    flash = {
        "id": "flash", "op": "flash_attention", "style": "fused",
        "fused": "flash_attention", "inputs": ["q", "k", "v"],
        "children": [
            {"id": "score", "op": "attention_score", "inputs": ["q", "k"],
             **declared_roles("attention_score")},
            {"id": "sm", "op": "softmax", "inputs": ["score"]},
            {"id": "apply", "op": "attention_apply", "inputs": ["sm", "v"],
             **declared_roles("attention_apply")},
        ],
    }
    # The renderer gives the fused kernel the parts its absorbed anchors
    # play, where it builds the fused node.
    flash["input_roles"] = _lifted_input_roles(flash)
    assert flash["input_roles"] == ["q", "k", "v"]
    attn = {
        "id": "self_attn", "op": "mla_attention",
        "children": [
            {"id": "x", "op": "input"},
            {"id": "q", "op": "linear", "inputs": ["x"]},
            {"id": "kv", "op": "linear", "inputs": ["x"]},
            {"id": "k", "op": "slice", "inputs": ["kv"]},
            {"id": "v", "op": "slice", "inputs": ["kv"]},
            flash,
            {"id": "o", "op": "linear", "inputs": ["flash"]},
        ],
    }
    stamp_l1([attn])
    assert attn["l1"]["available"] is True, attn["l1"].get("reason")
    boxes = {b["id"]: b for b in attn["l1"]["boxes"]}
    assert "q" in boxes and "core" in boxes and "output" in boxes


def test_stamp_l1_does_not_group_a_layer_around_its_moe_component():
    """An MoE `mlp` holds permute / combine anchors, but it is a component,
    not a fused kernel: its inner roles are not lifted onto it, so the
    decoder layer around it is not grouped as if it were one component."""
    from backend.views.graph_view import stamp_l1
    mlp = {
        "id": "mlp", "op": "moe_ffn", "inputs": ["norm"],
        "children": [
            {"id": "gate", "op": "linear", "inputs": ["norm"]},
            {"id": "permute", "op": "permute", "inputs": ["norm", "gate"],
             **declared_roles("permute")},
            {"id": "experts", "op": "swiglu_ffn", "inputs": ["permute"]},
            {"id": "combine", "op": "combine", "inputs": ["experts", "gate"],
             **declared_roles("combine")},
        ],
    }
    layer = {
        "id": "layer", "op": "decoder_layer",
        "children": [
            {"id": "x", "op": "input"},
            {"id": "norm", "op": "rmsnorm", "inputs": ["x"]},
            mlp,
            {"id": "out", "op": "add", "inputs": ["x", "mlp"]},
        ],
    }
    stamp_l1([layer])
    assert "input_roles" not in mlp
    assert "l1" not in layer
    assert mlp["l1"]["available"] is True, mlp["l1"].get("reason")


def test_stamp_l1_anchor_without_independent_totals_refuses():
    from backend.views.graph_view import stamp_l1
    attn = {"id": "self_attn", "op": "mla_attention", "children": _attn()}
    stamp_l1([attn])
    assert attn["l1"]["available"] is False
    assert "independent" in (attn["l1"].get("reason") or "")


def test_stamp_l1_uses_weight_bytes_not_storage_bytes():
    """Boxes sum `weight_bytes`; `storage_bytes` can differ by FP8 scales."""
    from backend.views.graph_view import stamp_l1
    kids = _attn()
    wb = sum(n["weight_bytes"] for n in kids)
    attn = {
        "id": "self_attn", "op": "mla_attention", "children": kids,
        "flops": sum(n["flops"] for n in kids),
        "params": sum(n["params"] for n in kids),
        "weight_bytes": wb,
        "storage_bytes": wb - 45792,
    }
    stamp_l1([attn])
    assert attn["l1"]["available"] is True, attn["l1"].get("reason")


def test_stamp_l1_folds_a_repeated_wrapper_to_one_iteration():
    """A repeated block's wrapper carries ALL repeats' totals while the
    children are ONE iteration (the vision tower's 27 blocks); the Σ gate
    compares per-iteration, so the totals fold by `repeat`."""
    from backend.views.graph_view import stamp_l1
    kids = _attn()
    repeat = 27
    attn = {
        "id": "encoder.blocks", "op": "vision_block", "children": kids,
        "flops": sum(n["flops"] for n in kids) * repeat,
        "params": sum(n["params"] for n in kids) * repeat,
        "weight_bytes": sum(n["weight_bytes"] for n in kids) * repeat,
        "repeat": repeat,
    }
    stamp_l1([attn])
    assert attn["l1"]["available"] is True, attn["l1"].get("reason")


def _wrapped(**over):
    kids = _attn()
    return {"id": "self_attn", "op": "mla_attention", "children": kids,
            "flops": sum(n["flops"] for n in kids),
            "params": sum(n["params"] for n in kids),
            "weight_bytes": 0, **over}


def test_inner_gate_does_not_compare_the_wrappers_own_weight_sum():
    """The wrapper's weight bytes are the renderer's sum of the same
    children; they are no leg of the gate (a wrong one does not refuse)."""
    from backend.views.graph_view import stamp_l1
    attn = _wrapped(weight_bytes=1)
    stamp_l1([attn])
    assert attn["l1"]["available"] is True, attn["l1"]["reason"]
    assert attn["l1"]["totals"]["weight_bytes"] is None


def test_inner_gate_compares_one_iteration_of_a_repeat_exactly():
    from backend.views.graph_view import stamp_l1
    kids = _attn()
    f, p = sum(n["flops"] for n in kids), sum(n["params"] for n in kids)
    attn = _wrapped(repeat=4, flops=4 * f, params=4 * p)
    stamp_l1([attn])
    assert attn["l1"]["available"] is True, attn["l1"]["reason"]
    assert attn["l1"]["totals"]["flops"] == f


@pytest.mark.parametrize("over", [
    {"repeat": 2.5},                          # not a count of iterations
    {"repeat": "n_experts"},
    {"repeat": 4, "flops": 4 * 31 + 1, "params": 4 * 12},   # not 4 equal iterations
])
def test_inner_gate_refuses_totals_it_cannot_state_per_iteration(over):
    from backend.views.graph_view import stamp_l1
    attn = _wrapped(**over)
    stamp_l1([attn])
    assert attn["l1"]["available"] is False
    assert attn["l1"]["reason"]


def test_a_costless_graph_groups_with_nothing_to_compare():
    """Topology without cost: the boxes are drawn, the totals say there is
    no figure (None), not 0 = 0."""
    from backend.views.graph_view import stamp_l1
    kids = [{k: v for k, v in n.items() if k not in ("flops", "params", "weight_bytes")}
            for n in _attn()]
    attn = {"id": "self_attn", "op": "mla_attention", "children": kids}
    stamp_l1([attn])
    assert attn["l1"]["available"] is True, attn["l1"]["reason"]
    assert attn["l1"]["totals"] == {"flops": None, "params": None, "weight_bytes": None}
