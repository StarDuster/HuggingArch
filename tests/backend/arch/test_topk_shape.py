"""Top-k shape contracts for generic routing and DSA selection."""

import json
from pathlib import Path

import sympy as sp

from backend.arch.axis_propagation import (
    compute_shape_out, infer_shape, shape_consistency_errors,
)
from tests.snapshots import shared_loader
from backend.arch.ops import get_op
from backend.rendering import build_block_graph_data


class _Source:
    shape = (1, 1, 1)
    out_shape = (1, 1, 1)


class _Input:
    src = _Source()


class _Node:
    shape = (1, 1, 1)
    out_shape = (1, 1, 1)
    inputs_obj = (_Input(),)
    axis = None


def _walk(nodes):
    for node in nodes:
        yield node
        yield from _walk(node.get("children") or [])


def test_topk_clamps_numeric_and_symbolic_extent():
    op = get_op("topk")
    assert compute_shape_out(op, [(2, 3, 4)], {
        "in": 4, "k": 8,
    }) == (2, 3, 4)
    assert compute_shape_out(op, [(2, 3, 4)], {
        "in": 4, "k": 2,
    }) == (2, 3, 2)

    width, sel = sp.symbols("width sel", positive=True, integer=True)
    assert compute_shape_out(op, [(2, 3, width)], {
        "in": width, "k": sel,
    }) == (2, 3, sp.Min(sel, width))


def test_axis_propagation_agrees_with_topk_clamp():
    dims = {"k": 2048}
    assert infer_shape("dsa_topk_mask", _Node(), [(1, 1, 1)], dims) == (1, 1, 1)
    assert shape_consistency_errors(
        "dsa_topk_mask", _Node(), [(1, 1, 1)], (1, 1, 2048),
    ) == ["dim 2: selection from in[0][2]=1 grew to out=2048"]


def test_deepseek_v32_short_context_renders_effective_topk():
    root = Path(__file__).resolve().parents[3]
    snapshot = json.loads((
        root / "backend/arch/snapshots/deepseek-ai__DeepSeek-V3.2.json"
    ).read_text())
    resolved = shared_loader().match(snapshot["config"], model_id="deepseek-ai/DeepSeek-V3.2")
    graph, _ = build_block_graph_data(resolved, batch_size=1, seq_len=1)

    topk_nodes = [
        node
        for block in graph.values()
        for node in _walk(block["nodes"])
        if node.get("op") == "dsa_topk_mask"
    ]

    assert topk_nodes
    assert {node["tensor_in"] for node in topk_nodes} == {"[1,1,1]"}
    assert {node["tensor_out"] for node in topk_nodes} == {"[1,1,1]"}
    assert all("Min(" in node["tensor_out_symbolic"] for node in topk_nodes)
