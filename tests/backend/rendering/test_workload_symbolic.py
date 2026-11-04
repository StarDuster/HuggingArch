"""A block graph drawn for the model, not for a run (`workload_symbolic`,
what the Inspector asks for): shape labels keep the workload axes symbolic
and the model's own dims as numbers, and a growing KV cache shows its
per-token row with no per-request or per-step size. Qwen3-8B: d_model 4096,
32 query heads and 8 KV heads of 128, so one cached K or V row is
8 x 128 x 2 bytes at the checkpoint's BF16.
"""
from __future__ import annotations

import pytest
import sympy

from backend.rendering import build_block_graph_data
from backend.arch.eval.precision import AS_SHIPPED
from tests.snapshots import resolve_snapshot


def _walk(nodes):
    for n in nodes:
        yield n
        yield from _walk(n.get("children") or [])


@pytest.fixture(scope="module")
def graphs():
    resolved = resolve_snapshot("Qwen/Qwen3-8B")
    symbolic, _ = build_block_graph_data(resolved, precision=AS_SHIPPED,
                                         workload_symbolic=True)
    concrete, _ = build_block_graph_data(resolved, 2, 64)
    return symbolic["block"], concrete["block"]


def test_shapes_keep_the_workload_axes_symbolic(graphs):
    symbolic, concrete = graphs
    nodes = {n["id"]: n for n in _walk(symbolic["nodes"])}
    assert nodes["x"]["tensor_out"] == "[B,s_query,4096]"
    labels = [n[k] for n in nodes.values() for k in ("tensor_in", "tensor_out") if n.get(k)]
    # Every activation is batch-major: each tensor of each label leads with
    # the workload's batch axis.
    tensors = [t.strip() for label in labels for t in label.split("],")]
    assert tensors and all(t.startswith("[B,") for t in tensors)
    assert any("[B,8,s_kv,128]" in label for label in labels)
    # A graph drawn for a run keeps the numbers it was drawn at.
    concrete_nodes = {n["id"]: n for n in _walk(concrete["nodes"])}
    assert concrete_nodes["x"]["tensor_out"] == "[2,64,4096]"


def test_l1_flops_keep_workload_axes_when_shapes_do():
    """Group headers must not quote a scalar FLOPs next to [B,s_query,…] shapes."""
    from backend.views.graph_view import stamp_l1
    resolved = resolve_snapshot("Qwen/Qwen3-8B")
    graphs, _ = build_block_graph_data(resolved, precision=AS_SHIPPED,
                                       workload_symbolic=True)
    stamp_l1(graphs["block"]["nodes"])
    boxed = [n for n in _walk(graphs["block"]["nodes"])
             if (n.get("l1") or {}).get("available")]
    assert boxed
    formulas = [b.get("flops_symbolic") or ""
                for n in boxed for b in n["l1"]["boxes"]
                if b["id"] != "input" and b.get("flops")]
    # Each box's work is a law in the workload: it scales with the tokens
    # the step processes, B·s_query. An attention box is written in the
    # step's pairs (`s_pairs`, ca6c76d26 "Write attention FLOPs in s_pairs
    # and retire causal_factor"), a name the phase declares in s_query
    # (`binding.STEP_LAWS`); the law is read through that declaration.
    workload = {sympy.Symbol("B"), sympy.Symbol("s_query")}
    assert formulas and all(
        workload <= _through_step_laws(sympy.sympify(f)).free_symbols
        for f in formulas), formulas


def _through_step_laws(expr):
    """`expr` with each name the phase declares (`binding.STEP_LAWS`)
    replaced by its declaration, until none is left."""
    from backend.arch.eval.binding import STEP_LAWS
    laws = {sympy.Symbol(name): sympy.sympify(text)
            for name, text in STEP_LAWS.items()}
    while expr.free_symbols & laws.keys():
        expr = expr.subs(laws)
    return expr


def test_a_growing_cache_shows_its_row_and_no_size(graphs):
    symbolic, concrete = graphs
    marks = [n["kv"] for n in _walk(symbolic["nodes"]) if n.get("kv")]
    entries = [e for m in marks for e in m["writes"] + m["reads"]]
    assert entries and all(e["bytes_per_position"] == 8 * 128 * 2 for e in entries)
    assert all(e["bytes_per_request"] is None and e["cache_len"] is None
               for e in entries)
    assert all(m["read_bytes_per_step"] is None for m in marks)
    concrete_marks = [n["kv"] for n in _walk(concrete["nodes"]) if n.get("kv")]
    assert all(m["cache_len"] == 64 and m["batch"] == 2 for m in concrete_marks)
