"""The whole-model graph draws what the model receives as boundary inputs.

A model input (`model_inputs:`) has no IR node: nothing in the graph
produces it. Its readers name it all the same, so the graph draws one input
node per model input the drawn step reads, with the tensor its readers
receive, and an edge into each reader — never a dangling edge, never a
shapeless one. The prefill step reads the token ids; an image is read by the
encoder, a walk of its own (`Program.walked_inputs`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.framework_bindings import get_binding
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from backend.rendering._compute_graph import serialize_post_tree
from backend.rendering.kv_marks import KVMarks
from tests.snapshots import resolve_snapshot


def _graph(model_id: str):
    resolved = resolve_snapshot(model_id)
    res = analyze(resolved, parallel=ParallelConfig(),
                  framework_binding=get_binding("vllm"),
                  batch_size=1, avg_input_tokens=4, avg_output_tokens=0, phase="prefill", breakdown=False)
    return serialize_post_tree(
        res.post_tree, resolved, pre_tree=res.comm_tree,
        fusion_matches=res.fusion_matches, program=res.program,
        shapes=res.shapes, full_shapes=res.full_shapes,
        pre_program=res.pre_program, pre_shapes=res.pre_shapes,
        pre_full_shapes=res.pre_full_shapes,
        kv=KVMarks.of_serving(res))


@pytest.mark.parametrize("model_id,reads", [
    ("meta-llama/Llama-3.2-1B", {"input_ids": ("embed_tokens", "[1,4]")}),
    ("moonshotai/Kimi-K2.5", {
        "input_ids": ("language_model.model.embed_tokens", "[1,4]")}),
])
def test_each_model_input_is_a_shaped_boundary_input(model_id, reads):
    graph = _graph(model_id)
    nodes = {n["id"]: n for n in graph["nodes"]}
    for name, (reader, shape) in reads.items():
        node = nodes[name]
        assert node["op"] == "input" and not node["inputs"]
        assert node["tensor_out"] == shape
        edges = [e for e in graph["edges"] if e["src"] == name]
        assert [(e["dst"], e["shape"]) for e in edges] == [(reader, shape)]


def test_the_step_draws_no_encoder_input():
    """Kimi-K2.5's image is the encode walk's input, not the step's."""
    nodes = {n["id"] for n in _graph("moonshotai/Kimi-K2.5")["nodes"]}
    assert "pixel_values" not in nodes
