"""A model-level read of a stack's inner stream, on the serving graph.

`inputs: [<stream>]` on a model-level stage reads the stream's FINAL value.
Which write that is, is the walk's answer: `Program.final_inner_stream`
resolves the stream in the reader's own scope and takes the maximum writing
layer. The graph draws the edge from the box holding that write — not from
whichever of the stack's templates is drawn last, and not through one
endpoint per stream NAME, because a name is unique only inside its scope.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.loader import SpecLoader
from backend.inference.framework_bindings import get_binding
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from backend.rendering._compute_graph import serialize_post_tree
from backend.rendering.kv_marks import KVMarks
from tests.snapshots import resolve_snapshot, shared_loader


def _serving(resolved, phase="decode"):
    result = analyze(
        resolved, parallel=ParallelConfig.from_dict({}),
        framework_binding=get_binding("vllm"), batch_size=1,
        avg_input_tokens=4, avg_output_tokens=0, phase=phase, breakdown=False)
    graph = serialize_post_tree(
        result.post_tree, resolved, pre_tree=result.comm_tree,
        fusion_matches=result.fusion_matches, program=result.program,
        shapes=result.shapes, full_shapes=result.full_shapes,
        pre_program=result.pre_program, pre_shapes=result.pre_shapes,
        pre_full_shapes=result.pre_full_shapes,
        kv=KVMarks.of_serving(result))
    return result, graph


def _holders(post_tree):
    """Which block box holds each node of the walk."""
    out = {}
    for meta, roots in post_tree.blocks:
        pending = list(roots)
        while pending:
            node = pending.pop()
            out[node.uid] = f"__block__{meta.label}"
            pending.extend(node.children or ())
    return out


def _sources(graph, dst):
    return {e["src"] for e in graph["edges"] if e["dst"] == dst}


def test_the_edge_leaves_the_block_that_holds_the_final_write():
    """DeepSeek-V4.1-Flash: the model-level `hc_exit` reduces the backbone's
    final `hc_pre_mix`, so its edge leaves the block holding layer 39's
    write. The stack draws one box per `assignments:` template and the two
    differ here — the write is in the `[compress_ratio=1]` selection
    template while the box drawn last is the shared-keys one — so a renderer
    reading drawing order fails this."""
    resolved = resolve_snapshot("deepseek-ai/DeepSeek-V4.1-Flash",
                                shared_loader())
    result, graph = _serving(resolved)
    [exit_node] = [n for n in result.post_tree.top_level
                   if n.name == "hc_exit"]
    final = result.program.final_inner_stream(
        tuple(exit_node.layer_path), "hc_pre_mix")
    assert final is not None
    holders = _holders(result.post_tree)
    assert holders[final.uid] in _sources(graph, "hc_exit")

    of_stack = {f"__block__{meta.label}" for meta, _roots
                in result.post_tree.blocks if meta.stack_name == "layers"}
    drawn_order = [n["id"] for n in graph["nodes"] if n["id"] in of_stack]
    assert len(drawn_order) > 1
    assert drawn_order[-1] != holders[final.uid]


#: Two aux heads, each with a stack publishing an inner stream called `s`
#: and a stage of its own reading it. Legal: `ArchSpec.inner_stream_named`
#: resolves `s` per reader scope and refuses only two declarations in ONE
#: scope. A graph that kept one endpoint per stream name would refuse this
#: whole model with "`s` names two endpoints of the graph".
_TWO_HEADS = """
match: { model_type: [_two_inner_streams] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size }
blocks:
  prod:
    stream_out: { s: mk }
    graph: { mk: { op: rmsnorm, in: x, dim: d_model,
                   weight_axes: [{ replicated: true }] } }
    outputs: [mk]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids],
      boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
  - name: head_a
    group: true
    aux_kind: mtp
    checkpoint: { prefix: model.head_a, index: local }
    inputs: [embed]
    layers:
      - { name: layers_a, repeat: n_layers, template: prod, inputs: [embed],
          streams: { hidden: { carries: true },
                     s: { shape: [B, s_query, d_model] } } }
      - { name: mix_a, op: mul, dim: d_model, inputs: [layers_a, s] }
  - name: head_b
    group: true
    aux_kind: mtp
    checkpoint: { prefix: model.head_b, index: local }
    inputs: [embed]
    layers:
      - { name: layers_b, repeat: n_layers, template: prod, inputs: [embed],
          streams: { hidden: { carries: true },
                     s: { shape: [B, s_query, d_model] } } }
      - { name: mix_b, op: mul, dim: d_model, inputs: [layers_b, s] }
  - { name: head, op: linear, boundary_role: output_head, inputs: [embed],
      dims: { in: d_model, out: n_vocab },
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""

_TWO_HEADS_CONFIG = {"torch_dtype": "bfloat16", "hidden_size": 8,
                     "num_hidden_layers": 2, "vocab_size": 16,
                     "model_type": "_two_inner_streams"}


def _resolve_text(spec: str, config: dict):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(spec)
        path = f.name
    loader = SpecLoader()
    try:
        model_spec = loader.load_model_spec(path)
    finally:
        os.unlink(path)
    return loader.resolve(model_spec, config)


def test_two_scopes_may_name_one_stream_and_each_reads_its_own():
    """Each head's stage reads the stream of ITS OWN stack: the name is
    unique in its scope, so the graph draws two edges, one per head, and
    neither reader is pointed at the other's write."""
    resolved = _resolve_text(_TWO_HEADS, _TWO_HEADS_CONFIG)
    _result, graph = _serving(resolved, phase="prefill")
    assert _sources(graph, "mix_a") == {"__block__prod@head_a"}
    assert _sources(graph, "mix_b") == {"__block__prod@head_b"}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
