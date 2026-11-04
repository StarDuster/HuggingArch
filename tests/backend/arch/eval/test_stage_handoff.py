"""A stage hands on its output settled, and every later stage reads that.

A vocab-parallel embedding leaves a pending sum; its stage settles it with one
all_reduce at the boundary (`Evaluator._closed`). A later model-level op whose
`inputs:` names the embedding resolves to the embedding node itself, and it
used to read the node's raw, still-pending output — so the same reduction was
priced a second time on that op's edge (DeepSeek-V4's `hc_expand`, the Qwen3.5
MTP's `pre_fc_norm_embedding`). One tensor, one reduction.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

SPEC = """\
match: { model_type: [handoff], model_id: [handoff] }
dims:
  d_model: config.hidden_size
  n_vocab: "64"
  n_layers: "1"
blocks:
  blk:
    graph:
      n: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
      o: { op: add, in: [x, n], dim: d_model }
    outputs: [o]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model },
      weight_axes: [{ world: attention, parts: n_vocab, fit: pad }, { replicated: true }] }
  - { name: pre, op: rmsnorm, dim: d_model, weight_axes: [{ replicated: true }],
      inputs: [embed] }
  - { name: layers, repeat: n_layers, streams: { hidden: { carries: true } }, template: blk, inputs: [pre] }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: d_model }, inputs: [layers],
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""


def test_the_embedding_sum_is_reduced_once():
    from backend.arch.loader import SpecLoader
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.pipeline import analyze
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / "handoff.yaml").write_text(SPEC)
        resolved = SpecLoader(base_dir=str(base)).match(
            {"model_type": "handoff", "torch_dtype": "bfloat16", "hidden_size": 8}, model_id="handoff")
    result = analyze(resolved, parallel=ParallelConfig(tp_attention=2,
                                                       tp_dense_ffn=2),
                     batch_size=1, avg_input_tokens=16, avg_output_tokens=0, phase="prefill", breakdown=False)
    nodes = {n.name: n for n in result.program.nodes()}
    embed, pre = nodes["embed"], nodes["pre"]
    # The embedding's own output is the pending sum ...
    assert result.analysis.states.values[embed.uid].value is not None
    # ... settled once, where its stage hands it on,
    moves = [r for r in result.analysis.redistributes if r.collectives]
    assert [(r.node, r.input_index) for r in moves
            if embed.uid in (r.node, r.producer)] == [(embed.uid, -1)]
    # and the reader sees the settled tensor: nothing moves on its edge.
    assert result.analysis.input_states[(pre.uid, 0)].value is None
    assert not [r for r in moves if r.node == pre.uid]
