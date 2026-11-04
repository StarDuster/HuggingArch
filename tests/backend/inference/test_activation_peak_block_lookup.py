"""A lookup table inside a block reads token ids, not an activation.

The block takes the token ids as a formal of their own (Qwen3.8-Flash-Next's
PLE n-gram bank, `layers.1.ple.ple_embedding.ngram_embedding.shard_0`, reads
`input_ids` beside the hidden stream), and the activation replay prices the
lookup on that edge.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

SPEC = """\
match: { model_type: [blk_lookup], model_id: [blk_lookup] }
dims:
  d_model: config.hidden_size
  n_layers: "2"
blocks:
  blk:
    inputs: [x, input_ids]
    graph:
      table: { op: embedding, in: input_ids, dims: { n_vocab: "16", d_model: d_model }, weight_axes: [{ replicated: true }, { replicated: true }] }
      o: { op: add, in: [x, table], dim: d_model }
    outputs: [o]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: "32", d_model: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
  - { name: layers, repeat: n_layers, template: blk, inputs: [embed, input_ids],
      streams: { hidden: { carries: true } } }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: "32" }, inputs: [layers],
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""


def _resolved():
    from backend.arch.loader import SpecLoader
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / "blk_lookup.yaml").write_text(SPEC)
        return SpecLoader(base_dir=str(base)).match(
            {"model_type": "blk_lookup", "torch_dtype": "bfloat16", "hidden_size": 8},
            model_id="blk_lookup")


def test_a_block_level_lookup_of_the_token_ids_has_a_peak():
    from backend.inference.step_eval import estimate_peak_activation
    out = estimate_peak_activation(
        _resolved(), batch=1, seq_len=16)
    assert out["peak_activation_bytes_per_rank"] > 0
