"""What a block binds a dim to is the frame the walk ran it in.

`BlockReport.dims` (what Compare's overlay and the module bench state a
differing dim by) and `Analysis.block_frame` (what the Playground's group
dims read) are one answer: the frame the walk opened for the block's stage —
its assignment's `dims:` over the model's. Not the first node's frame: a
block whose first node is a component call rebinding a dim (`n_experts:
n_draft_experts`) runs that call at the call's value while the block itself
binds the model's, and taking the first node walked for the block reported
the call's.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from backend.arch.eval.program import Program
from backend.arch.eval.projection.result import project_model
from backend.arch.eval.transforms import Deployment

COMPONENTS = PROJECT_ROOT / "backend" / "arch" / "components"
CONFIG = {
    "model_type": "frame_toy",
    "torch_dtype": "bfloat16", "hidden_size": 8,
    "moe_intermediate_size": 4,
    "num_experts": 12,
    "num_experts_per_tok": 4,
    "pool_num_experts": 4,
    "pool_num_experts_per_tok": 2,
    "vocab_size": 32,
}

MODEL = """\
match: { model_type: [frame_toy], model_id: [frame_toy] }
dims:
  d_model: config.hidden_size
  d_ffn_expert: config.moe_intermediate_size
  d_ffn_shared: config.moe_intermediate_size
  n_shared_experts: "1"
  n_vocab: config.vocab_size
  n_layers: "2"
  n_experts: config.num_experts
  n_experts_active: config.num_experts_per_tok
  n_pool_experts: config.pool_num_experts
  n_pool_experts_active: config.pool_num_experts_per_tok
dim_symbols:
  n_pool_experts: { kind: whole, group: moe_routing, meaning: "second pool size" }
  n_pool_experts_active: { kind: whole, group: moe_routing, meaning: "second pool top-k" }
blocks:
  blk:
    inputs: [x, input_ids]
    graph:
      pool: { op: dsv4_moe_sigmoid, in: [x, input_ids],
              dims: { n_experts: n_pool_experts,
                      n_experts_active: n_pool_experts_active } }
      ffn: { op: dsv4_moe_sigmoid, in: [x, input_ids] }
      both: { op: add, in: [pool, ffn], dim: d_model }
      o: { op: add, in: [x, both], dim: d_model }
    outputs: [o]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed_tokens, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
  - { name: layers, repeat: n_layers, template: blk, inputs: [embed_tokens, input_ids],
      streams: { hidden: { carries: true } } }
  - { name: norm, op: rmsnorm, dim: d_model, inputs: [layers],
      weight_axes: [{ replicated: true }] }
  - { name: lm_head, op: linear, boundary_role: output_head, inputs: [norm],
      dims: { in: d_model, out: n_vocab },
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""


@pytest.fixture(scope="module")
def resolved():
    from backend.arch.loader import SpecLoader
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        shutil.copytree(COMPONENTS / "library", base / "components" / "library")
        for sub in ("components/adapters", "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / "frame_toy.yaml").write_text(MODEL)
        return SpecLoader(base_dir=str(base)).match(CONFIG, model_id="frame_toy")


def test_a_block_binds_its_frame_not_its_first_calls(resolved):
    analysis = Deployment(Program.of(resolved)).analyze()
    block = analysis.program.blocks[0]
    pool = next(n for n in block.roots if n.name == "pool")
    # The case discriminates: the first node walked runs at the pool's count.
    assert int(analysis.node_dims.values[pool.uid]["n_experts"]) == 4
    # The block itself binds the model's.
    assert int(analysis.block_frame(block)["n_experts"]) == 12
    report = project_model(resolved).block_results[0][2]
    assert report.dims["n_experts"] == 12
    assert report.dims["n_experts_active"] == 4
