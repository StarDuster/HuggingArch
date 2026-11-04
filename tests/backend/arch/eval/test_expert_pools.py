"""A model that routes over more than one expert pool.

DeepSeek-V4.1-Flash's draft stages route over their own pool — 128 experts,
top-3 — while its backbone routes over 384, top-6 (model.py:142-149
`get_moe_config`). The pool sizes are bound where the draft blocks are
placed (the block assignment's `dims:`), so every count the routing needs is
read where each bank sits:

  - `Analysis.moe_pools` lists each pool with its own numbers, the backbone's
    first, and `Analysis.moe` is the backbone's;
  - expert parallelism divides each pool's copies by the degree, so a rank
    holds `copies / ep` of each;
  - a degree is a deployment only when it divides every pool.
"""
from __future__ import annotations

import gc
import shutil
import tempfile
from pathlib import Path

import pytest

from backend.arch import dims as dim_registry
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.inference.parallel_sweep import (
    parallel_configs_for_mode, sweep_caps)

COMPONENTS = Path(__file__).resolve().parents[4] / "backend" / "arch" / "components"
CONFIG = {
    "model_type": "pools_toy",
    "torch_dtype": "bfloat16", "hidden_size": 8,
    "moe_intermediate_size": 4,
    "num_experts": 12,
    "num_experts_per_tok": 4,
    "draft_num_experts": 4,
    "draft_num_experts_per_tok": 2,
    "vocab_size": 32,
}

MODEL = """\
match: { model_type: [pools_toy], model_id: [pools_toy] }
dims:
  d_model: config.hidden_size
  d_ffn_expert: config.moe_intermediate_size
  d_ffn_shared: config.moe_intermediate_size
  n_shared_experts: "1"
  n_vocab: config.vocab_size
  n_layers: "2"
  n_experts: config.num_experts
  n_experts_active: config.num_experts_per_tok
  n_draft_experts: config.draft_num_experts
  n_draft_experts_active: config.draft_num_experts_per_tok
dim_symbols:
  n_draft_experts: { kind: whole, group: mtp_speculative, meaning: "draft pool size" }
  n_draft_experts_active: { kind: whole, group: mtp_speculative, meaning: "draft pool top-k" }
blocks:
  blk:
    inputs: [x, input_ids]
    graph:
      ffn: { op: dsv4_moe_sigmoid, in: [x, input_ids] }
      o: { op: add, in: [x, ffn], dim: d_model }
    outputs: [o]
  draft_blk:
    inputs: [x, input_ids]
    graph:
      ffn: { op: dsv4_moe_sigmoid, in: [x, input_ids] }
      o: { op: add, in: [x, ffn], dim: d_model }
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
  - name: draft
    group: true
    aux_kind: mtp
    checkpoint: { prefix: mtp, index: local }
    inputs: [norm, input_ids]
    layers:
      - name: mtp
        inputs: [norm, input_ids]
        repeat: "1"
        streams: { hidden: { carries: true } }
        per_layer:
          assign:
            - { type: d, rest: true }
          map:
            d:
              template: draft_blk
              dims: { n_experts: n_draft_experts,
                      n_experts_active: n_draft_experts_active }
"""


def _resolve():
    from backend.arch.loader import SpecLoader
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        shutil.copytree(COMPONENTS / "library", base / "components" / "library")
        for sub in ("components/adapters", "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / "pools_toy.yaml").write_text(MODEL)
        return SpecLoader(base_dir=str(base)).match(CONFIG, model_id="pools_toy")


@pytest.fixture(scope="module")
def resolved():
    return _resolve()


def test_each_pool_carries_its_own_numbers(resolved):
    analysis = Deployment(Program.of(resolved)).analyze()
    backbone, draft = analysis.moe_pools
    assert (backbone.n_expert_copies, backbone.active_copies,
            backbone.n_experts, backbone.n_experts_active) == (12, 4, 12, 4)
    assert (draft.n_expert_copies, draft.active_copies,
            draft.n_experts, draft.n_experts_active) == (4, 2, 4, 2)
    assert analysis.moe == backbone


def test_expert_parallelism_divides_each_pool_by_its_own_count(resolved):
    from backend.arch.layout.physical_cost import physical_cost_plan

    program = Program.of(resolved)
    analysis = Deployment(program).at(
        batch=1, input_tokens=4).on(ep=2).analyze()
    local = physical_cost_plan(analysis).cost.values
    banks = {program.stack_of(n) or "backbone": n.uid for n in program.nodes()
             if n.name == "gate_proj" and n.parent is not None
             and n.parent.name == "experts"}
    # Each pool's experts are the leading axis of its weights, and a rank
    # holds half of each: 12 / 2 and 4 / 2 experts of 4 x 8.
    assert [int(e) for e in analysis.storage.values[banks["backbone"]][0].shape] == [12, 4, 8]
    assert [int(e) for e in analysis.storage.values[banks["draft"]][0].shape] == [4, 4, 8]
    assert int(local[banks["backbone"]].params) == 6 * 4 * 8
    assert int(local[banks["draft"]].params) == 2 * 4 * 8


def test_a_degree_is_a_deployment_only_when_it_divides_every_pool(resolved):
    caps = sweep_caps({}, resolved=resolved)
    assert [pool.n_expert_copies for pool in caps.moe_pools] == [12, 4]
    assert caps.ep_ceiling == 4
    assert caps.ep_divides(2) and caps.ep_divides(4)
    assert not caps.ep_divides(3)          # divides 12, not 4
    degrees = [x for x, _ in parallel_configs_for_mode("dp_ep", caps)]
    assert degrees == [2, 4]


def test_a_resolution_keeps_its_spec_symbols_registered():
    """A spec's own `dim_symbols` live as long as the loader that installed
    them; a resolution outlives the loader that made it, and holds it."""
    resolved = _resolve()
    gc.collect()
    assert resolved.declared_by is not None
    assert dim_registry.get("n_draft_experts") is not None
