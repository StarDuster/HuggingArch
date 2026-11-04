"""The share of itself a node runs per token, stated once.

`Analysis.active_share` is the one carrier: the copies that run over the
copies held, times the share of ONE copy a leaf runs where it runs only part
of itself (a routed bank's `n_experts_active / n_route_targets` of its expert
axis). Two readings used to carry it — the parameters a token touches AND a
published copy COUNT that had the fraction multiplied into it — and both
readers of the count divided the fraction back out to get at the share.

So this pins the three things that make one carrier one carrier: `instances`
is a whole count of copies, `active_share` is the exact share, and
`active_params` is `params` times that share. A leaf that runs all of itself
has a share of 1 and no third reading.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from tests.snapshots import shared_program, shared_walk

#: Qwen3-30B-A3B routes 8 of 128 destinations, and its destinations are its
#: expert copies, so a token runs 1/16 of every bank leaf.
_MODEL = "Qwen/Qwen3-30B-A3B"
_SHARE = Fraction(8, 128)


def _analysis():
    return (shared_walk(Deployment(shared_program(_MODEL))
            .at(batch=1, input_tokens=256)))


def _bank(analysis, path="mlp.experts.gate_proj"):
    return next(node for node in analysis.program.nodes()
                if node.display_path == path)


def test_a_bank_leafs_share_is_the_routers_exactly():
    analysis = _analysis()
    uid = _bank(analysis).uid
    assert analysis.active_share.values[uid] == _SHARE
    # And it is a SHARE, not a count with a fraction folded in: the copies the
    # model holds of this leaf are whole (48 layers).
    assert analysis.instances.values[uid] == 48


def test_the_parameters_a_token_touches_are_that_share_of_them():
    analysis = _analysis()
    cost = analysis.cost[_bank(analysis).uid]
    assert cost.params > 0
    assert cost.active_params == cost.params * _SHARE


def test_a_leaf_that_runs_all_of_itself_has_a_share_of_one():
    analysis = _analysis()
    uid = _bank(analysis, "self_attn.q_proj").uid
    assert analysis.active_share.values[uid] == 1
    cost = analysis.cost[uid]
    assert cost.active_params == cost.params


def test_every_node_of_the_walk_states_its_share():
    # A share nothing states is a hole a reader fills with a default, which is
    # how the fraction came to be divided back out of a count.
    analysis = _analysis()
    missing = [analysis.program.by_uid(uid).display_path
               for uid in analysis.instances.values
               if uid not in analysis.active_share.values]
    assert missing == []
    assert all(0 < share <= 1
               for share in analysis.active_share.values.values())


# -- the other half of the share: copies that run, of copies held -------------
#
# No spec in the corpus declares `active:` today (the expert banks that used to
# are shapes now), so the corpus cannot tell a count of copies apart from a
# count of the copies that RUN. A four-copy node of which one runs per token
# can, and that is what `active:` means.

_WIDTH, _VOCAB, _COPIES, _RUNS = 64, 1000, 4, 1

_SYNTHETIC = f"""\
match: {{ model_type: [ash], model_id: [ash] }}
dims:
  n_vocab: config.vocab_size
  d_model: config.hidden_size
blocks:
  blk:
    inputs: [x]
    graph:
      bank: {{ op: linear, in: x, repeat: "{_COPIES}", active: "{_RUNS}",
               dims: {{ in: d_model, out: d_model }},
               weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      res: {{ op: add, in: [x, bank], dim: d_model }}
    outputs: [res]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: table, op: embedding, inputs: [input_ids],
      boundary_role: input_embedding,
      dims: {{ n_vocab: n_vocab, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - {{ name: layers, template: blk, repeat: "1", inputs: [table],
      streams: {{ hidden: {{ carries: true }} }} }}
  - {{ name: head, op: linear, boundary_role: output_head, inputs: [layers],
      dims: {{ in: d_model, out: n_vocab }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""


def _synthetic():
    import tempfile
    from backend.arch.loader import SpecLoader

    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / "ash.yaml").write_text(_SYNTHETIC)
        resolved = SpecLoader(base_dir=str(base)).match(
            {"model_type": "ash", "vocab_size": _VOCAB,
             "hidden_size": _WIDTH, "torch_dtype": "bfloat16"},
            model_id="ash")
    return (shared_walk(Deployment(Program.of(resolved))
            .at(batch=1, input_tokens=8)))


def test_copies_held_and_the_share_of_them_that_runs_are_two_answers():
    analysis = _synthetic()
    uid = next(node.uid for node in analysis.program.nodes()
               if node.display_path.endswith("bank"))
    # Four copies are HELD, and `instances` is that count whole — not the
    # count with the running share folded into it.
    assert analysis.instances.values[uid] == _COPIES
    assert analysis.active_share.values[uid] == Fraction(_RUNS, _COPIES)
    cost = analysis.cost[uid]
    assert cost.params == _WIDTH * _WIDTH * _COPIES
    assert cost.active_params == cost.params * Fraction(_RUNS, _COPIES)
