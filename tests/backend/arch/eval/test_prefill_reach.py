"""Which layers a long prompt runs through in prefill (`cache.prefill_reach`).

A layer runs on every prompt position when it keeps a tensor that depends on
every position, or when a layer reading its output does. A layer that keeps
only a window runs on the prompt's last positions, so past `full_from` each
further prompt token touches only the layers that run on every position.
DeepSeek-V4.1-Flash states it as "8B activated in prefill, 16B in decode":
layers past 20 keep a 128-token window and read layer 20's compressed KV.
"""
from __future__ import annotations


from backend.arch.eval.cache import prefill_reach
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from tests.snapshots import resolve_snapshot, shared_loader

_V41 = "deepseek-ai/DeepSeek-V4.1-Flash"


def _reach(resolved):
    analysis = Deployment(Program.of(resolved)).at(input_tokens=64).analyze()
    # The analysis walked its own (phase-substituted, lowered) program; the
    # counts are keyed on its roots.
    return analysis.program, analysis, prefill_reach(analysis)


def _backbone_layers(program, counts):
    """Layer index → whether it runs on every prompt position."""
    out = {}
    for template in program.blocks:
        if program.stack_of(template.roots[0]) != "":
            continue
        dense = counts[template.roots[0].uid]
        if dense not in (0, template.count):
            raise AssertionError(f"{template.name} splits {dense}/{template.count}")
        for layer in template.layer_indices:
            out[layer] = dense == template.count
    return out


def _v41():
    return resolve_snapshot(_V41, shared_loader())


def test_v41_runs_every_prompt_token_through_layers_0_to_20_only():
    program, analysis, reach = _reach(_v41())
    layers = _backbone_layers(program, reach.counts)
    assert [i for i, dense in sorted(layers.items()) if dense] == list(range(21))
    # Each tail layer needs its successor's positions plus its 127-position
    # reach back, and at least its 128-position window: 128 + 18 x 127.
    assert reach.full_from == 128 + 18 * 127
    prefill = analysis.total_for(program.multiplicity(reach.counts))
    # README: 8B activated per prompt token against 16B per decoded token
    # (the decode total here also holds the DSpark stages).
    assert prefill.active_params < 0.55 * analysis.total().active_params


def test_a_model_whose_every_cache_grows_runs_every_layer_on_every_token():
    program, _analysis, reach = _reach(
        resolve_snapshot("meta-llama/Llama-3.2-1B", shared_loader()))
    assert all(_backbone_layers(program, reach.counts).values())
    assert reach.full_from == 0
    # The sampled head reads one position, not one per prompt token.
    head = next(n for n in program.ir.top_level if n.name == "lm_head")
    assert reach.counts[head.uid] == 0


def test_an_all_window_model_runs_on_its_receptive_field_only():
    """Mistral-7B-v0.1 keeps a 4096 window in all 32 layers: nothing depends on
    every position, and the reach back is 32 windows."""
    program, _analysis, reach = _reach(
        resolve_snapshot("mistralai/Mistral-7B-v0.1", shared_loader()))
    assert not any(_backbone_layers(program, reach.counts).values())
    assert reach.full_from > 31 * 4096
