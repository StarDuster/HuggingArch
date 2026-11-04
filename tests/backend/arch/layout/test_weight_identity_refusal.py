"""The duplicate-weight-identity refusal names where both copies live.

A weight's id is its stack, block template and path. Two live model layers of
one stack that use one template hold the same id, and the analysis refuses. The message printed
`stacks '' and ''` (the backbone's internal name is empty); it names the
model layer, template and stack of each holder instead.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.arch.eval.refusal import SpecRefusal
from backend.arch.layout.persistent import physical_weight_leaves
from backend.arch.loader import SpecLoader
from backend.inference.parallel_config import ParallelConfig
from backend.inference.step_eval import evaluate_step

ARCH = Path(__file__).resolve().parents[4] / "backend" / "arch"


def test_both_holders_are_named(tmp_path):
    text = (ARCH / "models" / "llama.yaml").read_text()
    text = text.replace("match:\n  model_type: [llama]",
                        "match:\n  model_id: [test/dup-template]")
    stack = ("  - { name: layers, repeat: n_layers, template: block, "
             "inputs: [embed_tokens], streams: { hidden: { carries: true } }}\n")
    assert stack in text
    text = text.replace(stack, (
        '  - { name: layers, repeat: "2", template: block, inputs: [embed_tokens], '
        "streams: { hidden: { carries: true } }}\n"
        '  - { name: more_layers, repeat: "2", template: block, inputs: [layers], '
        "streams: { hidden: { carries: true } }}\n"))
    text = text.replace("inputs: [layers]}", "inputs: [more_layers]}")
    models = tmp_path / "models"
    models.mkdir()
    (models / "test__dup-template.yaml").write_text(text)
    config = json.loads(
        (ARCH / "snapshots" / "meta-llama__Llama-3.2-1B.json").read_text())["config"]
    loader = SpecLoader(overlay_model_dirs=[str(models)])
    resolved = loader.resolve(loader.get("test__dup-template"), config)
    step = evaluate_step(resolved, ParallelConfig(), phase="prefill",
                         batch=1, input_tokens=16, output_tokens=0)
    with pytest.raises(SpecRefusal) as caught:
        physical_weight_leaves(step.analysis)
    message = str(caught.value)
    assert "stacks ''" not in message
    assert "in layers (template block, stack 'backbone')" in message
    assert "in more_layers (template block, stack 'backbone')" in message


def test_two_assignments_of_one_template_at_different_dims_are_two_ids():
    """DeepSeek-V4.1's pure CSA consumers bind one template, with the same
    slot bindings, at compress ratio 2 (layers 3-19) and 1 (layers 21-39):
    two `per_layer.assign` entries that differ in `dims:` only. Each holds
    its own layers' weights, so each invocation carries the dim that tells
    it apart in its label, and the weight walk holds both."""
    from backend.arch.spec_tree.build import enumerate_blocks
    from tests.snapshots import resolve_snapshot

    resolved = resolve_snapshot("deepseek-ai/DeepSeek-V4.1-Flash")
    reuse = [meta for meta in enumerate_blocks(resolved) if meta.distinct_dims]
    assert {(meta.distinct_dims, meta.count) for meta in reuse} == {
        ((("compress_ratio", "2"),), 15), ((("compress_ratio", "1"),), 15)}
    assert len({meta.template_name for meta in reuse}) == 1
    for meta in reuse:
        ratio = dict(meta.distinct_dims)["compress_ratio"]
        assert meta.label.endswith(f"[compress_ratio={ratio}]")

    step = evaluate_step(resolved, ParallelConfig(), phase="prefill",
                         batch=1, input_tokens=16, output_tokens=0)
    leaves = physical_weight_leaves(step.analysis)
    program = step.analysis.program
    held = sorted(
        leaf.global_instances for leaf in leaves.values()
        if program.by_uid(leaf.node_uid).name == "hc_attn_fn"
        and "[compress_ratio=" in program.by_uid(leaf.node_uid).template)
    assert held == [15, 15]
