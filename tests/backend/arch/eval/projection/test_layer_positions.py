"""A block invocation's layers are numbered in its own stack."""
from __future__ import annotations

from backend.arch.eval.projection import project_model
from tests.snapshots import resolve_snapshot


def test_a_stack_numbers_its_own_layers():
    """Qwen3-VL's deepstack mergers are a stack of their own, indexed from
    zero in the checkpoint (`model.visual.deepstack_merger_list.0` …): merger
    j is layer j of that stack, not a continuation of the 27 vision blocks
    before it."""
    report = project_model(
        resolve_snapshot("Qwen/Qwen3-VL-235B-A22B-Instruct"))
    positions = {name: layers for (name, _count, _block), layers
                 in zip(report.block_results, report.layer_positions(),
                        strict=True)}
    assert positions["qwen35_vision_block"] == list(range(27))
    assert positions["qwen3_vl_deepstack_merger"] == [0, 1, 2]
