"""A `range:` / `layers:` selector is a literal layer table: an index outside
its container's `repeat` is refused by the one resolver every consumer shares,
not silently counted as a layer."""
from __future__ import annotations

import pytest

from backend.arch.loader import LayerSpanError, SpecLoader
from backend.arch.spec_tree.build import iter_blocks
from tests.snapshots import resolve_snapshot, shared_loader, snapshot_config

# A corpus model whose per-layer selectors are literal layer tables beside a
# `rest` template (`hybrid_override_pattern` written out as `layers:`).
NEMOTRON = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


@pytest.fixture(scope="module")
def loader() -> SpecLoader:
    return shared_loader()


def _at_layers(loader, model_id, n):
    base = resolve_snapshot(model_id, loader)
    config = {**snapshot_config(model_id), "num_hidden_layers": n}
    return loader.resolve(base.spec, config, model_id=model_id)


def test_literal_layer_table_below_the_layer_count_is_refused(loader):
    # Nemotron-3-Nano lists its attention layers 5, 12, ..., 42 (and its
    # mamba layers) literally; at 12 layers the table names layers the stack
    # does not have.
    resolved = _at_layers(loader, NEMOTRON, 12)
    with pytest.raises(LayerSpanError, match=r"layers: \[.*\].*outside this stack's 12 layers"):
        iter_blocks(resolved)


def test_growing_past_a_literal_table_fills_rest(loader):
    # The 6 listed attention and 23 listed mamba layers stay put; the 8 new
    # ones go to the `rest` (MoE) template, 23 -> 31.
    resolved = _at_layers(loader, NEMOTRON, 60)
    counts = [m.count for m in iter_blocks(resolved) if m.aux_kind is None]
    assert sorted(counts) == [6, 23, 31]


def test_expression_ranges_scale_with_the_layer_count(loader):
    # DeepSeek-V3's dense range is `[0, n_dense - 1]`; it stays inside a
    # shorter stack and the MoE `rest` shrinks.
    resolved = _at_layers(loader, "deepseek-ai/DeepSeek-V3", 30)
    counts = [m.count for m in iter_blocks(resolved) if m.aux_kind is None]
    assert counts == [3, 27]
