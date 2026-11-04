"""Per-invocation KV: `CacheView.storage_bytes_by_block` groups the same
entries `storage_bytes_at` sums, keyed by the block ordinal `iter_blocks`
and `Program.blocks` share."""
from __future__ import annotations

import pytest

from backend.arch.eval.cache_view import cache_view_from_analysis
from backend.arch.eval.projection import project_model
from backend.arch.loader import SpecLoader
from backend.arch.spec_tree.build import iter_blocks
from tests.snapshots import resolve_snapshot, shared_loader


@pytest.fixture(scope="module")
def loader() -> SpecLoader:
    return shared_loader()


@pytest.mark.parametrize("model_id", [
    "openai/gpt-oss-20b",            # SWA + GQA groups: two laws
    "Qwen/Qwen3-Coder-Next",         # constant-state + growing
    "deepseek-ai/DeepSeek-V3",       # MTP excluded, as in storage_bytes_at
])
def test_by_block_sums_to_the_whole(loader, model_id):
    resolved = resolve_snapshot(model_id, loader)
    view = cache_view_from_analysis(
        project_model(resolved, batch=1, input_tokens=2048).analysis)
    by_block = view.storage_bytes_by_block(2048)
    assert sum(by_block.values()) == pytest.approx(view.storage_bytes_at(2048), rel=1e-12)
    metas = iter_blocks(resolved)
    for index in by_block:
        assert index is not None and metas[index].aux_kind is None


def test_gpt_oss_groups_differ_by_window(loader):
    resolved = resolve_snapshot("openai/gpt-oss-20b", loader)
    view = cache_view_from_analysis(
        project_model(resolved, batch=1, input_tokens=2048).analysis)
    by_block = view.storage_bytes_by_block(2048)
    assert len(by_block) == 2
    # 12 windowed layers hold 128 positions, 12 full layers hold 2048
    small, large = sorted(by_block.values())
    assert large == pytest.approx(small * 2048 / 128)
