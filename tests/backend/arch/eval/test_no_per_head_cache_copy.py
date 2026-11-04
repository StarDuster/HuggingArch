"""Cache-resident data is shown to the query heads as a view, never copied.

The domain fact: an MLA kernel run absorbed (FlashMLA, its sparse variant,
FlashInfer MLA) reads one KV head — vLLM passes `num_heads_k=1` and a
`(..., 1, d_kv_lora + d_qk_rope)` cache — and every query head scores against
that one row. An attention that keeps one key per position for all its heads
(MLA's latent, MQA's key) hands it to the heads through a stride-0 broadcast;
no kernel writes `n_heads` copies of a cached position.

The graph form of the mistake: a node that is not a view, whose output is
cache-resident data (`cache.context_sources`: the cached tensors closed over
re-arrangements and joins of them), carrying the `n_heads` axis — a join of
two head-broadcasts, which writes the broadcast out. Every absorbed MLA
component joined its latent and RoPE key that way, and the step billed
`n_heads x (d_kv_lora + d_qk_rope)` bytes per cached position as a write
(about 280 GB at DeepSeek-V3.2's 32K prefill).

The cached tensors themselves are exempt: a per-head K an MHA layer caches is
a real tensor of that shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.cache import (
    Edges, cached_node_uids, context_sources, is_view)
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import DECODE, PREFILL, Deployment
from backend.arch.snapshots import iter_snapshot_ids
from tests.snapshots import resolve_snapshot, shared_loader

pytestmark = pytest.mark.regression  # walks every snapshot with an absorbed form



def per_head_cache_copies(resolved, phase: str):
    """Display paths of the non-view nodes that write cache-resident data
    with an `n_heads` axis, in `phase`'s walk (the form the phase runs)."""
    deployment = Deployment(Program.of(resolved)).at(batch=1, input_tokens=64)
    if phase == DECODE:
        deployment = deployment.during(DECODE, cache_len=64)
    analysis = deployment.analyze()
    program = analysis.program
    seeds = cached_node_uids(program)
    out = set()
    for uid in context_sources(Edges(analysis)):
        node = program.by_uid(uid)
        if uid in seeds or node.children or is_view(node):
            continue
        shape = analysis.shapes.values.get(uid)
        if shape is not None and shape.out and any(
                "n_heads" in names for names in shape.names()):
            out.add(node.display_path)
    return sorted(out)


@pytest.mark.parametrize("phase", (PREFILL, DECODE))
# Every snapshot, by file name only; which ones the test applies to is
# asked of the resolved spec in the test, so collection resolves nothing.
@pytest.mark.parametrize("model_id", sorted(iter_snapshot_ids()))
def test_absorbed_mla_joins_its_cache_once(model_id, phase) -> None:
    resolved = resolve_snapshot(model_id, shared_loader())
    spec = resolved.spec
    bound = {**(spec.components or {}), **(spec.blocks or {})}
    if not any(block.decode_variant for block in bound.values()):
        pytest.skip("binds no component with an absorbed decode form")
    assert per_head_cache_copies(resolved, phase) == []
