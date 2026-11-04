"""A report's active-parameter totals are the sums of its own rows.

The walk's active count is exact and need not be whole: a token dispatched
over a routing bank leaves an expert a fraction of itself. The report shows
whole counts, and it rounds once per accounting root — the figure that root's
row shows. Every total is then a sum of those figures: a block's over its
roots, the model's over its blocks (times their layers) and its model-level
nodes, main plus every aux stack. Flooring the exact grand total instead left
the model one parameter off the sum of its own rows.

No model in this corpus actually HAS a fractional active count, and
`test_no_active_count_is_a_floats_shadow` is the gate on that — not a
convenience. Three rows of V4.1-Flash used to look fractional, by a margin of
1e-13 and with denominators of 2^51, 2^60 and 2^67: the share of a table a
token reads had been divided with a bare `/`, and `_exact_values` then
preserved the float's error forever as the exact rational that float IS. The
model's published total was one parameter low because of it — the very
failure the paragraph above describes, back through a different door. Both
reasons a count can be whole are structural: a lookup activates
`rows per token x row width` of a `rows x width` table, and a routed share's
denominator divides the bank. So a row that is genuinely fractional is news,
and the day one appears this test is where it reports in.

Qwen3-8B is the control for the convention itself: its active counts are
whole, so the rounding moves nothing and the totals are the walk's own.
"""
from __future__ import annotations

from functools import lru_cache

import pytest

from backend.arch.eval.access import _int
from backend.arch.eval.dim import as_extent, value_of
from backend.arch.eval.projection import project_model
from tests.snapshots import resolve_snapshot


@lru_cache(maxsize=None)
def _report(model):
    return project_model(resolve_snapshot(model), batch=1, input_tokens=1)


@pytest.mark.parametrize("model", [
    "deepseek-ai/DeepSeek-V4.1-Flash", "Qwen/Qwen3-8B",
])
def test_active_totals_are_the_sums_of_the_rows(model):
    report = _report(model)
    for _name, _count, block in report.block_results:
        assert block.total_active_params == sum(
            node.active_params for node in block.nodes)
    rows = (sum(count * block.total_active_params
                for _name, count, block in report.block_results)
            + sum(node.active_params
                  for node in report.model_level_nodes.values()))
    assert report.total_active_params == rows
    assert report.total_active_params == (
        report.main_active_params + sum(report.aux_active_params.values()))
    assert report.total_prefill_active_params == (
        report.main_prefill_active_params
        + sum(report.aux_prefill_active_params.values()))


def test_no_active_count_is_a_floats_shadow():
    """V4.1-Flash's rows are whole, so the report rounds nothing away and its
    total IS the walk's.

    The model that used to be this file's fractional example, and was one
    only because a share was divided with `/`: `markov_head.embed` activates
    one row of a 129280 x 256 table and came out
    `576460752303423495/2251799813685248`, which is 256 less 3e-15 — whole,
    floored to 255, and the model's total one parameter short of the sum of
    its own rows. A fractional count is legitimate (the docstring says when);
    a count whose denominator is a power of two near 2^53 is a float that was
    made exact after the fact, so this asks for whole and says why a
    fractional one would be worth reading rather than merely rounding.
    """
    report = _report("deepseek-ai/DeepSeek-V4.1-Flash")
    analysis = report.analysis
    fractional = {
        part.node.display_path: value_of(cost.active_params)
        for part in analysis.program.contributions
        if (cost := analysis.cost.values.get(part.node.uid)) is not None
        and as_extent(cost.active_params, rounding="ceil")
        != _int(cost.active_params)}
    assert fractional == {}, fractional
    exact = analysis.total().active_params
    assert report.total_active_params == _int(exact) == 17713741778


def test_whole_active_counts_are_the_walks_own():
    """Qwen3-8B's active counts are whole: the convention moves nothing."""
    report = _report("Qwen/Qwen3-8B")
    analysis = report.analysis
    assert report.total_active_params == _int(analysis.total().active_params)
    assert report.main_active_params == _int(
        analysis.main_total().active_params)
