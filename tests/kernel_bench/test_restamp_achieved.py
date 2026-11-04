"""A lake row's achieved rates follow today's walk, joined on the physical cell.

`tools.restamp_achieved` re-plans every (model, GPU) and re-stamps a row
whose numerators today's plan moves. The join is the physical cell — any
model's plan on the row's GPU may answer it, whatever deployment cut the
shape — and a cell today prices two ways, or not at all, is left alone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_ENV = {"activation": "bf16", "output": "bf16"}


def _plan(cache, model, gpu, surface, cells):
    from tools.restamp_achieved import _plan_file, _provenance_file
    path = _plan_file(cache, model, gpu, surface)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cells))
    _provenance_file(path).write_text("[]")


def _group(model, dims, old_bytes, *, op="permute", gpu="H100-SXM", n=3,
           env=None, mbu=1.05, parallel="tp=1", node_spec_id="A|n",
           phase="decode"):
    return {"gpu": gpu, "model": model, "op": op,
            # Stored text whose keys are not in sorted order: the join
            # compares cells, not spellings.
            "shape_json": json.dumps(dict(reversed(list(dims.items())))),
            "dtype_env_json": json.dumps(env or _ENV), "phase": phase,
            "parallel": parallel, "node_spec_id": node_spec_id,
            "n": n, "old_flops": 0.0, "old_bytes": old_bytes,
            "max_mbu": mbu, "max_mfu": 0.0, "max_p50_us": 10.0}


def test_rows_are_restamped_from_the_cell_today_prices(tmp_path):
    from tools.restamp_achieved import SURFACES, cell_of, plan_restamp
    gather = {"B": 1333, "d_model": 2048, "n_tokens": 1256}
    copy = {"B": 1, "d_model": 4096}
    split = {"B": 2, "d_model": 4096}
    # Model A's plan prices the gather (now its picked rows) and the copy;
    # model B's plan prices the split cell differently in two of its jobs.
    a = [[*cell_of("permute", gather, _ENV, "decode"), [[0.0, 10_300_000.0]]],
         [*cell_of("permute", copy, _ENV, "decode"), [[0.0, 16_384.0]]]]
    b = [[*cell_of("permute", split, _ENV, "decode"), [[0.0, 1.0], [0.0, 2.0]]]]
    for surface in SURFACES:
        _plan(tmp_path, "A", "H100-SXM", surface, a if surface == "routine" else [])
        _plan(tmp_path, "B", "H100-SXM", surface, b if surface == "routine" else [])
        _plan(tmp_path, "A", "RTX-4090", surface, [])
    from tools.restamp_achieved import _plan_file, _provenance_file
    split_cell = cell_of("permute", split, _ENV, "decode")
    _provenance_file(_plan_file(tmp_path, "B", "H100-SXM", "routine")).write_text(
        json.dumps([["B", "tp=1", "A|n", *split_cell,
                     [[0.0, 1.0], [0.0, 2.0]]]]))
    groups = [
        # Measured for B, priced only by A's plan: the cell answers it.
        _group("B", gather, 38_000_000.0),
        _group("A", copy, 16_384.0, n=5),
        _group("B", split, 1.0, n=7),
        _group("A", {"B": 9, "d_model": 9}, 1.0, n=11),
        # The same cell on another card: no plan on that card answers it.
        _group("A", gather, 38_000_000.0, gpu="RTX-4090", n=13),
    ]
    (rewrites, counts, refused, moved, _unplanned) = plan_restamp(
        groups, tmp_path, workers=1)
    assert refused == {}
    assert dict(counts) == {"restamped": 3, "unchanged": 5, "ambiguous": 7,
                            "unplanned": 24}
    assert len(rewrites) == 1
    assert rewrites[0]["shape_json"] == groups[0]["shape_json"]
    assert rewrites[0]["access_bytes_per_rank"] == 10_300_000.0
    assert abs(moved[0]["new_mbu"] - 1.05 * 10.3 / 38) < 1e-9


def test_a_numerator_inside_the_columns_rounding_is_not_moved():
    # 0.0001 TFLOPS at 5.3 us reads back as 530 FLOPs; billed 370, the
    # stored column could not have said otherwise.
    from tools.restamp_achieved import _READBACK, _moved
    slack = _READBACK["flops"] * 5.3
    assert not _moved(530.0, 370.0, slack)
    assert _moved(38_000_000.0, 10_300_000.0, _READBACK["bytes"] * 10.0)


def test_two_plans_that_differ_only_by_rounding_are_one_price(tmp_path):
    from tools.restamp_achieved import _fold_pairs
    assert _fold_pairs([(0.0, 1.0), (0.0, 1.000000001)]) == {(0.0, 1.0)}
    assert len(_fold_pairs([(0.0, 1.0), (0.0, 2.0)])) == 2


def test_a_boundary_ambiguous_cell_is_resolved_by_its_own_node(tmp_path):
    """One physical mHC cell has two boundary forms: the first seam's input
    is one stored copy fanned out to every hidden copy, while later seams
    carry distinct copies. The lake row names its node; a plan cache with
    that provenance can answer its own row without choosing for the other
    form."""
    from tools.restamp_achieved import (SURFACES, _plan_file, _provenance_file,
                                        cell_of, plan_restamp)
    dims = {"B": 2, "d_model": 64, "n_hidden_copies": 4, "s_query": 3}
    cell = cell_of("mhc_pre_big_fuse", dims, _ENV, "decode")
    for surface in SURFACES:
        _plan(tmp_path, "A", "H100-SXM", surface,
              [[*cell, [[0.0, 1.0], [0.0, 2.0]]]]
              if surface == "routine" else [])
    provenance = [[ "A", "tp=1", "model|first-seam", *cell, [[0.0, 1.0]] ]]
    _provenance_file(_plan_file(tmp_path, "A", "H100-SXM", "routine")
                     ).write_text(json.dumps(provenance))
    group = _group("A", dims, 2.0, op="mhc_pre_big_fuse",
                   node_spec_id="model|first-seam")
    rewrites, counts, _refused, moved, _unplanned = plan_restamp(
        [group], tmp_path, workers=1)
    assert dict(counts) == {"restamped": 3}
    assert rewrites[0]["model"] == "A"
    assert rewrites[0]["parallel"] == "tp=1"
    assert rewrites[0]["node_spec_id"] == "model|first-seam"
    assert rewrites[0]["access_bytes_per_rank"] == 1.0
    assert moved[0]["resolved_provenance"] is True


def test_a_prefill_row_without_its_prefix_axis_takes_its_fresh_point(tmp_path):
    """Kernel Bench's historical prefill tracks never carry a cached prefix.
    When an own-point pass offers under-keyed KDA cells at several prefixes,
    the fresh point is the one the measured row came from."""
    from tools.restamp_achieved import (SURFACES, _plan_file, _provenance_file,
                                        cell_of, plan_restamp)
    dims = {"B": 2, "d_lin_k_head": 8, "d_lin_v_head": 8,
            "n_lin_k_heads": 2, "n_lin_v_heads": 2, "s_query": 16}
    cell = cell_of("kda", dims, _ENV, "prefill")
    for surface in SURFACES:
        _plan(tmp_path, "A", "H100-SXM", surface,
              [[*cell, [[0.0, 10.0], [0.0, 12.0]]]]
              if surface == "routine" else [])
    provenance = [
        ["A", "tp=1", "model|kda", *cell, [[0.0, 10.0]],
         [16, 0, 1, 2]],
        ["A", "tp=1", "model|kda", *cell, [[0.0, 12.0]],
         [32, 16, 1, 2]],
    ]
    _provenance_file(_plan_file(tmp_path, "A", "H100-SXM", "routine")
                     ).write_text(json.dumps(provenance))
    group = _group("A", dims, 12.0, op="kda", node_spec_id="model|kda",
                   phase="prefill")
    rewrites, counts, _refused, moved, _unplanned = plan_restamp(
        [group], tmp_path, workers=1)
    assert dict(counts) == {"restamped": 3}
    assert rewrites[0]["access_bytes_per_rank"] == 10.0
    assert moved[0]["resolved_provenance"] is True


def test_an_ambiguous_cell_with_no_node_provenance_asks_its_point(tmp_path):
    """A physical cell may already have two prices from other deployments
    while the measured deployment no longer appears on a surface. Its own
    point plan is the second chance to name its node's provenance."""
    from tools.restamp_achieved import (SURFACES, _point_context,
                                        _points_file, _provenance_file,
                                        cell_of, plan_restamp)
    dims = {"B": 5, "d_model": 128, "s_query": 2}
    cell = cell_of("mhc_fused_post_pre", dims, _ENV, "decode")
    for surface in SURFACES:
        _plan(tmp_path, "A", "H100-SXM", surface,
              [[*cell, [[0.0, 1.0], [0.0, 2.0]]]]
              if surface == "routine" else [])
    group = _group("A", dims, 2.0, op="mhc_fused_post_pre",
                   parallel="tp=1", node_spec_id="model|entry")
    asked = [(dims["B"], dims["s_query"], _point_context(group, {1024}))]
    point = _points_file(tmp_path, "A", "H100-SXM", "tp=1", "decode", asked)
    point.parent.mkdir(parents=True)
    point.write_text(json.dumps([[*cell, [[0.0, 1.0]]]]))
    _provenance_file(point).write_text(json.dumps(
        [["A", "tp=1", "model|entry", *cell, [[0.0, 1.0]], [1024, 0, 2, 5]]]))
    rewrites, counts, _refused, moved, _unplanned = plan_restamp(
        [group], tmp_path, workers=1)
    assert dict(counts) == {"restamped": 3}
    assert rewrites[0]["node_spec_id"] == "model|entry"
    assert moved[0]["resolved_provenance"] is True


def test_a_numerator_that_puts_the_launch_above_the_card_is_not_stamped(tmp_path):
    # Today's law bills 2x what it did, and the row was at 0.9 of peak:
    # stamped, the launch would run at 1.8x the card. The write gate's rule.
    from tools.restamp_achieved import SURFACES, cell_of, plan_restamp
    dims = {"B": 772, "d_conv_channels": 8192}
    for surface in SURFACES:
        _plan(tmp_path, "A", "H100-SXM", surface,
              [[*cell_of("causal_conv1d", dims, _ENV, "decode"), [[0.0, 2.0e6]]]])
    (rewrites, counts, _refused, moved, _unplanned) = plan_restamp(
        [_group("A", dims, 1.0e6, op="causal_conv1d", n=4, mbu=0.9)],
        tmp_path, workers=1)
    assert rewrites == []
    assert dict(counts) == {"above_peak": 4}
    assert moved[0]["refused"] and abs(moved[0]["new_mbu"] - 1.8) < 1e-9


def test_own_points_reach_a_rows_batch_and_query_length():
    from backend.inference.parallel_config import ParallelConfig
    from tools.restamp_achieved import _point_context, own_points
    # A physical shape that names no context axis is context-free; s_kv pins
    # a known context, and an off-grid cache length asks for them all.
    def row(dims):
        return {"phase": "decode", "shape_json": json.dumps(dims)}

    assert _point_context(row({"B": 1, "s_query": 2}), {1024}) == 0
    assert _point_context(row({"B": 1, "s_query": 2, "s_kv": 1024}),
                          {1024}) == 1024
    assert _point_context(row({"B": 1, "s_query": 2, "s_kv": 1000}),
                          {1024}) == -1
    # Decode under data-parallel attention: the row's B is one rank's share,
    # so the request batch is B or 4B; s_query is the verify width.
    dp = ParallelConfig.from_dict({"dp_attention": 4, "ep_moe": 4})
    decode_calls = own_points("decode", dp, [(3, 2, 0)])
    assert len(decode_calls) == 1
    decode = decode_calls[0]
    assert {(p.batch, p.s_verify, p.s_prefix_cached) for p in decode} == {
        (3, 2, 0), (12, 2, 0)}
    assert len({p.seq_len for p in decode}) == 1
    every = own_points("decode", dp, [(3, 2, -1)])[0]
    assert len({p.seq_len for p in every}) > 1
    # The split form is the fallback after one context refuses.
    split = own_points("decode", dp, [(3, 2, -1)], split_contexts=True)
    assert len(split) > 1
    assert all(len({p.seq_len for p in call}) == 1 for call in split)
    # Prefill under context parallelism: the rank's 512 new tokens are the
    # request's 1024 — over nothing cached, or the rest of a longer context.
    cp = ParallelConfig.from_dict({"cp": 2, "ep_moe": 2})
    calls = own_points("prefill", cp, [(1, 512, 0)])
    prefill = [p for call in calls for p in call]
    assert all(p.seq_len - p.s_prefix_cached == 1024 for p in prefill)
    assert any(p.s_prefix_cached == 0 and p.seq_len == 1024 for p in prefill)
    assert any(p.s_prefix_cached > 0 for p in prefill)
    # One call per context: a refused context loses only its own points.
    assert all(len({p.seq_len for p in call}) == 1 for call in calls)


def test_a_row_off_todays_grid_is_priced_at_its_own_point(tmp_path):
    from tools.restamp_achieved import (SURFACES, _points_file,
                                        _provenance_file, cell_of,
                                        plan_restamp)
    dims = {"B": 618, "d_model": 2048, "n_tokens": 99, "s_query": 2}
    for surface in SURFACES:
        _plan(tmp_path, "A", "H100-SXM", surface, [])
    group = {**_group("A", dims, 38_000_000.0), "parallel": "ep_moe=16"}
    point = _points_file(tmp_path, "A", "H100-SXM", "ep_moe=16", "decode",
                         [(618, 2)])
    point.parent.mkdir(parents=True)
    point.write_text(json.dumps(
        [[*cell_of("permute", dims, _ENV, "decode"), [[0.0, 10_300_000.0]]]]))
    _provenance_file(point).write_text("[]")
    (rewrites, counts, _refused, _moved, _unplanned) = plan_restamp(
        [group], tmp_path, workers=1)
    assert dict(counts) == {"restamped": 3}
    assert rewrites[0]["access_bytes_per_rank"] == 10_300_000.0


def test_an_unplanned_key_says_which_part_of_the_identity_is_gone():
    """The stale report is a diagnosis, not a rename: an absent op names the
    nearest current op only as evidence, while a surviving op says whether
    the point, the routed geometry, another shape axis, or a dtype slot moved."""
    from tools.restamp_achieved import _stale_reasons, cell_of

    def cell(op, dims, env=None):
        return ("H100-SXM", cell_of(op, dims, env or _ENV, "decode"))

    rope_dims = {"B": 1, "d_head": 256, "d_rope": 64, "n_heads": 1,
                 "n_kv_heads": 1, "s_query": 64}
    priced = {
        cell("fused_qk_rmsnorm_rope_gate", rope_dims): {(1.0, 2.0)},
        cell("permute", {"B": 2, "d_model": 2048, "n_experts": 2,
                         "n_tokens": 2, "s_expert_rows": 1,
                         "s_query": 1}): {(0.0, 2.0)},
        cell("rmsnorm", {"B": 2, "d_model": 256, "s_query": 3}): {(1.0, 2.0)},
        cell("linear", {"B": 1, "d_in": 8, "d_out": 8, "s_query": 1},
             {"activation": "bf16", "output": "bf16", "weight": "fp8_e4m3"}):
            {(1.0, 2.0)},
    }
    groups = [
        _group("A", rope_dims, 1.0, op="fused_qk_rope_partial"),
        _group("A", {"B": 1, "d_model": 2048, "n_experts": 1,
                     "n_tokens": 1, "s_expert_rows": 1, "s_query": 1}, 1.0,
               op="permute"),
        _group("A", {"B": 1, "d_model": 256, "s_query": 2}, 1.0,
               op="rmsnorm"),
        _group("A", {"B": 1, "d_in": 8, "d_out": 8, "s_query": 1}, 1.0,
               op="linear"),
    ]
    groups = [{**g, "parallel": "tp=1"} for g in groups]
    reasons = {g["op"]: row for g, row in zip(groups, _stale_reasons(
        groups, priced), strict=True)}
    assert reasons["fused_qk_rope_partial"]["reason"] == "op-gone"
    assert (reasons["fused_qk_rope_partial"]["nearest_op"]
            == "fused_qk_rmsnorm_rope_gate")
    assert reasons["permute"]["reason"] == "routed-geometry-gone"
    assert set(reasons["permute"]["changed_axes"]) == {"B", "n_experts",
                                                       "n_tokens"}
    assert reasons["rmsnorm"]["reason"] == "point-gone"
    assert set(reasons["rmsnorm"]["changed_axes"]) == {"B", "s_query"}
    assert reasons["linear"]["reason"] == "dtype-gone"
