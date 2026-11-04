"""Row scatters under a cut sequence: v's rows co-shard with x's sequence.

A vision tower's output `v: [B, N, D]` is written into the text stream
`x: [B, S, D]` at its placeholder rows. The tower is not in the text model's
context-parallel world, so `v` arrives whole while `x` arrives cut along S.
Rank r keeps the placeholder rows that fall in its own sequence block; with no
mask in the IR that is the uniform-density co-shard projection — `v`'s N axis
cut in the same world as a padded Block, `N_rank = ceil(N / W)` — reached by a
local slice. The numbers pinned here are what that projection promises: local
rows, local add FLOPs and local `v` bytes per rank, conservation of the global
N through the valid counts, and no communication for the slice.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

SPEC = """\
match: {{ model_type: [{name}], model_id: [{name}] }}
dims:
  d_model: config.hidden_size
  n_layers: "1"
  n_spatial_merge: "2"
blocks:
  blk:
    graph:
      n: {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      o: {{ op: add, in: [x, n], dim: d_model }}
    outputs: [o]
model_inputs:
  input_ids: {{ kind: tokens }}
  pixel_values: {{ kind: image, axes: [n_img, img_h, img_w] }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: d_model, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - name: vis
    group: true
    aux_kind: vision
    layers:
      - {{ name: vis.patchify, op: patchify, inputs: [pixel_values],
          dims: {{ patch: "1", temporal: "1", channels: d_model, align: "1" }},
          segment: item, source_anchor: "a test processor" }}
      - {{ name: vis.patch, op: linear, inputs: [vis.patchify], dims: {{ in: d_model, out: d_model }},
          weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      - {{ name: vis.fold, op: spatial_merge, dims: {{ n_spatial_merge: n_spatial_merge }},
          inputs: [vis.patch] }}
      - {{ name: vis.proj, op: linear, dims: {{ in: "d_model * n_spatial_merge * n_spatial_merge", out: d_model }},
          weight_axes: [{{ replicated: true }}, {{ replicated: true }}], inputs: [vis.fold] }}
  - {{ name: merge, op: {op}, inputs: [embed, vis.proj],
      prompt_slots: {{ domain: pixel_values, defines: true, source_anchor: "a test merge" }} }}
  - {{ name: layers, repeat: n_layers, streams: {{ hidden: {{ carries: true }} }}, template: blk, inputs: [merge] }}
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: d_model }}, inputs: [layers],
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""

D_MODEL = 8
#: The toy model's own activation width: it states no dtype, so BF16.
ACT_BYTES = 2


def _resolved(name: str, op: str):
    from backend.arch.loader import SpecLoader
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / f"{name}.yaml").write_text(
            SPEC.format(name=name, op=op))
        return SpecLoader(base_dir=str(base)).match(
            {"model_type": name, "torch_dtype": "bfloat16", "hidden_size": D_MODEL}, model_id=name)


# Image sides whose 1x1 patches fold 2x2 into `n_rows` rows.
_IMAGE_SIDES = {4: (4, 4), 5: (4, 5)}


def _run(op: str, cp: int, seq: int, n_rows: int):
    """The prefill walk of a prompt of `seq` positions whose merge is handed
    `n_rows` rows: one image's, as its encode walk hands them over.

    A language-model step is handed no media rows (the encoders are their
    own accounting), so the rows are put on the edge here: what is under
    test is the layout of a row scatter, whatever its row count."""
    from unittest import mock

    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    from backend.inference.parallel_config import ParallelConfig
    height, width = _IMAGE_SIDES[n_rows]
    program = Program.of(_resolved(f"scatter_{op}_{cp}_{seq}", op))
    image = {"pixel_values": {"n_img": 1, "img_h": height, "img_w": width}}
    deployment = Deployment(program, config=ParallelConfig(cp=cp)).at(
        batch=1, input_tokens=seq, media=image)
    assert deployment.media_tokens() == {"pixel_values": n_rows}
    handed = deployment.handed_over()
    with mock.patch.object(Deployment, "handed_over",
                           lambda self, media=None: handed):
        return deployment.analyze()


def _merge(analysis):
    """The merge, with its per-rank FLOPs and access bytes."""
    from backend.arch.layout.physical_cost import (physical_access,
                                                   physical_cost_plan)
    node = next(n for n in analysis.program.nodes() if n.name == "merge")
    flops = physical_cost_plan(analysis).cost.values[node.uid].flops
    access = physical_access(analysis, scale_dtype_bytes=4.0).values[node.uid]
    return node, {"flops_per_rank": int(flops),
                  "act_in_bytes_per_rank": access.act_in_bytes_per_rank,
                  "act_out_bytes_per_rank": access.act_out_bytes_per_rank}


@pytest.mark.parametrize("op", ["scatter_rows_replace", "scatter_rows_add"])
@pytest.mark.parametrize("seq,n_rows", [(16, 4), (20, 5)])
def test_rows_co_shard_with_the_sequence_under_cp2(op, seq, n_rows):
    from backend.arch.layout.layout import Block, axis_fully_replicated
    from backend.arch.layout.physical import physical_shapes

    analysis = _run(op, 2, seq, n_rows)
    node, row = _merge(analysis)
    _, row1 = _merge(_run(op, 1, seq, n_rows))

    # Globally v is [1, N, D]: the tower folds 2x2 patches of the image.
    assert [int(e) for e in analysis.shapes.values[node.uid].ins[1]] == \
        [1, n_rows, D_MODEL]
    # The required state of v: N cut over cp as a padded Block.
    rows = analysis.input_states[(node.uid, 1)].layout[1]
    assert isinstance(rows, Block) and rows.group.id == "cp"
    assert rows.pad_to == 1
    # Physical capacity per rank is ceil(N / 2); the valid rows over the
    # parts add back to the global N.
    n_rank = -(-n_rows // 2)
    local = physical_shapes(analysis).values[node.uid]
    assert [int(e) for e in local.ins[1]] == [1, n_rank, D_MODEL]
    assert [int(e) for e in local.ins[0]] == [1, seq // 2, D_MODEL]
    assert sum(rows.valid_counts_per_shard(n_rows)) == n_rows

    # FLOPs: replace is free; add is B * N_rank * D per rank.
    expected_flops = (0 if op == "scatter_rows_replace"
                      else 1 * n_rank * D_MODEL)
    assert row["flops_per_rank"] == expected_flops
    assert row1["flops_per_rank"] == (
        0 if op == "scatter_rows_replace" else n_rows * D_MODEL)
    # Bytes: read x, read v, write y at the local shapes.
    x_bytes = seq // 2 * D_MODEL * ACT_BYTES
    v_bytes = n_rank * D_MODEL * ACT_BYTES
    assert row["act_in_bytes_per_rank"] == x_bytes + v_bytes
    assert row["act_out_bytes_per_rank"] == x_bytes
    assert row1["act_in_bytes_per_rank"] == (seq + n_rows) * D_MODEL * ACT_BYTES
    if n_rows % 2 == 0:
        assert 2 * v_bytes == n_rows * D_MODEL * ACT_BYTES
        if op == "scatter_rows_add":
            assert 2 * row["flops_per_rank"] == row1["flops_per_rank"]

    # The tower's output is handed over whole (an encoder's rows are not in
    # the text model's sequence worlds); it reaches its co-shard by a local
    # slice: the transition prices no collective, and the walk records no
    # communication event on either of the merge's edges.
    arrived = analysis.rule.model_input_state(
        "vis.proj", analysis.shapes.values[node.uid].ins[1], encoder=True)
    assert axis_fully_replicated(arrived.layout[1])
    assert arrived != analysis.input_states[(node.uid, 1)]
    assert analysis.rule.lower_diff(
        arrived, analysis.input_states[(node.uid, 1)],
        node=node, input_index=1) == ()
    assert not [r for r in analysis.redistributes if node.uid == r.node]


@pytest.mark.parametrize("op", ["scatter_rows_replace", "scatter_rows_add"])
def test_a_whole_sequence_keeps_the_rows_whole(op):
    from backend.arch.layout.layout import axis_fully_replicated
    analysis = _run(op, 1, 16, 4)
    node, _ = _merge(analysis)
    for (uid, _index), state in analysis.input_states.items():
        if uid == node.uid:
            assert all(axis_fully_replicated(a) for a in state.layout)

