"""A rescaled row's NAMED axes are read at the batch, like its shapes.

Every per-op row carries its shape as a law in the batch, and
`batch_sweep.rows_at` re-reads `shape_in` / `shape_in_exact` /
`shape_out` / `shape_outs` there so one walk can answer every batch of a
sweep. It did not re-read `shape_in_dims` / `shape_out_dims`, which are the
same axes under their own names, and those are the only statement an axis's
extent has when it is not a whole number: a MoE cell whose per-expert share
is 8/7 of a row floors to 0 in `shape_in` and reads `8/7` here.

`kernel_bench.plan._starved_axes` refuses to build a kernel at such a cell
by reading exactly those two fields, so leaving them at the anchor's values
made that gate judge every non-anchor batch by the anchor's extents. The
plan walks once at `max(batch_sizes)`, so that was every batch but one — and
the cells it then let through exported launch shapes with 0-extent axes,
which no kernel can allocate and no measurement can ever match.

The assertion is the one that matters: the rows a sweep hands the planner at
a batch are the rows a walk at that batch would have handed it.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.analyzer import fetch_config
from backend.arch.loader import SpecLoader
from backend.inference.parallel_config import ParallelConfig
from kernel_bench.plan import build_jobs, build_jobs_over_batches

#: A sparse MoE whose dispatch share is a fraction of a row, a smaller MoE
#: at a different expert-parallel degree, and a model with a cadence axis.
CASES = (
    ("moonshotai/Kimi-K3", {"tp_attention": 8, "ep_moe": 8}),
    ("Qwen/Qwen3-30B-A3B", {"dp_attention": 8, "ep_moe": 8}),
    ("deepseek-ai/DeepSeek-V4-Flash", {"tp_attention": 8, "ep_moe": 8}),
)
BATCHES = (1, 8, 64)
SEQ = 4096


def _by_sid(jobs):
    return {j["node_spec_id"]: json.dumps(j["dims"], sort_keys=True)
            for j in jobs}


@pytest.mark.regression
class TestASweptBatchMatchesAWalkAtThatBatch(unittest.TestCase):
    def test_same_cells_same_dims_same_refusals(self):
        for model_id, parallel_dict in CASES:
            config = fetch_config(model_id)
            resolved = SpecLoader().match(config, model_id=model_id)
            self.assertIsNotNone(resolved, model_id)
            parallel = ParallelConfig.from_dict(parallel_dict)
            # One walk at max(BATCHES); every other batch is its rows re-read.
            swept = build_jobs_over_batches(avg_output_tokens=0,
                model_id=model_id, batch_sizes=list(BATCHES), seq_len=SEQ,
                phase="decode", parallel=parallel, resolved=resolved)
            for batch in BATCHES:
                with self.subTest(model=model_id, batch=batch):
                    swept_jobs, _swept_dropped = swept[batch]
                    own_jobs, _own_dropped = build_jobs(avg_output_tokens=0,
                        model_id=model_id, batch_size=batch, seq_len=SEQ,
                        phase="decode", parallel=parallel, resolved=resolved)
                    swept_dims, own_dims = _by_sid(swept_jobs), _by_sid(own_jobs)
                    # The gate drops the same cells either way.
                    # A swept batch may key FEWER cells than a walk, and
                    # only for one reason: a collective whose payload is not
                    # its flow (an all-gather widens an axis) has no law to
                    # be re-read from, so the rescale drops its payload
                    # columns and `CommLaunch.from_row` then keys nothing
                    # rather than keying the walked batch's buffer. It must
                    # never key a cell a walk does not, and the cells it
                    # does key must be identical.
                    extra = sorted(set(swept_dims) - set(own_dims))
                    self.assertEqual(
                        extra, [],
                        "a swept batch keyed a cell a walk at that batch "
                        "does not")
                    missing = sorted(set(own_dims) - set(swept_dims))
                    # Asked of the row's own flag — the walk's projection of
                    # `OpSpec.is_comm_op` — not of the node id's spelling.
                    collectives = {
                        str(row.get("aggregation_id") or "")
                        for row in _walk(resolved, parallel,
                                         batch).per_op_timing
                        if row.get("is_comm")}
                    self.assertTrue(
                        set(missing) <= collectives,
                        f"a swept batch lost non-collective cells: "
                        f"{sorted(set(missing) - collectives)}")
                    for sid, dims in swept_dims.items():
                        self.assertEqual(dims, own_dims[sid], sid)

    def test_no_exported_cell_has_a_zero_extent_axis(self):
        """What the stale gate let through. A launch shape with a 0-extent
        axis allocates an empty tensor or fails outright; either way the row
        is a permanent silent lookup miss, so no job may carry one."""
        for model_id, parallel_dict in CASES:
            config = fetch_config(model_id)
            resolved = SpecLoader().match(config, model_id=model_id)
            parallel = ParallelConfig.from_dict(parallel_dict)
            swept = build_jobs_over_batches(avg_output_tokens=0,
                model_id=model_id, batch_sizes=list(BATCHES), seq_len=SEQ,
                phase="decode", parallel=parallel, resolved=resolved)
            for batch in BATCHES:
                with self.subTest(model=model_id, batch=batch):
                    for job in swept[batch][0]:
                        flat = [v for v in _extents(job.get("shape_in"))]
                        self.assertNotIn(
                            0, flat,
                            f"{job['node_spec_id']} ({job['signature']}) "
                            f"exports shape_in {job.get('shape_in')}")


def _extents(shape):
    if isinstance(shape, (list, tuple)):
        for item in shape:
            if isinstance(item, (list, tuple)):
                yield from _extents(item)
            elif isinstance(item, int):
                yield item


@pytest.mark.regression
class TestTheNamedAxesAreReadAtTheBatch(unittest.TestCase):
    """The field-level assertion, which is where the bug is visible.

    The plan-level test above is currently satisfied on both sides of this
    fix, because a separate commit refuses every dispatch-zone cell before
    the starved-axis gate is reached, and those were the cells with
    fractional extents. That makes the stale-field bug LATENT, not fixed:
    the moment the MoE launch geometry lands and those cells are keyed
    again, a gate reading the anchor's extents goes blind again. So this
    asserts the invariant itself — a rescaled row's named axes are the
    named axes of a walk at that batch — which fails on the unfixed code
    for 1307 of Kimi-K3's 1323 rows.
    """

    def test_a_rescaled_row_names_the_batch_it_was_read_at(self):
        from backend.arch.eval.analysis import Point
        from backend.inference.batch_sweep import rows_at
        from backend.inference.comm_cost import Fabric
        for model_id, parallel_dict in CASES[:2]:
            config = fetch_config(model_id)
            resolved = SpecLoader().match(config, model_id=model_id)
            parallel = ParallelConfig.from_dict(parallel_dict)
            anchor = _walk(resolved, parallel, max(BATCHES))
            for batch in BATCHES:
                with self.subTest(model=model_id, batch=batch):
                    rescaled = rows_at(
                        anchor.per_op_timing, Point({"B": batch}),
                        cell=anchor, peak_flops=None,
                        peak_bw=None, fabric=Fabric(None, None, None),
                        breakdown=True)
                    walked = {str(r.get("aggregation_id") or ""): r
                              for r in _walk(resolved, parallel,
                                             batch).per_op_timing}
                    checked = 0
                    for row in rescaled:
                        sid = str(row.get("aggregation_id") or "")
                        own = walked.get(sid)
                        if own is None or not row.get("shape_at_batch"):
                            continue
                        checked += 1
                        for field in ("shape_in_dims", "shape_out_dims"):
                            self.assertEqual(
                                _named(row, field), _named(own, field),
                                f"{sid} [{field}] at batch {batch}")
                    self.assertTrue(checked, "no row joined the two walks")


#: The rescale's own marker — a walked row does not carry it, because a walk
#: did not read anything at another batch.
_RESCALE_MARKER = ("shape_at_batch",)

#: The one field this invariant does NOT yet hold for, named here so it is a
#: declared residue and not a silence.
#:
#: `node_dims` is the leaf's own dim table. Its entries that are AXES of the
#: leaf's shape are re-read from the refreshed named axes, which is exact.
#: Its entries that are not — the step's `B` on a `parameter_tensor`, whose
#: tensor is the same at every batch, and the token count `M` — have no law
#: to be read from, and the table cannot simply be dropped either: it is
#: where `RowIdentity` finds the spec constants a tensor shape cannot carry,
#: and dropping it changes the measurement key (the keys assertion in the
#: class above fails).
#:
#: It cannot be patched here, because `B` in this table does not always mean
#: the batch: inside a MoE dispatch zone it is one expert's SHARE of it
#: (`1/7` of a row), so assigning the batch to it would overwrite a share
#: with a whole. The fix belongs where the fact is made — the walk should
#: state these dims as laws, the way it states the shape — and until it does,
#: this test asserts that the stale entries are exactly the ones no axis of
#: the row can state.
_NO_LAW_YET = "node_dims"


@pytest.mark.regression
class TestARescaledRowIsEqualOrSilent(unittest.TestCase):
    maxDiff = None

    """The whole-row invariant, over EVERY field rather than the ones we
    happened to notice.

    Three fields were found stale in two days by three different readers:
    the named dims (this module), the collective payload columns (the
    comm-signature work), and the row's own dim table. Each was the same
    defect — the shape rescale moved the shape columns and the
    numerators and left a batch-dependent field at the walked batch — and
    each was found by someone tripping over it rather than by a gate.

    So the gate is the invariant, not a list: for every field on a rescaled
    row, either it equals a walk at that batch, or **the row does not carry
    it**. A field that cannot be read at another batch by substituting into
    a law is dropped, which is what the phase rescale already does with the
    formula popovers. Silently keeping the anchor's value is the one
    forbidden option, and it is what every instance of this bug was.
    """

    def test_no_field_silently_keeps_the_anchors_value(self):
        from backend.arch.eval.analysis import Point
        from backend.inference.batch_sweep import rows_at
        from backend.inference.comm_cost import Fabric
        from backend.inference.module_rows import served_rows
        for model_id, parallel_dict in CASES[:2]:
            config = fetch_config(model_id)
            resolved = SpecLoader().match(config, model_id=model_id)
            parallel = ParallelConfig.from_dict(parallel_dict)
            anchor = _walk(resolved, parallel, max(BATCHES))
            for batch in BATCHES:
                with self.subTest(model=model_id, batch=batch):
                    # The rows as a reader is served them: the laws a row
                    # carries for a reader at another point are expressions
                    # the walk keeps in process (`served_rows`), and a law
                    # is the same at every batch by construction — it is
                    # what a batch is read out of.
                    rescaled = served_rows(rows_at(
                        anchor.per_op_timing, Point({"B": batch}),
                        cell=anchor, peak_flops=None,
                        peak_bw=None, fabric=Fabric(None, None, None),
                        breakdown=True))
                    walked = {str(r.get("aggregation_id") or ""): r
                              for r in served_rows(
                                  _walk(resolved, parallel,
                                        batch).per_op_timing)}
                    anchors = {str(r.get("aggregation_id") or ""): r
                               for r in served_rows(anchor.per_op_timing)}
                    stale = []
                    checked = 0
                    for row in rescaled:
                        sid = str(row.get("aggregation_id") or "")
                        own, was = walked.get(sid), anchors.get(sid)
                        if own is None or was is None:
                            continue
                        checked += 1
                        for field in row:
                            if field in _RESCALE_MARKER:
                                continue
                            if row.get(field) is None:
                                continue        # silent, which is allowed
                            here, there = _named(row, field), _named(own, field)
                            if here == there:
                                continue
                            # The field moved between the anchor and this
                            # batch, and the rescaled row still shows the
                            # anchor's answer: that is the staleness. A
                            # field where the LAW and the walk disagree at
                            # every batch alike is a different defect and
                            # not this gate's.
                            if field == _NO_LAW_YET:
                                stale += _unstatable(sid, row, own, was)
                                continue
                            if here == _named(was, field):
                                stale.append(f"{sid} [{field}]")
                    self.assertTrue(checked, "no row joined the two walks")
                    self.assertEqual(
                        stale[:12], [],
                        f"{len(stale)} field(s) on a rescaled row hold the "
                        f"anchor batch's value. Refresh them from the row's "
                        f"law, or drop them from the rescaled row — do not "
                        f"leave a reader an answer about another batch.")


def test_a_multi_output_rows_exact_ports_are_read_at_the_batch():
    """A node with no single output — a fused gate_up writing its gate and up
    halves — states its exact extents and named axes PER PORT, aligned with
    `shape_outs`, so a routed port's mean rows are not floored away
    (`phases`). The rescale reads them at the batch from the same ports,
    and they are what a walk at that batch states."""
    from backend.arch.eval.analysis import Point
    from backend.inference.batch_sweep import rows_at
    from backend.inference.comm_cost import Fabric
    model_id = "Qwen/Qwen3-30B-A3B"
    resolved = SpecLoader().match(fetch_config(model_id), model_id=model_id)
    parallel = ParallelConfig.from_dict({"tp_attention": 4, "ep_moe": 4})
    anchor = _walk(resolved, parallel, 64)
    rescaled = rows_at(anchor.per_op_timing, Point({"B": 8}), cell=anchor,
                       peak_flops=None, peak_bw=None,
                       fabric=Fabric(None, None, None), breakdown=True)
    walked = {str(r.get("aggregation_id") or ""): r
              for r in _walk(resolved, parallel, 8).per_op_timing}
    ported = [r for r in rescaled
              if r.get("shape_out") is None and r.get("shape_outs")
              and str(r.get("aggregation_id") or "") in walked]
    assert ported, "no multi-output row joined the two walks"
    for row in ported:
        own = walked[str(row["aggregation_id"])]
        assert len(own["shape_out_exact"]) == len(own["shape_outs"])
        for field in ("shape_out_exact", "shape_out_dims"):
            assert _named(row, field) == _named(own, field), (
                row["aggregation_id"], field)
    # 8 tokens x top-8 / 128 experts: half a row per expert, stated exactly.
    assert any("1/2" in _named(row, "shape_out_exact") for row in ported)


def _walk(resolved, parallel, batch):
    from backend.inference.framework_bindings import DEFAULT_FRAMEWORK, get_binding
    from backend.inference.pipeline import analyze
    return analyze(
        resolved, parallel=parallel,
        framework_binding=get_binding(DEFAULT_FRAMEWORK),
        batch_size=batch,
        avg_input_tokens=SEQ, avg_output_tokens=0,
        phase="decode", include_aux_timing=True, breakdown=False)


def _named(row, field):
    return json.dumps(row.get(field), sort_keys=True, default=str)


def _unstatable(sid, row, own, was):
    """`node_dims` entries that are stale AND are axes of this row's shape.

    An entry an axis states must be re-read (that is exact); an entry no
    axis states is the declared residue above. So this reports only the
    first kind, and an empty list is the invariant holding as far as it can.
    """
    here, there = row.get(_NO_LAW_YET) or {}, own.get(_NO_LAW_YET) or {}
    anchor_dims = was.get(_NO_LAW_YET) or {}
    statable = {axis.get("expr") for axis in
                _axes(row.get("shape_in_dims")) + _axes(row.get("shape_out_dims"))}
    return [f"{sid} [node_dims.{name}]" for name in sorted(set(here) | set(there))
            if here.get(name) != there.get(name)
            and here.get(name) == anchor_dims.get(name)
            and name in statable]


def _axes(named):
    if isinstance(named, dict):
        return [named]
    if isinstance(named, (list, tuple)):
        return [a for item in named for a in _axes(item)]
    return []


if __name__ == "__main__":
    unittest.main()
