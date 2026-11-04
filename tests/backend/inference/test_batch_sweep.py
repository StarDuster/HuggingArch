"""`batch_sweep` — a phase and its rows re-evaluated at another batch.

Falsification targets:
  - at the anchor batch the rescale is the identity — the phase the
    estimate computed is what a rescale to that batch gives, or every
    other batch is read from the wrong origin;
  - a row's part fixed per step does not grow with the batch, and its
    part per request does;
  - a collective is re-priced at the batch on the card's fabric, not
    carried at the anchor's time;
  - a measured row keeps its value at every B (CSV t_op is canonical at
    B=1 and does not rescale), while a theoretical row's t_op moves.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.sweep_parity import rescale_phase, rescale_rows, response_cell
from backend.arch.eval.transforms import Workload
from backend.inference.comm_cost import Fabric
from backend.inference.row_roofline import (
    LawBook, RowRoofline, RowTimes, build_row_exprs, comm_factor, law_columns,
    row_law)

_PEAK_FLOPS = 1.0e15
_PEAK_BW = 1.0e12
_FABRIC = Fabric(8, 1.0e11, 1.0e10)
#: The workload the rows were walked at, but for the batch.
_POINT = {"B": 4, "s_input": 8, "s_output": 0, "s_prefix": 0, "s_verify": 1,
          "rho": 1}
B = Workload.symbols()["batch"]
#: The batch's routing skew, which a routed row's law reads.
RHO = Workload.symbols()["ep_load_factor"]
#: The synthetic rows' laws.
_BOOK = LawBook()
#: The walk the synthetic rows are read off (`batch_sweep.rows_at`'s `cell`).
_CELL = response_cell(_BOOK)


def _row(*, flops_fixed=0, flops_per_request=200, act_fixed=0,
         act_per_request=4, act_out_per_request=1, read_per_row=100,
         weight_total=100, metadata_total=10, comm_per_request=0.0,
         routed=False, sq=8, is_comm=False, comm_op="", comm_world=1, **kw):
    """A row whose law is stated from these components at `sq` tokens per
    request — what `phases` builds from the walk's extents. `act_*` is its
    whole activation traffic (reads and writes), `act_out_*` the writes of
    it; `sq` is the rows one request runs on. The row is one instance,
    walked at the anchor batch `_at` uses. A `routed` row's per-request
    parts are the busiest rank's, which reads the batch's skew `rho`."""
    per = RHO if routed else 1
    row = {
        "op": "linear",
        "is_comm": is_comm,
        "comm_op": comm_op,
        "comm_world": comm_world,
        "comm_base_stride": 1,
        "count_per_step": 1,
        "act_dtype": None,
        "t_comm_s": 0.0,
        "t_measured_s": None,
        "source": "theoretical_naive",
        **law_columns(build_row_exprs(
            count=1, flops=flops_fixed + B * per * flops_per_request,
            act_in=act_fixed + B * per * (act_per_request
                                          - act_out_per_request),
            act_out=B * per * act_out_per_request,
            demand=B * per * sq * read_per_row,
            weight_total=weight_total,
            metadata_total=metadata_total,
            comm_payload=B * per * comm_per_request,
            comm_factor=comm_factor(comm_op, comm_world) if is_comm else 0.0,
            comm_priced=is_comm and comm_world > 1), _BOOK),
        # Walked at the anchor batch; no shape, dims or display facts.
        "shape_law": None,
        "point_law": {"point": {**_POINT, "B": 4}, "ports": [], "wire": None, "dims": None,
                      "definitions": None,
                      "flops": {"law": sp.Integer(0), "one": sp.Integer(0), "record": None,
                                "identity": None},
                      "access": None, "in_terms": None, "out_terms": None,
                      "access_in": None, "comm": None},
    }
    row.update(kw)
    return row


def _comm_row(routed=False):
    return _row(op="all_reduce", is_comm=True, comm_op="all_reduce",
                comm_world=4, read_per_row=0, weight_total=0,
                metadata_total=0, flops_per_request=0, act_per_request=0,
                act_out_per_request=0, comm_per_request=1000.0,
                routed=routed)


def _phase(batch, sq=8, phase="decode"):
    return {
        "phase": phase,
        "batch": batch,
        "seq_effective": sq,
        "note_detail": ".",
        "binding": {"s_query": sq, "tokens_per_step": 1, "s_verify": 1,
                    "acceptance_rate": 1.0},
        "flops_by_module": [{"name": "mlp", "flops_per_rank": 400,
                             "flops_full": 400}],
    }


def _at(rows, batch, anchor=4, sq=8):
    return rescale_phase(_phase(anchor, sq), rows, {'B': batch}, _PEAK_FLOPS,
                          _PEAK_BW, _FABRIC, cell=_CELL, breakdown=True)


class TestAnchorIdentity(unittest.TestCase):
    def test_anchor_batch_reproduces_the_phase_math(self):
        # One row, non-MoE (read_per_row == total → saturation caps at
        # total): at B=4, Sq=8 the sums are hand-computable.
        dec = _at([_row()], 4)
        # flops: 4 × 200; access: min(32×100, 100) + 10 + 4×4 = 126
        self.assertEqual(dec["flops_per_step"], 800)
        self.assertEqual(dec["total_bytes"], 126)
        self.assertEqual(dec["mem_read_bytes"], 126 - 4)
        self.assertEqual(dec["mem_write_bytes"], 4)
        self.assertAlmostEqual(dec["t_compute_s"], 800 / _PEAK_FLOPS)
        self.assertAlmostEqual(dec["t_memory_s"], 126 / _PEAK_BW)
        self.assertEqual(dec["t_sol_s"],
                         max(800 / _PEAK_FLOPS, 126 / _PEAK_BW, 0))
        # flops_by_module scales with B / phase.batch
        self.assertEqual(dec["flops_by_module"][0]["flops_per_rank"], 400)

    def test_linear_scaling_between_batches(self):
        self.assertEqual(_at([_row()], 2)["flops_per_step"],
                         2 * _at([_row()], 1)["flops_per_step"])

    def test_the_fixed_part_does_not_grow(self):
        row = _row(flops_fixed=1000, act_fixed=50)
        one, five = _at([row], 1), _at([row], 5)
        self.assertEqual(five["flops_per_step"] - one["flops_per_step"], 4 * 200)
        self.assertEqual(five["total_bytes"] - one["total_bytes"], 4 * 4)


class TestCollectiveAtTheBatch(unittest.TestCase):
    def test_payload_is_priced_at_the_batch_on_the_fabric(self):
        rows = rescale_rows([_comm_row()], {'B': 3}, _PEAK_FLOPS, _PEAK_BW,
                             _FABRIC, _CELL, breakdown=True)
        # ring all_reduce over 4 ranks moves 2·(4-1)/4 of the payload,
        # 3 requests x 1000 bytes, over the intra-node fabric.
        self.assertAlmostEqual(rows[0]["t_comm_s"], 2 * 3 / 4 * 3000 / 1.0e11)
        dec = _at([_comm_row(), _row()], 3)
        self.assertAlmostEqual(dec["t_comm_s"], rows[0]["t_comm_s"])

    def test_the_busiest_rank_sends_its_own_rows(self):
        """A routed exchange's payload is the busiest rank's, a law in the
        batch's skew: read at `rho` 1.5 it prices 1.5 times the balanced
        point's, with nothing multiplied in beside the law."""
        row = _comm_row(routed=True)
        mean = rescale_rows([row], {'B': 3}, _PEAK_FLOPS, _PEAK_BW,
                             _FABRIC, _CELL, breakdown=True)[0]
        hot = rescale_rows([row], {'B': 3, 'rho': sp.Rational(3, 2)},
                            _PEAK_FLOPS, _PEAK_BW, _FABRIC, _CELL,
                            breakdown=True)[0]
        self.assertAlmostEqual(hot["t_comm_s"], 1.5 * mean["t_comm_s"])


class TestMoeSaturation(unittest.TestCase):
    def test_weight_read_caps_at_total(self):
        # read_per_row = 1 → at B=4, Sq=8 the row-driven read is 32,
        # below the 100 total, so the cap does not fire yet.
        row = _row(read_per_row=1)
        dec = _at([row], 4)
        # access: min(32×1, 100) + 10×32/100 + 16 = 51
        # (scales follow the routed weight fraction)
        self.assertEqual(dec["total_bytes"], 51)


class TestByBound(unittest.TestCase):
    def test_each_row_counts_once_under_the_term_that_set_it(self):
        dec = _at([_comm_row(), _row()], 4)
        self.assertAlmostEqual(sum(dec["t_sol_by_bound"].values()),
                               dec["t_sol_s"])
        self.assertGreater(dec["t_sol_by_bound"]["comm"], 0)
        self.assertGreater(dec["t_sol_by_bound"]["memory"], 0)
        shares = dec["t_sol_by_bound_share"]
        self.assertAlmostEqual(sum(shares.values()), 1.0)
        for bound, t in dec["t_sol_by_bound"].items():
            self.assertAlmostEqual(shares[bound] * dec["t_sol_s"], t)

    def test_a_row_names_the_term_that_set_it_and_its_intensity(self):
        rows = rescale_rows([_comm_row(), _row(), _row(flops_per_request=10**6,
                                                    act_per_request=0,
                                                    weight_total=1,
                                                    read_per_row=1,
                                                    metadata_total=0)],
                             {'B': 4}, _PEAK_FLOPS, _PEAK_BW, _FABRIC, _CELL, breakdown=True)
        bounds = ("compute", "memory", "comm", "launch")
        for r in rows:
            terms = (r["t_compute_s"], r["t_memory_s"], r["t_comm_s"],
                     r["t_launch_s"])
            self.assertEqual(r["bound"], bounds[terms.index(max(terms))])
            if r["access_bytes_per_rank"]:
                self.assertEqual(r["arithmetic_intensity"],
                                 r["flops_per_rank"] / r["access_bytes_per_rank"])
            else:
                self.assertIsNone(r["arithmetic_intensity"])
        self.assertEqual([r["bound"] for r in rows],
                         ["comm", "memory", "compute"])

    def test_a_tie_goes_compute_then_memory_then_comm_then_launch(self):
        t = 1e-6
        self.assertEqual(RowTimes(t, t, t, t).bound, "compute")
        self.assertEqual(RowTimes(t, t, 0.0, t).bound, "compute")
        self.assertEqual(RowTimes(t, 0.0, t, t).bound, "compute")
        self.assertEqual(RowTimes(0.0, t, t, t).bound, "memory")
        self.assertEqual(RowTimes(0.0, 0.0, t, t).bound, "comm")
        self.assertEqual(RowTimes(0.0, 0.0, t, t, t).bound, "comm")
        self.assertEqual(RowTimes(0.0, 0.0, 0.0, t, t).bound, "launch")
        # A tied row counts once, under the first of its tied terms.
        tied = {"t_compute_s": 0.0, "t_memory_s": t, "t_comm_s": t,
                "t_launch_s": t, "t_op_s": t}
        self.assertEqual(RowRoofline.by_bound([tied]),
                         {"compute": 0.0, "memory": t, "comm": 0.0,
                          "launch": 0.0})
        # A row that takes no time has no winner.
        self.assertIsNone(RowRoofline.labels(
            RowTimes(0.0, 0.0, 0.0, 0.0), flops=0, access=0)["bound"])

    def test_a_measured_row_keeps_its_theoretical_bound(self):
        theory = rescale_rows([_row()], {'B': 4}, _PEAK_FLOPS, _PEAK_BW, _FABRIC, _CELL, breakdown=True)[0]
        measured = rescale_rows([_row(t_measured_s=1.0, source="measured")],
                                 {'B': 4}, _PEAK_FLOPS, _PEAK_BW, _FABRIC, _CELL, breakdown=True)[0]
        self.assertEqual(measured["bound"], theory["bound"])
        self.assertEqual(measured["arithmetic_intensity"],
                         theory["arithmetic_intensity"])


class TestMeasuredRows(unittest.TestCase):
    def test_a_measurement_does_not_survive_another_batch(self):
        """A measurement is of one key; at another batch the row's key is
        another, so the rescaled row has no measured time until its own key
        is looked up (`apply_measured`)."""
        row = _row(t_measured_s=1.25e-6, source="measured", measured_state="measured")
        for b in (1, 2, 16):          # walked at 4
            rescaled = rescale_rows([row], {'B': b}, _PEAK_FLOPS, _PEAK_BW, _FABRIC, _CELL, breakdown=True)
            self.assertIsNone(rescaled[0]["t_measured_s"], f"B={b}")

    def test_a_rescaled_phase_reads_no_measurement_from_its_rows(self):
        """The phase at a point does not add its rows' measured times up:
        the measured total is the measured fold's, over the point's own keys
        (`apply_measured`), so before it the phase states no measured row."""
        measured = _row(t_measured_s=1.25e-6, source="measured", measured_state="measured")
        dec = _at([measured, {**measured, "t_measured_s": 2.5e-6}], 4)
        self.assertIsNone(dec["t_measured_s"])
        self.assertEqual(dec["coverage"], {"measured": 0, "total": 2})

    def test_unmeasured_step_has_no_measured_time(self):
        """Nothing measured → no measured time, not the SOL time renamed."""
        row = rescale_rows([_row()], {'B': 4}, _PEAK_FLOPS, _PEAK_BW,
                            _FABRIC, _CELL, breakdown=True)[0]
        self.assertIsNone(row["t_measured_s"])
        dec = _at([_row()], 4)
        self.assertIsNone(dec["t_measured_s"])
        self.assertEqual((dec["measured_rows"], dec["total_rows"]), (0, 1))

    def test_a_pipeline_step_is_every_stage_at_the_batch(self):
        """pp>1: the step at another batch is every stage's row laws read
        there, summed — the bound stage's rows are one of them — and no
        measured step is published before the fold: its terms are every
        stage's rows."""
        stage0 = [_row(), _comm_row()]
        stage1 = [_row(flops_per_request=9e6)]
        phase = {**_phase(4, 8, "decode"),
                 "pipeline_time": {
                     "stages": 2, "stage": 1,
                     "stage_laws": [[row_law(r) for r in stage]
                                    for stage in (stage0, stage1)]}}
        # The bound stage's row at the point (batch 2), measured there.
        measured = {**stage1[0], "t_measured_s": 1.25e-6,
                    "source": "measured", "measured_state": "measured",
                    "point_law": {**stage1[0]["point_law"],
                                  "point": {**_POINT, "B": 2}}}
        out = rescale_phase(phase, [measured], {'B': 2}, _PEAK_FLOPS, _PEAK_BW,
                             _FABRIC, cell=_CELL, breakdown=True)
        each = [sum(r["t_op_s"] for r in rescale_rows(
                    stage, {'B': 2}, _PEAK_FLOPS, _PEAK_BW, _FABRIC, _CELL, breakdown=True))
                for stage in (stage0, stage1)]
        self.assertEqual(out["pipeline_time"]["per_stage"], each)
        self.assertAlmostEqual(out["t_sol_s"], sum(each))
        self.assertAlmostEqual(sum(out["t_sol_by_bound"].values()),
                               out["t_sol_s"])
        self.assertGreater(out["t_comm_s"], 0)
        self.assertIsNone(out["t_measured_s"])
        self.assertEqual(out["coverage"], {"measured": 0, "total": 3})


if __name__ == "__main__":
    unittest.main()


class TestASweptPhaseStatesItsOwnCommBytes(unittest.TestCase):
    """The phase path's collective bytes belong to the phase's batch.

    Three consecutive review rounds caught the same class of defect — a
    rescaled row carrying the anchor batch's value for a field the rescale
    did not re-read — and each time it was found by measuring, not by a
    test. `comm_bytes_per_rank` was the third: `rows_at`
    re-read it and the phase rescale did not, so the scaling panel showed the
    anchor's communication bytes while the planner saw the batch's. The
    refresh moved into `rows_at`, which both callers take. This pins
    it there.

    Two things are asserted, because they are the two halves of the rule:
    a collective whose payload the walk states a law for carries that
    batch's per-rank WIRE VOLUME (the ring factor over the payload — a
    different quantity from the law's payload, which is what made the
    earlier diagnosis wrong), and a collective whose payload has no law
    carries nothing at all rather than the anchor's number or a zero.
    """

    def test_the_volume_is_this_batch_s_and_equals_a_walk_there(self):
        """Every collective of a TP2 decode, walked at B=1 and read at B=8
        through `rows_at` (the path the phase rescale takes), carries
        the per-rank wire volume a walk at B=8 states — not the anchor's.

        The rows state that volume as a law (`row_roofline`: ring factor x
        the payload's law x count), so it is read at every batch, an
        all-gather's included; main's earlier fix refreshed it from the
        shape law and dropped it where the shape law said nothing."""
        from backend.inference.parallel_config import ParallelConfig
        from tests.sweep_parity import Deployment
        dep = Deployment("meta-llama/Llama-3.2-3B", parallel=ParallelConfig(
            tp_attention=2, tp_dense_ffn=2, tp_moe_expert=2))
        anchor = dep.direct(avg_input_tokens=512, prefill_batch=1, decode_batch=1)
        walked = dep.direct(avg_input_tokens=512, prefill_batch=8, decode_batch=8)
        read = rescale_rows(anchor["per_op_timing_decode"], {'B': 8},
                             _PEAK_FLOPS, _PEAK_BW, _FABRIC, _CELL, breakdown=True)
        there = {r["uid"]: r for r in walked["per_op_timing_decode"]}
        comm = [r for r in read if r.get("is_comm")]
        self.assertTrue(comm, "a TP2 decode has collectives")
        moved = 0
        for row in comm:
            self.assertIsNotNone(row["comm_bytes_per_rank"], row["aggregation_id"])
            self.assertAlmostEqual(row["comm_bytes_per_rank"],
                                   there[row["uid"]]["comm_bytes_per_rank"],
                                   msg=row["aggregation_id"])
            before = next(a for a in anchor["per_op_timing_decode"]
                          if a["uid"] == row["uid"])
            moved += row["comm_bytes_per_rank"] != before["comm_bytes_per_rank"]
        self.assertTrue(moved, "no collective's volume moved with the batch")

    def test_an_absent_volume_makes_the_rollup_absent(self):
        """And the absence propagates: a subtree total that silently omits
        one of its children is worse than one that says it cannot be
        computed."""
        from backend.inference.phase_formulas import _rollup_column
        rows = [
            {"depth": 0, "comm_bytes_per_rank": 4.0},
            {"depth": 1, "comm_bytes_per_rank": None},
            {"depth": 0, "comm_bytes_per_rank": 8.0},
        ]
        _rollup_column(rows, "comm_bytes_per_rank", "comm_bytes_agg")
        self.assertIsNone(rows[0]["comm_bytes_agg"],
                          "a parent of an unpublished child publishes no total")
        self.assertIsNone(rows[1]["comm_bytes_agg"])
        self.assertEqual(rows[2]["comm_bytes_agg"], 8.0,
                         "an unrelated subtree still totals")

    def test_an_absent_volume_makes_the_table_total_absent(self):
        """The per-op table's Total row (`op_table_total`) follows the same
        rule, for the fold and for a flat row list alike; the other columns
        still total."""
        from backend.inference.phase_formulas import table_total
        fold = [
            {"depth": 0},                                    # untimed parent
            {"depth": 1, "t_op_s": 1.0, "comm_bytes_per_rank": 4.0},
            {"depth": 1, "t_op_s": 2.0, "comm_bytes_per_rank": None},
        ]
        flat = [{"t_op_s": 1.0, "comm_bytes_per_rank": 4.0},
                {"t_op_s": 2.0}]                             # field dropped
        for rows in (fold, flat):
            total = table_total(rows)
            self.assertIsNone(total["comm_bytes_per_rank"])
            self.assertEqual(total["t_sol_s"], 3.0)
        self.assertEqual(table_total(fold[:2])["comm_bytes_per_rank"], 4.0,
                         "an untimed parent with no field is not unpublished")


class TestMeasuredTermIndex(unittest.TestCase):
    """The uid → term index every measured reader takes is one producer's
    (`terms_by_uid`): a uid maps to a list, and `Folded.subtotal` gets flat
    term indices. The Playground's model editor once built `{uid: k}` itself
    and crashed on every model as soon as a measurement was there."""

    class _Folded:
        def __init__(self, n):
            from backend.measure.auto_sweep import MEASURED
            self.rows = tuple((MEASURED, 1.0, f"k{i}", None) for i in range(n))
            self.asked = []

        def subtotal(self, terms):
            terms = list(terms)
            self.asked.append(terms)
            return float(sum(self.rows[k][1] for k in terms))  # indexes: ints only

    def test_a_module_table_reads_the_shared_index(self):
        from backend.inference.batch_sweep import _measured_table, terms_by_uid
        rows = [{"uid": "a"}, {"uid": "b"}, {"uid": None}, {"uid": "c"}]
        terms = terms_by_uid(r.get("uid") for r in rows)
        self.assertEqual(terms, {"a": [0], "b": [1], "c": [3]})
        folded = self._Folded(len(rows))
        table = _measured_table([{"uid": "a", "depth": 0}, {"uid": "b", "depth": 1},
                                 {"uid": "c", "depth": 1}], terms, folded)
        self.assertEqual([m["t_measured_agg"] for m in table], [3.0, 1.0, 1.0])
        self.assertTrue(all(isinstance(k, int) for ask in folded.asked for k in ask))

    def test_a_declaration_parent_sums_flat_terms(self):
        from backend.inference.batch_sweep import _terms_of_rows, terms_by_uid
        rows = [{"uid": "a"}, {"uid": "a"}, {"uid": "b"}]
        terms = terms_by_uid(r["uid"] for r in rows)
        self.assertEqual(terms, {"a": [0, 1], "b": [2]})
        self.assertEqual(_terms_of_rows(rows, [0, 2], terms), [0, 1, 2])
        self.assertEqual(self._Folded(3).subtotal(_terms_of_rows(rows, [2], terms)), 1.0)
