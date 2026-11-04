"""Speculative draft cost: the MTP stack's forwards, counted on request.

By default the decode step is the target's verify forward alone. With
`speculative_draft` set, the step also runs the draft: the model's MTP stack
(`aux_kind: mtp`), priced by the same per-op machinery —

  recursive (EAGLE): K draft forwards, 1 token per sequence each;
  parallel (DFlash): one draft forward over K positions,

where K is the number of draft tokens per step; the target verifies K + 1
positions (the last accepted token plus the drafts).

The draft is amortised per accepted token exactly as the verify is: the
accepted tokens per step do not change, the step's time grows by the draft's.
A model without an MTP stack has no draft graph and is refused with a reason.
"""
from __future__ import annotations

import functools
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import resolve_snapshot, snapshot_config
from backend.inference.defaults import COMM_LATENCY_US_DEFAULT, LAUNCH_FLOOR_US_DEFAULT

MTP_MODEL = "XiaomiMiMo/MiMo-7B-RL"      # dense, one MTP layer
NO_MTP_MODEL = "meta-llama/Llama-3.2-1B"
DECODE_BATCH = 8


@functools.lru_cache(maxsize=None)
def _estimate(model_id: str, draft_tokens: int, style=None) -> dict:
    from backend.inference.capacity import estimate_capacity
    from backend.inference.phases import Speculation
    return estimate_capacity(
        config=snapshot_config(model_id),
        gpu_sku="H100-SXM",
        avg_input_tokens=2048,
        avg_output_tokens=256,
        model_id=model_id,
        s_verify=Speculation(draft_tokens).s_verify,
        acceptance_rate=0.7,
        speculative_draft=style,
        decode_batch_size=DECODE_BATCH,
    )


def _decode(res: dict) -> dict:
    return res["phases"]["decode"]


class TestDraftNotCountedByDefault(unittest.TestCase):

    def test_default_step_is_the_verify_alone(self):
        res = _estimate(MTP_MODEL, 2)
        self.assertNotIn("speculative_draft", _decode(res))
        self.assertNotIn("per_op_timing_draft", res)


class TestDraftCache(unittest.TestCase):
    """A draft request keeps the MTP layer's cache beside the backbone's: the
    MTP layer attends over the request's positions, so its cache grows with
    them. Without a draft only the backbone's cache is held."""

    def test_the_draft_layer_joins_the_cache_budget(self):
        verify = _estimate(MTP_MODEL, 1)["kv_cache"]
        drafted = _estimate(MTP_MODEL, 1, "recursive")["kv_cache"]
        self.assertEqual(verify["num_layers"], 36)
        self.assertEqual(drafted["num_layers"], 37)
        # MiMo's MTP layer is a backbone-shaped GQA layer: one more layer's
        # per-token bytes.
        per_layer = verify["growth_formula"]["result_bytes"] / 36
        self.assertEqual(drafted["growth_formula"]["result_bytes"],
                         verify["growth_formula"]["result_bytes"] + per_layer)

    def test_the_budget_serves_fewer_tokens_with_the_draft_cache(self):
        verify = _estimate(MTP_MODEL, 1)
        drafted = _estimate(MTP_MODEL, 1, "recursive")
        self.assertLess(drafted["max_tokens_per_gpu"],
                        verify["max_tokens_per_gpu"])

    def test_the_verify_forward_reads_only_the_backbone_cache(self):
        """The decode note describes the verify's KV read: the backbone's
        layers, with or without the draft's cache in the budget."""
        verify = _decode(_estimate(MTP_MODEL, 1))["note"]
        drafted = _decode(_estimate(MTP_MODEL, 1, "recursive"))["note"]
        self.assertTrue(drafted.startswith(verify.rstrip(".")))
        self.assertNotIn("hybrid", drafted)


class TestDraftWeights(unittest.TestCase):
    """A counted draft runs the MTP stack, so its weights are loaded as its
    cache is held: the budget once priced the draft's forwards and cache
    while saying "not loaded: aux module weights are off"."""

    def test_the_draft_stack_s_weights_join_the_budget(self):
        verify = _estimate(MTP_MODEL, 1)
        drafted = _estimate(MTP_MODEL, 1, "recursive")
        left_out = verify["weight_breakdown"]["left_out_stacks"]
        self.assertTrue(left_out, "the verify alone leaves the MTP stack out")
        self.assertEqual(drafted["weight_breakdown"]["left_out_stacks"], [])
        self.assertAlmostEqual(
            drafted["weight_bytes_total"],
            verify["weight_bytes_total"] + sum(s["bytes"] for s in left_out),
            delta=1.0)


class TestKIsDraftTokens(unittest.TestCase):

    def test_verify_covers_the_drafts_plus_one(self):
        """K draft tokens are verified in one forward over K + 1 positions,
        and 1 + αK tokens per sequence are kept."""
        for k in (1, 3):
            with self.subTest(k=k):
                dec = _decode(_estimate(MTP_MODEL, k))
                b = dec["binding"]
                self.assertEqual(b["speculative_draft_tokens"], k)
                self.assertEqual(b["s_query"], k + 1)
                self.assertEqual(b["s_verify"], k + 1)
                self.assertAlmostEqual(b["effective_tokens_per_step"],
                                       (1 + 0.7 * k) * dec["batch"])

    def test_zero_draft_tokens_is_autoregressive_decode(self):
        b = _decode(_estimate(MTP_MODEL, 0))["binding"]
        self.assertEqual(b["s_query"], 1)
        self.assertEqual(b["effective_tokens_per_step"], DECODE_BATCH)


class TestRecursiveDraft(unittest.TestCase):

    def test_one_draft_token_is_one_draft_forward(self):
        """K = 1: one 1-token forward through the MTP stack, and the step
        grows by exactly that forward."""
        verify = _decode(_estimate(MTP_MODEL, 1))
        res = _estimate(MTP_MODEL, 1, "recursive")
        draft = _decode(res)["speculative_draft"]
        self.assertEqual(draft["forwards"], 1)
        self.assertEqual(draft["tokens_per_forward"], 1)
        self.assertEqual(draft["t_sol_s"], draft["per_forward"]["t_sol_s"])
        self.assertGreater(draft["t_sol_s"], 0.0)
        self.assertAlmostEqual(
            _decode(res)["t_sol_s"], verify["t_sol_s"] + draft["t_sol_s"],
            delta=1e-12 * verify["t_sol_s"])

    def test_forwards_grow_with_draft_tokens_not_their_width(self):
        """Every recursive forward runs over one token, so more draft tokens
        add forwards of the same cost: K of them."""
        k1 = _decode(_estimate(MTP_MODEL, 1, "recursive"))["speculative_draft"]
        k3 = _decode(_estimate(MTP_MODEL, 3, "recursive"))["speculative_draft"]
        self.assertEqual(k3["forwards"], 3)
        self.assertEqual(k3["draft_tokens"], 3)
        self.assertEqual(k3["per_forward"]["flops_per_step"],
                         k1["per_forward"]["flops_per_step"])
        self.assertEqual(k3["flops_per_step"],
                         3 * k3["per_forward"]["flops_per_step"])

    def test_draft_reads_its_own_weights_and_cache(self):
        """The forward's traffic is the MTP stack's: its weights and the
        cache its attention reads, not zero and not the backbone's."""
        res = _estimate(MTP_MODEL, 1, "recursive")
        rows = res["per_op_timing_draft"]
        self.assertTrue(rows)
        weight = sum(int(r.get("weight_read_bytes_per_rank") or 0) for r in rows)
        act_in = sum(int(r.get("act_in_bytes_per_rank") or 0) for r in rows)
        self.assertGreater(weight, 0)
        # The attention leaf's KV read rides its input bytes.
        self.assertTrue(any(
            "attention" in r["op"] and int(r.get("act_in_bytes_per_rank") or 0) > 0
            for r in rows))
        self.assertGreater(act_in, 0)
        backbone = sum(int(r.get("weight_read_bytes_per_rank") or 0)
                       for r in res["per_op_timing_decode"])
        self.assertLess(weight, backbone)


class TestParallelDraft(unittest.TestCase):

    def test_one_forward_over_the_draft_positions(self):
        """A parallel draft is one forward over the K positions it drafts:
        every dense linear in it does K/2 times the work at K as at 2."""
        k2 = _estimate(MTP_MODEL, 2, "parallel")
        k4 = _estimate(MTP_MODEL, 4, "parallel")
        for res, k in ((k2, 2), (k4, 4)):
            d = _decode(res)["speculative_draft"]
            self.assertEqual(d["forwards"], 1)
            self.assertEqual(d["tokens_per_forward"], k)
        lin2 = {r["aggregation_id"]: r["flops_per_rank"]
                for r in k2["per_op_timing_draft"]
                if r["op"] == "linear" and r["flops_per_rank"]}
        lin4 = {r["aggregation_id"]: r["flops_per_rank"]
                for r in k4["per_op_timing_draft"]
                if r["op"] == "linear" and r["flops_per_rank"]}
        self.assertTrue(lin2)
        self.assertEqual(lin2.keys(), lin4.keys())
        for agg, flops in lin2.items():
            self.assertEqual(lin4[agg], 2 * flops, agg)


    def test_one_draft_token_is_the_recursive_forward(self):
        """At K = 1 the two styles run the same single 1-token forward."""
        par = _decode(_estimate(MTP_MODEL, 1, "parallel"))["speculative_draft"]
        rec = _decode(_estimate(MTP_MODEL, 1, "recursive"))["speculative_draft"]
        self.assertEqual(par["tokens_per_forward"], 1)
        self.assertEqual(par["per_forward"]["flops_per_step"],
                         rec["per_forward"]["flops_per_step"])
        self.assertEqual(par["t_sol_s"], rec["t_sol_s"])


class TestDraftAmortisation(unittest.TestCase):

    def test_accepted_tokens_unchanged_and_divided_by_the_whole_step(self):
        verify = _decode(_estimate(MTP_MODEL, 3))
        res = _estimate(MTP_MODEL, 3, "recursive")
        dec = _decode(res)
        tokens = dec["binding"]["effective_tokens_per_step"]
        self.assertEqual(tokens, verify["binding"]["effective_tokens_per_step"])
        self.assertAlmostEqual(dec["throughput"]["sol"]["per_group"],
                               tokens / dec["t_sol_s"])
        self.assertLess(dec["throughput"]["sol"]["per_group"],
                        verify["throughput"]["sol"]["per_group"])
        share = dec["speculative_draft"]["t_sol_share"]
        self.assertAlmostEqual(share, dec["speculative_draft"]["t_sol_s"] / dec["t_sol_s"])

    def test_step_folds_still_close(self):
        """The phase total equals its rows, the draft's included."""
        from backend.inference.invariants import (
            _inv_flops_closure, _inv_phase_closure)
        for style in ("recursive", "parallel"):
            res = _estimate(MTP_MODEL, 3, style)
            for r in _inv_phase_closure(res) + _inv_flops_closure(res):
                self.assertTrue(r.passed, f"{style}: {r.name}: {r.detail}")

    def test_batch_rescale_keeps_the_draft(self):
        """A rescale re-prices the draft too: at the anchor batch it
        reproduces the phase it was rescaled from."""
        from tests.sweep_parity import rescale_phase, response_cell
        from backend.inference.comm_cost import Fabric
        from backend.inference.gpu_specs import get_gpu
        res = _estimate(MTP_MODEL, 3, "recursive")
        dec = _decode(res)
        at_anchor = rescale_phase(
            dec, res["per_op_timing_decode"], {"B": dec["batch"]},
            res["peak_flops_per_sec"], res["peak_hbm_bw_bytes_per_sec"],
            Fabric.of(get_gpu("H100-SXM"), COMM_LATENCY_US_DEFAULT * 1e-6),
            cell=response_cell(), draft_cell=response_cell(), breakdown=True,
            draft_rows=res["per_op_timing_draft"],
            launch_s=LAUNCH_FLOOR_US_DEFAULT * 1e-6)
        self.assertAlmostEqual(at_anchor["t_sol_s"], dec["t_sol_s"],
                               delta=1e-9 * dec["t_sol_s"])
        self.assertAlmostEqual(at_anchor["speculative_draft"]["t_sol_s"],
                               dec["speculative_draft"]["t_sol_s"],
                               delta=1e-9 * dec["t_sol_s"])


class TestPerOpTableTotal(unittest.TestCase):
    """The per-op table's Total row is the backend's (`op_table_total`) and
    says what the phase says: the page draws it and adds nothing."""

    def _close(self, a, b, what):
        self.assertAlmostEqual(a, b, delta=max(1e-12, 1e-9 * abs(b)), msg=what)

    def test_phase_table_total_is_the_phase_step(self):
        for style in (None, "recursive"):
            dec = _decode(_estimate(MTP_MODEL, 3, style))
            total = dec["op_table_total"]
            self._close(total["t_sol_s"], dec["t_sol_s"], f"{style}: t_sol")
            self._close(total["flops_per_rank"], dec["flops_per_step"],
                        f"{style}: flops")
            self._close(total["access_bytes_per_rank"], dec["total_bytes"],
                        f"{style}: bytes")
            # The measured total is not a column sum: it is the phase's one
            # measured formula, all or nothing (`dec["t_measured_s"]`).
            self.assertNotIn("t_measured_s", total)

    def test_draft_table_total_is_one_forward(self):
        res = _estimate(MTP_MODEL, 3, "recursive")
        draft = _decode(res)["speculative_draft"]
        total = draft["op_table_total"]
        fwd = draft["per_forward"]
        self._close(total["t_sol_s"], fwd["t_sol_s"], "t_sol")
        self._close(total["flops_per_rank"], fwd["flops_per_step"], "flops")
        self._close(total["access_bytes_per_rank"], fwd["total_bytes"], "bytes")
        self._close(total["comm_bytes_per_rank"], fwd["comm_bytes_per_rank"],
                    "comm bytes")
        self._close(total["t_sol_s"],
                    sum(r["t_sol_s"] for r in res["per_op_timing_draft"]),
                    "Σ draft rows")

    def test_every_priced_row_states_its_own_terms(self):
        """t_comm_s and t_sol_s on every per-op row and every fold leaf —
        the draft's depth-0 row included — so no consumer substitutes one
        field for a missing other."""
        res = _estimate(MTP_MODEL, 3, "recursive")
        dec = _decode(res)
        rows = (res["per_op_timing_decode"] + res["per_op_timing_draft"]
                + [r for r in dec["flops_by_module"] if "t_op_s" in r])
        self.assertTrue(rows)
        for r in rows:
            for key in ("t_compute_s", "t_memory_s", "t_comm_s", "t_sol_s"):
                self.assertIsNotNone(r.get(key), f"{r.get('name')}: {key}")

    def test_rescale_at_the_anchor_keeps_the_table_total(self):
        from tests.sweep_parity import rescale_phase, response_cell
        from backend.inference.comm_cost import Fabric
        from backend.inference.gpu_specs import get_gpu
        res = _estimate(MTP_MODEL, 3, "recursive")
        dec = _decode(res)
        at_anchor = rescale_phase(
            dec, res["per_op_timing_decode"], {"B": dec["batch"]},
            res["peak_flops_per_sec"], res["peak_hbm_bw_bytes_per_sec"],
            Fabric.of(get_gpu("H100-SXM"), COMM_LATENCY_US_DEFAULT * 1e-6),
            cell=response_cell(), draft_cell=response_cell(), breakdown=True,
            draft_rows=res["per_op_timing_draft"],
            launch_s=LAUNCH_FLOOR_US_DEFAULT * 1e-6)
        self._close(at_anchor["op_table_total"]["t_sol_s"], at_anchor["t_sol_s"],
                    "rescaled table total is the rescaled step")
        self._close(at_anchor["op_table_total"]["t_sol_s"],
                    dec["op_table_total"]["t_sol_s"], "anchor batch")


class TestPipelinedDraft(unittest.TestCase):
    """pp=2 with a recursive draft: the published step and its split by
    bound are on one basis — every stage plus the draft — and a re-pricing
    sums the same stages and the same draft."""

    @classmethod
    def setUpClass(cls):
        from backend.inference.capacity import estimate_capacity
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.phases import Speculation
        cls.res = estimate_capacity(
            config=snapshot_config(MTP_MODEL), gpu_sku="H100-SXM", num_gpus=2,
            parallel=ParallelConfig(pp=2), avg_input_tokens=2048,
            avg_output_tokens=256, model_id=MTP_MODEL, s_verify=Speculation(3).s_verify,
            acceptance_rate=0.7, speculative_draft="recursive",
            decode_batch_size=DECODE_BATCH)

    def test_the_anchor_step_and_its_bounds_include_the_draft(self):
        dec = _decode(self.res)
        pipe = dec["pipeline_time"]
        draft = dec["speculative_draft"]
        self.assertGreater(draft["t_sol_s"], 0)
        self.assertAlmostEqual(pipe["t_sol"], dec["t_sol_s"],
                               delta=1e-12 * dec["t_sol_s"])
        for by_bound in (dec["t_sol_by_bound"], pipe["t_sol_by_bound"]):
            self.assertAlmostEqual(sum(by_bound.values()), dec["t_sol_s"],
                                   delta=1e-12 * dec["t_sol_s"])
        self.assertAlmostEqual(draft["t_sol_share"],
                               draft["t_sol_s"] / dec["t_sol_s"])

    def test_a_reprice_keeps_the_step_and_the_draft_share(self):
        """Re-priced at the anchor batch, the pipelined step — every stage
        plus the draft — and the draft's share of it are the anchor's."""
        from tests.sweep_parity import rescale_phase, response_cell
        from backend.inference.comm_cost import Fabric
        from backend.inference.gpu_specs import get_gpu
        res, dec = self.res, _decode(self.res)
        out = rescale_phase(
            dec, res["per_op_timing_decode"], {"B": dec["batch"]},
            res["peak_flops_per_sec"], res["peak_hbm_bw_bytes_per_sec"],
            Fabric.of(get_gpu("H100-SXM"), COMM_LATENCY_US_DEFAULT * 1e-6),
            cell=response_cell(), draft_cell=response_cell(), breakdown=True,
            draft_rows=res["per_op_timing_draft"],
            launch_s=LAUNCH_FLOOR_US_DEFAULT * 1e-6)
        self.assertAlmostEqual(out["t_sol_s"], dec["t_sol_s"],
                               delta=1e-9 * dec["t_sol_s"])
        self.assertEqual(out["time_unpublished_reason"], "")
        self.assertAlmostEqual(out["speculative_draft"]["t_sol_share"],
                               dec["speculative_draft"]["t_sol_share"],
                               delta=1e-9)


class TestEveryKIsOneWalk(unittest.TestCase):
    """A sweep over verify widths (`s_verifies`): every speculative width
    is one walk of the deployment, each K's points its rows read at K + 1
    — and each point is the estimate a direct call at that K publishes."""

    WIDTHS = (2, 3, 5)
    SEQ = 2048

    def _sweep(self, widths, pendings=None):
        from unittest import mock

        from backend.inference import capacity
        real = capacity._capacity_pending

        def counted(p, **kw):
            if pendings is not None:
                pendings.append(p.s_verify)
            return real(p, **kw)
        with mock.patch.object(capacity, "_capacity_pending", counted):
            plan, walked = capacity.sweep_points(
                seq_lens=[self.SEQ] * len(widths), s_verifies=list(widths),
                config=snapshot_config(MTP_MODEL), gpu_sku="H100-SXM",
                avg_output_tokens=256, model_id=MTP_MODEL,
                s_verify=widths[0], acceptance_rate=0.7,
                speculative_draft="recursive",
                batch_grid=[{"key": "b8", "decode_batch_size": DECODE_BATCH},
                            {"key": "max", "decode_batch_size": None}])
        return plan.shape(walked)

    def test_every_speculative_width_reads_one_walk(self):
        pendings: list = []
        swept = self._sweep(self.WIDTHS, pendings)
        self.assertEqual(pendings, [self.WIDTHS[0]])
        for i, width in enumerate(self.WIDTHS):
            got = swept["b8"][i]["sol"]
            want = _estimate(MTP_MODEL, width - 1, "recursive")
            self.assertEqual(_decode(got)["binding"]["s_verify"], width)
            self.assertEqual(_decode(got)["speculative_draft"]["draft_tokens"],
                             width - 1)
            self.assertEqual(_decode(got)["speculative_draft"]["forwards"],
                             _decode(want)["speculative_draft"]["forwards"])
            self.assertAlmostEqual(_decode(got)["t_sol_s"],
                                   _decode(want)["t_sol_s"],
                                   delta=1e-9 * _decode(want)["t_sol_s"])
            self.assertEqual(got["max_batch_per_gpu"], want["max_batch_per_gpu"])
            self.assertEqual(swept["max"][i]["sol"]["max_batch_per_gpu"],
                             want["max_batch_per_gpu"])

    def test_a_step_that_does_not_speculate_is_its_own_walk(self):
        pendings: list = []
        from unittest import mock

        from backend.inference import capacity
        real = capacity._capacity_pending

        def counted(p, **kw):
            pendings.append(p.s_verify)
            return real(p, **kw)
        with mock.patch.object(capacity, "_capacity_pending", counted):
            _plan, walked = capacity.sweep_points(
                seq_lens=[self.SEQ] * 3, s_verifies=[1, 2, 3],
                config=snapshot_config(MTP_MODEL), gpu_sku="H100-SXM",
                avg_output_tokens=256, model_id=MTP_MODEL,
                decode_batch_size=DECODE_BATCH)
        self.assertEqual(pendings, [1, 2])
        steps = [_decode(cell[0]["sol"])["t_sol_s"] for cell in walked]
        self.assertLess(steps[0], steps[1])
        self.assertLess(steps[1], steps[2])


class TestRefusals(unittest.TestCase):

    def test_model_without_mtp_stack_is_refused(self):
        from backend.arch.eval.program import Program
        from backend.inference.speculative_draft import SpeculativeDraft
        program = Program.of(resolve_snapshot(NO_MTP_MODEL))
        reason = SpeculativeDraft.refusal(program, draft_tokens=2)
        self.assertIsNotNone(reason)
        self.assertIn("MTP", reason)
        with self.assertRaises(ValueError) as ctx:
            _estimate(NO_MTP_MODEL, 2, "recursive")
        self.assertIn(reason, str(ctx.exception))

    def test_zero_draft_tokens_drafts_nothing(self):
        with self.assertRaises(ValueError):
            _estimate(MTP_MODEL, 0, "parallel")

    def test_a_wire_draft_of_zero_tokens_is_refused_as_of_refuses_it(self):
        # `of` and `from_wire` build the same value; the declaration that
        # 0 draft tokens drafts nothing holds for both. A parallel draft of
        # 0 tokens read back off the wire was a one-token forward
        # (`s_verify or 1`) priced into decode.
        from backend.arch.eval.program import Program
        from backend.inference.speculative_draft import SpeculativeDraft
        program = Program.of(resolve_snapshot(MTP_MODEL))
        reason = SpeculativeDraft.refusal(program, draft_tokens=0)
        self.assertIsNotNone(reason)
        wire = SpeculativeDraft.of(
            program, style="parallel", draft_tokens=2).to_wire()
        self.assertEqual(SpeculativeDraft.from_wire(wire).draft_tokens, 2)
        with self.assertRaises(ValueError) as ctx:
            SpeculativeDraft.from_wire({**wire, "draft_tokens": 0})
        self.assertIn(reason, str(ctx.exception))
        with self.assertRaises(ValueError):
            SpeculativeDraft.of(program, style="parallel", draft_tokens=0)

    def test_endpoint_reports_the_estimate_refusal(self):
        snapshot_config(NO_MTP_MODEL)
        from fastapi.testclient import TestClient
        from backend.api import app
        client = TestClient(app)
        no = client.get("/inference/speculative-draft",
                        params={"model_id": NO_MTP_MODEL}).json()
        self.assertFalse(no["available"])
        self.assertIn("MTP", no["unavailable_reason"])
        yes = client.get("/inference/speculative-draft",
                         params={"model_id": MTP_MODEL,
                                 "speculative_draft_tokens": 2}).json()
        self.assertTrue(yes["available"])
        self.assertIsNone(yes["unavailable_reason"])
        self.assertEqual(yes["draft_stacks"], ["mtp"])
        refused = client.post("/inference/estimate", json={
            "model_id": NO_MTP_MODEL, "gpu_sku": "H100-SXM",
            "speculative_draft_tokens": 2, "acceptance_rate": 0.7,
            "speculative_draft": "recursive"})
        self.assertEqual(refused.status_code, 400)
        self.assertIn(no["unavailable_reason"], refused.json()["detail"])

    def test_the_retired_window_field_is_refused_by_name(self):
        """`speculative_window` meant draft tokens + 1. A caller still
        sending it is told, not silently given autoregressive decode."""
        from fastapi.testclient import TestClient
        from backend.api import app
        resp = TestClient(app).post("/inference/estimate", json={
            "model_id": MTP_MODEL, "gpu_sku": "H100-SXM",
            "speculative_window": 3})
        self.assertEqual(resp.status_code, 422)
        self.assertIn("speculative_draft_tokens", resp.text)

    def test_the_get_endpoints_refuse_the_retired_window_query(self):
        """The two GET endpoints that take K refuse the old query name with
        the body validator's wording, rather than ignoring it into K = 0."""
        from fastapi.testclient import TestClient
        from backend.api import app
        from backend.api._models import SPECULATIVE_WINDOW_RETIRED
        client = TestClient(app)
        for path in ("/inference/speculative-draft",
                     "/inference/compute-graph"):
            with self.subTest(path=path):
                resp = client.get(path, params={
                    "model_id": MTP_MODEL, "speculative_window": 3})
                self.assertEqual(resp.status_code, 422)
                self.assertEqual(resp.json()["detail"],
                                 SPECULATIVE_WINDOW_RETIRED)


class TestDisaggregated(unittest.TestCase):

    def test_draft_runs_on_the_decode_deployment(self):
        """Disaggregated serving drafts where it decodes: a pipelined prefill
        deployment (which could not place the MTP head) does not refuse a
        draft the decode deployment can run."""
        snapshot_config(MTP_MODEL)
        from fastapi.testclient import TestClient
        from backend.api import app
        resp = TestClient(app).post("/inference/estimate", json={
            "model_id": MTP_MODEL, "gpu_sku": "H100-SXM",
            "avg_input_tokens": 2048, "avg_output_tokens": 256,
            "speculative_draft_tokens": 2, "acceptance_rate": 0.7,
            "speculative_draft": "recursive",
            "parallel": {"pp": 2}, "num_gpus": 2,
            "decode": {"parallel": {}, "num_gpus": 1}})
        self.assertEqual(resp.status_code, 200, resp.text)
        j = resp.json()
        self.assertEqual(j["phases"]["decode"]["speculative_draft"]["forwards"], 2)
        self.assertTrue(j["per_op_timing_draft"])


if __name__ == "__main__":
    unittest.main()
