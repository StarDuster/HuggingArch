"""One roofline law per row, and another batch or card priced by the same law.

Each per-op row carries its law as sympy expressions over the batch and the
card (`roofline_expr`); the anchor row's numbers are that law at the phase's
batch on the phase's card, and every re-evaluation — the parallel sweep's
max batch, another card — substitutes into the same expressions.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.defaults import COMM_LATENCY_US_DEFAULT, LAUNCH_FLOOR_US_DEFAULT
from backend.inference.comm_cost import Fabric
from backend.inference.parallel_config import ParallelConfig
from backend.inference.row_roofline import LawBook, RowCard

pytestmark = pytest.mark.regression

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _config(stem: str) -> dict:
    return json.loads((SNAPSHOTS / f"{stem}.json").read_text())["config"]


def _fabric(gpu_sku: str) -> Fabric:
    """The card's interconnect, as the estimate priced it."""
    from backend.inference.gpu_specs import get_gpu
    return Fabric.of(get_gpu(gpu_sku), COMM_LATENCY_US_DEFAULT * 1e-6)


def _launch(gpu_sku: str) -> float:
    """The launch floor, as the estimate priced it (the default)."""
    return LAUNCH_FLOOR_US_DEFAULT * 1e-6


def _resolved(stem: str):
    """Resolve the spec once so capacity never re-matches (single entry)."""
    from tests.snapshots import shared_loader
    return shared_loader().match(_config(stem), model_id=stem.replace("__", "/", 1))


class TestTheAnchorRowIsTheLaw(unittest.TestCase):
    """Every per-op row's published times equal its own expressions at the
    phase's batch on the phase's card — including under a
    routing imbalance, where the ladder used to drift from the anchor."""

    def _check(self, estimate, phase_key, rows_key, fabric):
        phase = estimate["phases"][phase_key]
        batch = int(phase["batch"])
        comm = 0
        for row in estimate[rows_key]:
            at = LawBook().row_at(row, {**row["point_law"]["point"], "B": batch}, RowCard.of(
                row, peak_flops=estimate["peak_flops_per_sec"],
                peak_bw=estimate["peak_hbm_bw_bytes_per_sec"], fabric=fabric,
                launch_s=_launch("H100-SXM")))
            self.assertEqual(at.flops, row["flops_per_rank"], row["op"])
            self.assertEqual(at.access, row["access_bytes_per_rank"], row["op"])
            self.assertAlmostEqual(at.times.t_compute, row["t_compute_s"], delta=1e-12, msg=row["op"])
            self.assertAlmostEqual(at.times.t_memory, row["t_memory_s"], delta=1e-12, msg=row["op"])
            self.assertAlmostEqual(at.times.t_comm, row["t_comm_s"], delta=1e-12, msg=row["op"])
            self.assertAlmostEqual(at.times.t_launch, row["t_launch_s"], delta=1e-12, msg=row["op"])
            self.assertAlmostEqual(at.times.t_op, row["t_op_s"], delta=1e-12, msg=row["op"])
            comm += bool(row["t_comm_s"])
            # The row's labels come off the same times and components.
            terms = (row["t_compute_s"], row["t_memory_s"], row["t_comm_s"],
                     row["t_launch_s"])
            if row["t_op_s"] > 0:
                self.assertEqual(
                    row["bound"],
                    ("compute", "memory", "comm", "launch")[terms.index(max(terms))],
                    row["op"])
            else:
                self.assertIsNone(row["bound"], row["op"])
            if row["access_bytes_per_rank"]:
                self.assertEqual(row["arithmetic_intensity"],
                                 row["flops_per_rank"] / row["access_bytes_per_rank"])
        self.assertGreater(comm, 0, "DP-attention x EP=4 prices its route exchanges")
        by_bound = phase["t_sol_by_bound"]
        self.assertAlmostEqual(sum(by_bound.values()), phase["t_sol_s"],
                               delta=1e-12 * phase["t_sol_s"])
        for bound, t in by_bound.items():
            self.assertAlmostEqual(phase["t_sol_by_bound_share"][bound]
                                   * phase["t_sol_s"], t,
                                   delta=1e-12 * phase["t_sol_s"])
        # The module table: a leaf carries its row's bound, a parent none;
        # every row's intensity is its shown FLOPs over its shown bytes.
        table = phase["flops_by_module"]
        for i, m in enumerate(table):
            parent = (i + 1 < len(table)
                      and table[i + 1]["depth"] > m["depth"])
            if parent:
                self.assertIsNone(m["bound"], m["name"])
            if m["access_bytes_agg"]:
                self.assertAlmostEqual(
                    m["arithmetic_intensity"],
                    (m["flops_per_rank"] or 0) / m["access_bytes_agg"],
                    msg=m["name"])

    def test_balanced_and_imbalanced_routing(self):
        from backend.inference.capacity import estimate_capacity
        config = _config("Qwen__Qwen3-30B-A3B")
        resolved = _resolved("Qwen__Qwen3-30B-A3B")
        rows_at = {}
        for rho in (1.0, 1.5):
            estimate = estimate_capacity(
                config=config, resolved=resolved, gpu_sku="H100-SXM", num_gpus=4,
                parallel=ParallelConfig(dp_attention=4, ep_moe=4),
                ep_load_factor=rho,
                avg_input_tokens=2048, avg_output_tokens=256,
                decode_batch_size=8)
            rows_at[rho] = {r["uid"]: r for r in estimate["per_op_timing_decode"]}
            self.assertTrue(all(r.get("act_dtype") for r in rows_at[rho].values()
                                if r.get("flops_per_rank")),
                            "every leaf that does FLOPs names its activation dtype")
            self._check(estimate, "prefill", "per_op_timing", _fabric("H100-SXM"))
            self._check(estimate, "decode", "per_op_timing_decode", _fabric("H100-SXM"))
        # The busiest rank carries more than the mean token load: in the
        # experts it computes and in the rows its route exchange moves.
        grown = {rows_at[1.0][uid]["op"] for uid in rows_at[1.0]
                 if (rows_at[1.5][uid]["access_bytes_per_rank"]
                     > rows_at[1.0][uid]["access_bytes_per_rank"])}
        self.assertIn("all_to_all", grown)
        self.assertTrue(grown - {"all_to_all", "permute", "combine"})


class TestARowReadAtAnotherBatch(unittest.TestCase):
    """The parallel sweep walks once at B=1 and reads every row at a card's
    max batch. The collectives and the FLOPs read there are what a walk at
    that batch prices — the route exchange's payload included, under a
    routing imbalance, which is the busiest rank's buffer — and so
    are the bytes, the embedding's gathered rows among them."""

    def test_rescaled_comm_and_compute_equal_a_walk(self):
        from backend.inference.capacity import estimate_capacity
        from tests.sweep_parity import rescale_phase, response_cell
        config = _config("Qwen__Qwen3-30B-A3B")
        common = dict(config=config, resolved=_resolved("Qwen__Qwen3-30B-A3B"),
                      gpu_sku="H100-SXM", num_gpus=4,
                      parallel=ParallelConfig(dp_attention=4, ep_moe=4),
                      ep_load_factor=1.5,
                      avg_input_tokens=2048, avg_output_tokens=256)
        one = estimate_capacity(prefill_batch_size=1, decode_batch_size=1, **common)
        six = estimate_capacity(prefill_batch_size=6, decode_batch_size=6, **common)
        for phase, rows in (("prefill", "per_op_timing"),
                            ("decode", "per_op_timing_decode")):
            got = rescale_phase(one["phases"][phase], one[rows], {"B": 6},
                                 one["peak_flops_per_sec"],
                                 one["peak_hbm_bw_bytes_per_sec"],
                                 _fabric("H100-SXM"), cell=response_cell(),
                                 breakdown=True, launch_s=_launch("H100-SXM"))
            want = six["phases"][phase]
            self.assertGreater(want["t_comm_s"], 0)
            for key in ("t_comm_s", "t_compute_s", "t_memory_s", "t_sol_s"):
                self.assertAlmostEqual(got[key], want[key],
                                       delta=1e-12 * want[key], msg=(phase, key))
            self.assertAlmostEqual(sum(got["t_sol_by_bound"].values()),
                                   got["t_sol_s"], delta=1e-12 * got["t_sol_s"])


class TestTheSecondCardIsPricedByTheSameLoop(unittest.TestCase):
    """A7: the Gallery re-stamps one layout onto another card. Its comm rows
    and per-row peaks must equal a direct estimate on that card, row for
    row — the same `price_collective` and the same `act_dtype` lookup."""

    def test_stamped_rows_equal_a_direct_estimate(self):
        from backend.inference.parallel_sweep import _estimate_layout
        from backend.inference.capacity import estimate_capacity
        config = _config("Qwen__Qwen3-8B")
        parallel = ParallelConfig(tp_attention=2, tp_dense_ffn=2)
        # Same overlay batch on both sides: the stamp keeps the source card's
        # anchor rows, and a direct estimate anchors decode at ITS max batch,
        # so the equality is stated at a pinned batch (comm bytes scale with it).
        common = dict(config=config, resolved=_resolved("Qwen__Qwen3-8B"),
                      parallel=parallel, num_gpus=2,
                      avg_input_tokens=1024, avg_output_tokens=128,
                      prefill_batch_size=1, decode_batch_size=8)
        stamped = _estimate_layout({**common, "gpu_skus": ["H100-SXM", "A100-80G-SXM"]})
        direct = estimate_capacity(gpu_sku="A100-80G-SXM", **common)
        for key in ("per_op_timing", "per_op_timing_decode"):
            got = {r["aggregation_id"]: r for r in stamped["A100-80G-SXM"]["sol"][key]}
            want = {r["aggregation_id"]: r for r in direct[key]}
            self.assertEqual(set(got), set(want))
            comm_rows = 0
            for agg, w in want.items():
                g = got[agg]
                self.assertEqual(g["peak_flops_per_sec_for_op"], w["peak_flops_per_sec_for_op"], agg)
                if w.get("is_comm") and w.get("comm_world", 0) > 1:
                    comm_rows += 1
                    self.assertAlmostEqual(g["t_comm_s"], w["t_comm_s"],
                                           delta=1e-12 * w["t_comm_s"], msg=agg)
                    self.assertAlmostEqual(g["comm_bytes_per_rank"], w["comm_bytes_per_rank"],
                                           delta=1e-12 * w["comm_bytes_per_rank"], msg=agg)
                    # A Gallery point shows no breakdown: the collective's
                    # per-phase rows are the Inference estimate's alone.
                    self.assertNotIn("comm_phases", g, agg)
            self.assertGreater(comm_rows, 0, f"{key}: TP=2 has collectives to re-price")


_PP_TIMES = ("t_compute_s", "t_memory_s", "t_comm_s", "t_sol_s")


def _pp_estimate(stem: str, batch: int, gpu_sku: str = "H100-SXM",
                 draft: bool = False) -> dict:
    """A pp=2 estimate pinned at `batch` in both phases."""
    from backend.inference.capacity import estimate_capacity
    from backend.inference.phases import Speculation
    spec = (dict(s_verify=Speculation(3).s_verify, acceptance_rate=0.7,
                 speculative_draft="recursive") if draft else {})
    return estimate_capacity(
        config=_config(stem), resolved=_resolved(stem), gpu_sku=gpu_sku,
        num_gpus=2, parallel=ParallelConfig(pp=2), avg_input_tokens=2048,
        avg_output_tokens=256, model_id=stem.replace("__", "/", 1),
        prefill_batch_size=batch, decode_batch_size=batch, **spec)


class TestAPipelineStepReadElsewhere(unittest.TestCase):
    """pp>1: the step is every stage's rows plus the hops across the cuts.
    Read at another batch or on another card, it is what a walk there
    prices — each stage substituted into its own row laws, the speculative
    draft beside them — and the measured track stays unpublished."""

    CASES = (("XiaomiMiMo__MiMo-7B-RL", False),
             ("XiaomiMiMo__MiMo-7B-RL", True),
             ("Qwen__Qwen3-30B-A3B", False))

    def _same_step(self, got: dict, want: dict, msg: Any) -> None:
        self.assertEqual(got["batch"], want["batch"], msg)
        self.assertEqual(got["time_unpublished_reason"], "", msg)
        for key in _PP_TIMES:
            self.assertIsNotNone(got[key], (msg, key))
            self.assertAlmostEqual(got[key], want[key],
                                   delta=1e-9 * want[key], msg=(msg, key))
        for bound, t in want["t_sol_by_bound"].items():
            self.assertAlmostEqual(got["t_sol_by_bound"][bound], t,
                                   delta=1e-9 * want["t_sol_s"],
                                   msg=(msg, bound))
        got_pp, want_pp = got["pipeline_time"], want["pipeline_time"]
        self.assertEqual(len(got_pp["per_stage"]), 2, msg)
        for g, w in zip(got_pp["per_stage"], want_pp["per_stage"],
                        strict=True):
            self.assertAlmostEqual(g, w, delta=1e-9 * w, msg=msg)
        self.assertGreater(got_pp["t_comm"], 0, (msg, "the hop is priced"))
        self.assertIsNone(got["t_measured_s"], msg)
        if want.get("speculative_draft"):
            g, w = got["speculative_draft"], want["speculative_draft"]
            self.assertAlmostEqual(g["t_sol_s"], w["t_sol_s"],
                                   delta=1e-9 * w["t_sol_s"], msg=msg)
            self.assertAlmostEqual(g["t_sol_share"], w["t_sol_share"],
                                   delta=1e-9, msg=msg)

    def test_another_batch_is_a_walk_at_that_batch(self):
        from tests.sweep_parity import rescale_phase, response_cell
        for stem, draft in self.CASES:
            four = _pp_estimate(stem, 4, draft=draft)
            sixteen = _pp_estimate(stem, 16, draft=draft)
            for phase, rows in (("prefill", "per_op_timing"),
                                ("decode", "per_op_timing_decode")):
                got = rescale_phase(
                    four["phases"][phase], four[rows], {"B": 16},
                    four["peak_flops_per_sec"],
                    four["peak_hbm_bw_bytes_per_sec"], _fabric("H100-SXM"),
                    cell=response_cell(), draft_cell=response_cell(), breakdown=True,
                    launch_s=_launch("H100-SXM"),
                    draft_rows=(four.get("per_op_timing_draft")
                                if phase == "decode" else None))
                self._same_step(got, sixteen["phases"][phase],
                                (stem, draft, phase))

    def test_another_card_is_an_estimate_on_that_card(self):
        """The sweep's points: one walk, every card re-priced from it."""
        from backend.inference.parallel_sweep import _estimate_layout
        stem = "Qwen__Qwen3-30B-A3B"
        common = dict(config=_config(stem), resolved=_resolved(stem),
                      parallel=ParallelConfig(pp=2), num_gpus=2,
                      avg_input_tokens=2048, avg_output_tokens=256,
                      prefill_batch_size=4, decode_batch_size=16)
        swept = _estimate_layout({**common,
                                  "gpu_skus": ["H100-SXM", "A100-80G-SXM"]})
        direct = _pp_estimate(stem, 4, gpu_sku="A100-80G-SXM")
        direct_decode = _pp_estimate(stem, 16, gpu_sku="A100-80G-SXM")
        point = swept["A100-80G-SXM"]["sol"]["phases"]
        self._same_step(point["prefill"], direct["phases"]["prefill"],
                        "prefill")
        self._same_step(point["decode"], direct_decode["phases"]["decode"],
                        "decode")
        for phase in ("prefill", "decode"):
            self.assertGreater(
                point[phase]["throughput"]["sol"]["per_group"], 0, phase)


class TestAWithheldStepStaysWithheld(unittest.TestCase):
    """MiniMax-M2.7 at pp=2 holds 107 GB of weights per rank: no batch runs
    on an H100, so it publishes no step — no batch, no time, no per-op rows,
    no module table. The walk behind it is re-priced at another batch and
    still has no time; re-priced onto an H200, which admits a batch, it has
    one."""

    STEM = "MiniMaxAI__MiniMax-M2.7"
    ROWS = ("per_op_timing", "per_op_timing_decode", "per_op_timing_draft")

    def _common(self) -> dict:
        return dict(config=_config(self.STEM), resolved=_resolved(self.STEM),
                    model_id=self.STEM.replace("__", "/", 1),
                    parallel=ParallelConfig(pp=2), num_gpus=2,
                    avg_input_tokens=2048, avg_output_tokens=256,
                    selection_mode="theoretical_only")

    def _no_step(self, result: dict, msg: Any) -> None:
        reason = result["infeasibility_reason"]
        self.assertTrue(reason, msg)
        self.assertFalse(set(self.ROWS) & set(result), msg)
        self.assertNotIn("vs_naive_delta", result, msg)
        for phase in ("prefill", "decode"):
            self.assertEqual(result["phases"][phase],
                             {"phase": phase, "batch": None,
                              "time_unpublished_reason": reason}, (msg, phase))

    def _withheld(self, phase: dict, msg: Any) -> None:
        self.assertTrue(phase["time_unpublished_reason"], msg)
        for key in (*_PP_TIMES, "t_sol_by_bound", "t_measured_s"):
            self.assertIsNone(phase[key], (msg, key))
        self.assertFalse({"t_sol", "t_compute", "t_memory", "t_comm",
                          "per_stage", "t_sol_by_bound"}
                         & set(phase["pipeline_time"]), msg)

    def test_the_estimate_publishes_no_step_and_its_walk_no_time(self):
        from tests.sweep_parity import rescale_phase, response_cell
        from backend.inference.capacity import estimate_capacity, estimate_pending
        self._no_step(estimate_capacity(gpu_sku="H100-SXM", **self._common()),
                      "estimate")
        walk = estimate_pending(gpu_sku="H100-SXM", **self._common()).finish()
        for phase, rows in (("prefill", "per_op_timing"),
                            ("decode", "per_op_timing_decode")):
            anchor = walk["phases"][phase]
            self._withheld(anchor, phase)
            again = rescale_phase(
                anchor, walk[rows], {"B": 4 * anchor["batch"]},
                walk["peak_flops_per_sec"],
                walk["peak_hbm_bw_bytes_per_sec"], _fabric("H100-SXM"),
                cell=response_cell(), breakdown=True, launch_s=_launch("H100-SXM"))
            self._withheld(again, phase)
            self.assertEqual(again["time_unpublished_reason"],
                             anchor["time_unpublished_reason"])

    def test_the_sweep_withholds_on_the_card_without_room(self):
        from backend.inference.parallel_sweep import _estimate_layout
        swept = _estimate_layout({**self._common(),
                                  "gpu_skus": ["H100-SXM", "H200"]})
        h100, h200 = swept["H100-SXM"]["sol"], swept["H200"]["sol"]
        self.assertTrue(h100["infeasibility_reason"])
        self.assertIsNone(h200["infeasibility_reason"])
        self._no_step(h100, "H100-SXM")
        for phase in ("prefill", "decode"):
            self.assertEqual(h200["phases"][phase]["time_unpublished_reason"],
                             "")
            self.assertGreater(h200["phases"][phase]["t_sol_s"], 0)
            self.assertGreater(
                h200["phases"][phase]["throughput"]["sol"]["per_gpu"], 0)


class TestPeakSource(unittest.TestCase):
    def test_the_response_names_the_cell_and_an_unpriceable_card_refuses(self):
        from backend.inference.capacity import estimate_capacity
        from backend.inference.gpu_specs import get_gpu, peak_source
        config = _config("Qwen__Qwen3-8B")
        resolved = _resolved("Qwen__Qwen3-8B")
        estimate = estimate_capacity(
            config=config, resolved=resolved, gpu_sku="H100-SXM", num_gpus=1,
            avg_input_tokens=512, avg_output_tokens=64)
        self.assertEqual(estimate["peak_source"], "H100-SXM.bf16_tflops")
        self.assertEqual(peak_source(get_gpu("A100-80G-SXM"), "fp8"), "unknown")
        with self.assertRaisesRegex(ValueError, "H100-SXM lists no FP4 peak throughput"):
            # NVFP4 W4A4 runs its matmuls in FP4; H100 has no FP4 column.
            estimate_capacity(
                config=config, resolved=resolved, gpu_sku="H100-SXM", num_gpus=1,
                avg_input_tokens=512, avg_output_tokens=64,
                precision={"preset": "w_nvfp4_w4a4_kv_fp8"})
