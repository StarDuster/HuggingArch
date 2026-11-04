"""A job is one rank's launch, so under pipeline parallelism it is a stage's.

`plan` used to walk the whole model whatever the deployment was, so at pp>1
its rows were byte-identical to pp=1 and every job claimed the model's own
layer count. No rank runs that: DeepSeek-V3's 61 layers at pp=4 are held
15/15/16/15, and a per-layer kernel was billed 58 times per rank per step
where the busiest rank runs 16. It was silent rather than a lookup miss,
because the parallel signature does carry `pp`, so those rows got their own
key and were found — with a count no rank has, off by a factor that grows
with pp.

The fix asks the walk for the rank: `analyze(stage=s)` already projects one
stage (`layout.stages.StageView.rows`), so every stage is walked and the
jobs are folded back into one per launch. What this file pins:

  * a stage's count is the layers IT holds, and the stages' counts sum to
    the whole model's (`stage_layer_ranges`, uneven splits included);
  * the measurement is unchanged by the split — same keys, same per-launch
    numerators — which is why one job stands for every stage;
  * `launches x the job's per-launch FLOPs` is the FLOPs of the row of the
    rank the job names;
  * pp=1 walks once and folds nothing.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import resolve_snapshot

from backend.arch.layout.parallel_config import stage_layer_ranges
from backend.inference.benchmark_identity import RowIdentity
from backend.inference.framework_bindings import DEFAULT_FRAMEWORK, get_binding
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from kernel_bench.plan import (
    _one_job_per_launch,
    build_jobs,
)

#: Walks two snapshots at four pipeline degrees — a per-model test, which
#: this repository does not run on every push (CLAUDE.md §5). Run it with
#: `-m regression` when changing the plan or the stage placement.
pytestmark = pytest.mark.regression

#: 28 layers, so `stage_layer_ranges(28, 8)` is the uneven 3/3/3/4/4/4/4/3
#: the dim registry documents — the heaviest stage is 4 and the first and
#: last stages (which also carry the embedding and the head) are 3.
SMALL = "meta-llama__Llama-3.2-3B"
MODEL_ID = "meta-llama/Llama-3.2-3B"
SMALL_LAYERS = 28

SEQ = 2048
BATCH = 8


def _resolved(name=SMALL):
    return resolve_snapshot(name)


def _jobs(resolved, *, pp: int, phase: str = "decode"):
    jobs, _dropped = build_jobs(avg_output_tokens=0,
        model_id=MODEL_ID, batch_size=BATCH, seq_len=SEQ, phase=phase,
        parallel=(ParallelConfig.from_dict({"pp": pp}) if pp > 1
                  else ParallelConfig()),
        resolved=resolved)
    return jobs


def _rows(resolved, *, pp: int, stage, phase: str = "decode", knobs=None):
    """One stage's per-op rows, walked the way the plan walks them."""
    deployment = dict(knobs or {})
    if pp > 1:
        deployment["pp"] = pp
    result = analyze(
        resolved,
        parallel=(ParallelConfig.from_dict(deployment) if deployment
                  else ParallelConfig()),
        framework_binding=get_binding(DEFAULT_FRAMEWORK),
        batch_size=BATCH,
        avg_input_tokens=SEQ, avg_output_tokens=0,
        phase=phase,
        stage=stage, include_aux_timing=True, breakdown=False)
    return result


def _by_key(jobs):
    """Jobs by the measurement key plus the kernel parameters, which is what
    one lake row is (the paged-KV fan-out writes one job per block size)."""
    out = {}
    for job in jobs:
        key = (job["signature"], job["node_spec_id"],
               tuple(sorted(job["dims"].items())),
               tuple(sorted(job["dtype_env"].items())),
               tuple(sorted((k, str(v)) for k, v
                            in job["kernel_params"].items())))
        assert key not in out, f"two jobs under one lake row: {key}"
        out[key] = job
    return out


class TestAStageBillsItsOwnLayers(unittest.TestCase):
    """The count a job publishes is the layers one rank holds."""

    @classmethod
    def setUpClass(cls):
        cls.resolved = _resolved()
        cls.at = {pp: _by_key(_jobs(cls.resolved, pp=pp)) for pp in (1, 4, 8)}

    def test_a_per_layer_kernel_counts_the_heaviest_stage(self):
        # Every stage holds the same templates here (one dense block), so the
        # count a job publishes is the heaviest stage's share of the 28.
        for pp, expected in ((1, SMALL_LAYERS), (4, 7), (8, 4)):
            held = max(len(r) for r in stage_layer_ranges(SMALL_LAYERS, pp))
            self.assertEqual(
                expected, held,
                f"pp={pp}: the partition itself changed; the expectation "
                f"below is written against `stage_layer_ranges`")
            per_layer = {key: job for key, job in self.at[pp].items()
                         if job["count_per_step"] > 1}
            self.assertTrue(per_layer, f"pp={pp}: no per-layer job at all")
            for key, job in per_layer.items():
                self.assertEqual(
                    held, job["count_per_step"],
                    f"pp={pp}: {key[0]} [{key[1]}] bills "
                    f"{job['count_per_step']} launches per rank per step, "
                    f"but the busiest stage holds {held} layers")

    def test_the_split_adds_the_hop_and_moves_nothing_else(self):
        # Splitting the layers across ranks does not change what any kernel
        # is measured at: same keys, same per-launch numerators. That is why
        # ONE job can stand for every stage — a measurement key names no
        # stage. The one thing a pipeline ADDS is its own hop, which has no
        # pp=1 counterpart because there is no cut to send across.
        base = self.at[1]
        for pp in (4, 8):
            added = {key: self.at[pp][key] for key in set(self.at[pp]) - set(base)}
            self.assertEqual(
                [], sorted(key[0] for key in added if key[0] != "send_recv"),
                f"pp={pp}: the split added a measurement that is not a hop")
            self.assertEqual(
                [], sorted(set(base) - set(self.at[pp])),
                f"pp={pp}: the split dropped a measurement")
            for key, job in added.items():
                # One hop per rank per step, whatever the stage count: a
                # rank sends once across its own exit cut. Its world is the
                # pipeline group the cut spans.
                self.assertEqual(
                    1, job["count_per_step"],
                    f"pp={pp}: {key[1]} sends {job['count_per_step']} times "
                    f"per rank per step")
                self.assertEqual(pp, job["dims"]["n_comm_ranks"],
                                 f"pp={pp}: the hop's world is not the "
                                 f"pipeline group")
            for key, job in self.at[pp].items():
                if key in added:
                    continue
                for field in ("flops_per_rank", "access_bytes_per_rank",
                              "shape_in", "shape_out", "shape_weight",
                              "dims", "dtype_env"):
                    want, got = base[key][field], job[field]
                    if isinstance(want, float) and isinstance(got, float):
                        # FLOPs are rounded once, after scaling by the count
                        # (`phases.compute_per_op_timing`), so a per-launch
                        # numerator read off a 4-layer stage and off the
                        # whole 28 differ by under half a FLOP.
                        self.assertLessEqual(
                            abs(want - got), 1.0,
                            f"pp={pp}: {key[0]} [{key[1]}] {field} moved "
                            f"{want} -> {got}, more than rounding")
                    else:
                        self.assertEqual(
                            want, got,
                            f"pp={pp}: {key[0]} [{key[1]}] {field} moved")


class TestTheStagesAccountForTheWholeModel(unittest.TestCase):
    """Σ over stages is the whole model, stage by stage and row by row."""

    @classmethod
    def setUpClass(cls):
        cls.resolved = _resolved()

    def _counts(self, pp, stage):
        rows = _rows(self.resolved, pp=pp, stage=stage).per_op_timing
        out = {}
        for row in rows:
            key = row["aggregation_id"]
            self.assertNotIn(key, out, "two rows under one aggregation id")
            out[key] = int(row["count_per_step"])
        return out

    def test_every_rows_counts_sum_over_the_stages(self):
        whole = self._counts(1, None)
        for pp in (2, 4, 8):
            per_stage = [self._counts(pp, s) for s in range(pp)]
            for path, total in whole.items():
                summed = sum(stage.get(path, 0) for stage in per_stage)
                self.assertEqual(
                    total, summed,
                    f"pp={pp}: {path} runs {total} times per step over the "
                    f"model but {summed} over its stages")

    def test_a_stages_count_is_the_layers_it_holds(self):
        # The rows the whole model runs once per layer — named by what they
        # count, not by this spec's template name.
        whole = self._counts(1, None)
        per_layer = {key for key, n in whole.items() if n == SMALL_LAYERS}
        self.assertTrue(per_layer, "no row runs once per layer")
        for pp in (2, 4, 8):
            held = [len(r) for r in stage_layer_ranges(SMALL_LAYERS, pp)]
            for stage, expected in enumerate(held):
                counts = self._counts(pp, stage)
                for key in per_layer:
                    self.assertEqual(
                        expected, counts[key],
                        f"pp={pp} stage {stage} holds {expected} of "
                        f"{SMALL_LAYERS} layers, but {key} runs "
                        f"{counts[key]} times there")


class TestLaunchesTimesTheJobIsTheRanksRow(unittest.TestCase):
    """`launches x the job's per-launch numerator == the row's` — of the row
    of the rank the job names, which is the point: before this the two sides
    were different ranks and the identity held against neither.

    Every job is checked, and both numerators. This used to skip whichever
    signatures the plan recomputed from the benched dims
    (`_benched_roofline`, a projection with its own account) — that helper is
    gone, so there is nothing left to skip: FLOPs and bytes alike are the
    walk's row over the launches the rank makes of it.
    """

    def test_the_identity_holds_against_the_stage_that_runs_it(self):
        resolved = _resolved()
        pp = 8
        jobs = _jobs(resolved, pp=pp)
        walked = [_rows(resolved, pp=pp, stage=s) for s in range(pp)]
        # Per row, the share of a launch one call is (`RowIdentity.cadence`,
        # what `PointRow.billed` multiplies the launches by).
        rows = [
            (row, float(RowIdentity(result).cadence(row)))
            for result in walked for row in result.per_op_timing]
        checked = 0
        for job in jobs:
            found = [
                (row, events) for row, events in rows
                if row["aggregation_id"] == job["node_spec_id"]
                and int(row["count_per_step"]) == job["count_per_step"]
            ]
            self.assertTrue(
                found,
                f"{job['signature']} [{job['node_spec_id']}] bills "
                f"{job['count_per_step']} launches per rank per step, and no "
                f"stage's row runs it that many times")
            row, cadence = found[0]
            launches = job["count_per_step"] * cadence
            for field in ("flops_per_rank", "access_bytes_per_rank"):
                if job[field] is None:
                    self.assertIsNone(row.get(field),
                                      f"{job['node_spec_id']}: {field}")
                    continue
                self.assertAlmostEqual(
                    float(row[field]),
                    launches * job[field],
                    delta=max(1.0, 1e-9 * float(row[field])),
                    msg=(f"{job['signature']} [{job['node_spec_id']}]: "
                         f"{launches} launches x {job[field]} is not the "
                         f"{row[field]} its rank's row runs ({field})"))
            checked += 1
        self.assertGreater(checked, 0, "no job to check")


class TestTwoTemplatesAreTwoAccounts(unittest.TestCase):
    """The audit's own case, and CLAUDE.md §2.5's: DeepSeek-V3's 61 layers
    are 3 dense-FFN then 58 MoE, and `stage_layer_ranges(61, 4)` deals them
    15/15/16/15. The three dense layers all land on stage 0; the MoE
    template splits 12/15/16/15. A job of each template therefore bills a
    different number, and neither is 61 or 58 — the assignment is not
    flattened into one per-layer count.
    """

    def test_each_template_bills_its_own_busiest_stage(self):
        resolved = _resolved("deepseek-ai__DeepSeek-V3")
        jobs, _dropped = build_jobs(avg_output_tokens=0,
            model_id="deepseek-ai/DeepSeek-V3", batch_size=BATCH,
            seq_len=SEQ, phase="decode", resolved=resolved,
            parallel=ParallelConfig.from_dict(
                {"pp": 4, "tp_attention": 8, "ep_moe": 8}))
        self.assertEqual([15, 15, 16, 15],
                         [len(r) for r in stage_layer_ranges(61, 4)])
        counts = {}
        for job in jobs:
            counts.setdefault(job["count_per_step"], []).append(job)
        # The MoE template (58 layers, busiest stage 16) and the dense one
        # (3 layers, all on stage 0). Nothing bills 58 or 61.
        self.assertEqual({1, 3, 16}, set(counts),
                         "a job bills a layer count no rank runs")
        for count in (3, 16):
            self.assertTrue(counts[count],
                            f"no job bills {count} launches per step")


class TestCollectivesFoldLikeAnythingElse(unittest.TestCase):
    """A collective is a launch, so the stage fold is its fold too.

    Bare collectives became measurable alongside this change (F6), and they
    need the fold for two different reasons:

      * a collective INSIDE a layer runs once per layer, so at pp>1 a stage
        runs it as often as it holds layers — DeepSeek-V3's per-layer MoE
        all-reduce and its fused attention all-reduce bill 16 at pp=4, not
        58. Without the per-stage walk the new collective jobs would have
        shipped carrying F7 themselves;
      * the pipeline's own hop is one row per cut, and the cuts are the same
        launch — same payload, same world, same cell. pp-1 stage rows fold
        to the one measurement, at the one-per-step a rank actually sends.

    The fold's identity for them is `RowIdentity.of`'s own cell
    (`_collective_identity`: `n_comm_elems` / `n_comm_ranks` /
    `n_comm_stride`), carried whole — not a second reading of it.
    """

    @classmethod
    def setUpClass(cls):
        cls.resolved = _resolved("deepseek-ai__DeepSeek-V3")
        cls.pp = 4
        cls.jobs, _dropped = build_jobs(avg_output_tokens=0,
            model_id="deepseek-ai/DeepSeek-V3", batch_size=BATCH,
            seq_len=SEQ, phase="decode", resolved=cls.resolved,
            parallel=ParallelConfig.from_dict(
                {"pp": cls.pp, "tp_attention": 8, "ep_moe": 8}))

    def test_a_collective_inside_a_layer_counts_by_stage(self):
        # Per template, exactly as a GEMM does: the 58 MoE layers' busiest
        # stage holds 16 of them, the 3 dense-FFN layers all sit on stage 0.
        # A collective is no more flattened across an assignment than a
        # matmul is (CLAUDE.md §2.5).
        self.assertEqual([15, 15, 16, 15],
                         [len(r) for r in stage_layer_ranges(61, self.pp)])
        counts = sorted({job["count_per_step"] for job in self.jobs
                         if job["count_per_step"] > 1
                         and job["signature"] in ("all_reduce",
                                                  "allreduce_fused_rmsnorm")})
        self.assertEqual(
            [3, 16], counts,
            f"per-layer collectives bill {counts} collectives per rank per "
            f"step; the two templates' busiest stages hold 3 and 16 layers "
            f"(and the whole model's 58 and 3 is what no rank runs)")

    def test_every_cut_is_one_hop_measurement(self):
        # Each stage but the last sends across its own exit cut, and every
        # one of those rows is the same launch.
        sent = []
        for stage in range(self.pp):
            rows = [row for row in _rows(self.resolved, pp=self.pp,
                                         stage=stage,
                                         knobs={"tp_attention": 8,
                                                "ep_moe": 8}).per_op_timing
                    if row["op"] == "send_recv"]
            sent.append([(int(row["count_per_step"]), row["comm_world"],
                          tuple(row["comm_payload_shape"])) for row in rows])
        self.assertEqual([], sent[-1], "the last stage has an exit cut")
        for stage, rows in enumerate(sent[:-1]):
            self.assertEqual(
                [(1, self.pp, sent[0][0][2])], rows,
                f"stage {stage} does not send one hop of the cut's payload "
                f"over the pipeline group")
        hops = [job for job in self.jobs if job["signature"] == "send_recv"]
        self.assertEqual(
            1, len(hops),
            f"{self.pp - 1} cuts sending one launch became {len(hops)} "
            f"measurements")
        self.assertEqual(1, hops[0]["count_per_step"])
        self.assertEqual(self.pp, hops[0]["dims"]["n_comm_ranks"])

    def test_the_fold_keys_a_collective_by_the_rows_own_cell(self):
        # The identity the fold folds on IS the measurement key's bytes, so
        # a collective job's `dims` are the cell `RowIdentity` published and
        # nothing else: no shape axes a collective has no business carrying.
        for job in self.jobs:
            if not job["signature"].startswith(("all_", "reduce_", "send_")):
                continue
            self.assertEqual(
                {"n_comm_elems", "n_comm_ranks", "n_comm_stride"},
                set(job["dims"]),
                f"{job['signature']} [{job['node_spec_id']}] is keyed by "
                f"something other than the row's collective cell")


class TestWithoutAPipelineNothingChanges(unittest.TestCase):
    """pp=1 is one rank, walked once, folded not at all — and the whole
    export is pinned so that stays a gate and not a claim.

    `PIN` is what `main` exported for this cell before the per-stage walk
    landed, confirmed by dumping both revisions and diffing (and
    independently by review). It is the complete export, jobs and dropped
    list, in order — so anything that moves a shape, a numerator, a dtype
    env or the dropped set shows up here with the field named.

    It therefore also moves for a legitimate reason: a change to what the
    plan exports a numerator FROM, or to this model's spec, re-pins it.
    Re-pin with `python <this file> --repin` and say in the commit what moved
    and why. It was last re-pinned when every job began to state its leaf's
    variant facts beside its key: all 13 jobs gained `variant: {}`, and no
    other field, shape, numerator, dtype environment or dropped entry moved.
    Before that, no dtype slot borrowed a
    model-level value any more: a job that streams no weight omits `weight`
    (it was the activation's), one that touches no cache omits `kv` (it was
    the step's widest cache), which moved `dtype_env` and `physical_key` on
    all 13 jobs and nothing else. Before that, `metadata` stopped borrowing
    the model's scale dtype, moving the same two fields.
    """

    PIN = PROJECT_ROOT / "tests" / "golden" / "kernel_bench_plan_pp1.json"

    @staticmethod
    def export(resolved):
        """The cell's complete export, through one JSON round trip so a
        tuple and a list of the same numbers compare equal."""
        jobs, dropped = build_jobs(avg_output_tokens=0,
            model_id=MODEL_ID, batch_size=BATCH, seq_len=SEQ, phase="decode",
            parallel=ParallelConfig(), resolved=resolved)
        return json.loads(json.dumps(
            {"jobs": jobs, "dropped": sorted(dropped)},
            sort_keys=True, default=str))

    def test_the_whole_pp1_export_is_what_it_was(self):
        now = self.export(_resolved())
        pinned = json.loads(self.PIN.read_text())
        self.assertEqual(pinned["dropped"], now["dropped"],
                         "the dropped list moved")
        self.assertEqual(
            len(pinned["jobs"]), len(now["jobs"]),
            f"{len(pinned['jobs'])} jobs were pinned, {len(now['jobs'])} "
            f"exported")
        for index, (was, is_now) in enumerate(zip(pinned["jobs"],
                                                  now["jobs"], strict=True)):
            for field in sorted(set(was) | set(is_now)):
                self.assertEqual(
                    was.get(field), is_now.get(field),
                    f"job {index} ({is_now.get('op')} "
                    f"[{is_now.get('node_spec_id')}]): {field} moved")

    def test_one_rank_is_returned_as_walked(self):
        # The fold copies nothing and reorders nothing on a single rank: the
        # jobs come back as the objects the walk's own pass built.
        job = {"signature": "gemm", "count_per_step": 3}
        self.assertEqual([job], _one_job_per_launch([[(("k",), job)]]))
        self.assertIs(job, _one_job_per_launch([[(("k",), job)]])[0])

    def test_the_count_is_the_walks_own(self):
        resolved = _resolved()
        whole = {row["aggregation_id"]: int(row["count_per_step"])
                 for row in _rows(resolved, pp=1, stage=None).per_op_timing}
        for job in _jobs(resolved, pp=1):
            self.assertEqual(
                whole[job["node_spec_id"]], job["count_per_step"],
                f"{job['signature']} [{job['node_spec_id']}]: the job's "
                f"count is not the row's")

    def test_no_row_is_skipped_for_running_nowhere(self):
        # The one behaviour the per-stage walk adds to the pp=1 path is the
        # skip of a row the rank runs zero times. Without a pipeline no row
        # is one, so the skip cannot fire and the export above is reachable
        # for every measurable row.
        for row in _rows(_resolved(), pp=1, stage=None).per_op_timing:
            self.assertGreaterEqual(
                int(row["count_per_step"]), 1,
                f"{row['aggregation_id']} runs 0 times without a pipeline, "
                f"so the plan would now skip a row it used to export")


def _repin() -> None:
    """Rewrite `PIN` from this tree (`--repin`)."""
    path = TestWithoutAPipelineNothingChanges.PIN
    payload = TestWithoutAPipelineNothingChanges.export(_resolved())
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(payload['jobs'])} jobs → {path}")


if __name__ == "__main__":
    if "--repin" in sys.argv:
        _repin()
    else:
        unittest.main()
