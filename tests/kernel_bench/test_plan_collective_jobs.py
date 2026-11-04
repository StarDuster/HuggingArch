"""A collective row becomes a job, and the job bills exactly the row.

The planner exports `count_per_step`; the read side multiplies it back up
(`calibration_overlay.point_keys`: `billed = launches x cadence`) and prices
the launch by the job's own numerators. So the invariant that has to
hold for every job is

    launches x job_per_launch == row_per_rank

for FLOPs and for bytes. For a collective the payload is the thing to check
on top of that: the cell says how many elements one launch moves, and that
has to be the payload the walk priced the row by.

Checked on gpt-oss-20b at TP8 because it is a seven-second walk that emits
both a per-layer all-reduce and a vocabulary all-gather, beside the fused
all-reduce+RMSNorm that was already measured and must not move.
"""
from __future__ import annotations

import json
import sys
import unittest
from fractions import Fraction
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.dtypes import dtype_bytes
from backend.arch.ops import OP_REGISTRY
from backend.inference import pipeline
from backend.inference.benchmark_identity import RowIdentity
from backend.inference.parallel_config import ParallelConfig
from kernel_bench import plan as planmod
from tests.snapshots import shared_loader

MODEL = "openai/gpt-oss-20b"
SNAPSHOT = (PROJECT_ROOT / "backend" / "arch" / "snapshots"
            / "openai__gpt-oss-20b.json")
BATCH = 8


def _is_collective(op: str) -> bool:
    """One declaration, asked — not a list of collective names here."""
    spec = OP_REGISTRY.get(op)
    return spec is not None and spec.is_comm_op


class TestCollectiveJobsBillTheirRow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads(SNAPSHOT.read_text())["config"]
        resolved = shared_loader().match(config, model_id=MODEL)
        walked = []
        original = pipeline.analyze

        def spy(*args, **kwargs):
            result = original(*args, **kwargs)
            walked.append(result)
            return result

        pipeline.analyze = spy
        try:
            cls.jobs, cls.dropped = planmod.build_jobs(avg_output_tokens=0, 
                model_id=MODEL, batch_size=BATCH, seq_len=4096,
                phase="decode", resolved=resolved,
                parallel=ParallelConfig(tp_attention=8, tp_dense_ffn=8,
                                        tp_moe_expert=8))
        finally:
            pipeline.analyze = original
        cls.rows, cls.identities = {}, {}
        for result in walked:
            identity = RowIdentity(result)
            for row in result.per_op_timing:
                sid = str(row.get("aggregation_id") or "")
                cls.rows.setdefault(sid, row)
                cls.identities.setdefault(sid, identity)

    def _launches(self, job):
        """The row and what the read side bills: the step's launches of the
        job's cell times the share of a launch one call is
        (`RowIdentity.launches` x `cadence`, as `PointRow.billed` is)."""
        row = self.rows[job["node_spec_id"]]
        identity = self.identities[job["node_spec_id"]]
        launches = max(identity.launches(row), 1)
        return row, Fraction(launches) * identity.cadence(row)

    def test_every_collective_row_became_a_job(self):
        """At the batch the walk ran at, no collective is silently missing.
        This is the gap: before the signatures existed, every one of these
        rows was dropped and communication time was never calibrated."""
        rows = {sid for sid, row in self.rows.items()
                if _is_collective(str(row.get("op") or ""))}
        self.assertTrue(rows, "gpt-oss at TP8 grafts collectives")
        jobs = {job["node_spec_id"] for job in self.jobs
                if _is_collective(job["op"])}
        self.assertEqual(rows, jobs)
        self.assertEqual(len(jobs), len([j for j in self.jobs
                                         if _is_collective(j["op"])]),
                         "a collective is one job, not one per paged block")

    def test_launches_times_the_job_is_the_row(self):
        """The invariant, for every collective job and for the fused
        collective beside them.

        Scoped to the collectives because that is what this file is about.
        It holds for every job in every cell now that the planner has one
        source for the numerators — the recompute that used to break it on
        `gemm` / `rmsnorm` / `embedding` (the measurement-plan audit's
        F9/F11) is gone, and the cell-wide version of this assertion is
        `test_kernel_bench.py::test_job_numerators_are_the_walk_divided_by_the_launch_count`.
        """
        checked = 0
        for job in self.jobs:
            if not (_is_collective(job["op"])
                    or job["signature"] == "allreduce_fused_rmsnorm"):
                continue
            checked += 1
            row, launches = self._launches(job)
            with self.subTest(sig=job["signature"], sid=job["node_spec_id"]):
                for key in ("flops_per_rank", "access_bytes_per_rank"):
                    billed = job[key]
                    if billed is None:
                        continue
                    self.assertEqual(
                        launches * Fraction(str(billed)),
                        Fraction(str(row[key])),
                        f"{key}: {launches} launches of {billed} is not the "
                        f"row's {row[key]}")
        self.assertTrue(checked, "no collective job to check")

    def test_the_cell_moves_the_payload_the_row_priced(self):
        """The payload check, against the GRAPH's width and not the job's own.

        `n_comm_elems` times the width the walk priced the payload at
        (`comm_payload_dtype_bytes`) must be the tensor bytes it billed. The
        width is checked against the job's dtype environment too, in the
        same breath: a collective whose payload is not the dtype the harness
        will build the buffer at — DCP's fp32 log-sum-exp all-gather beside
        bf16 activations — must fail here, at plan time, rather than ship a
        half-width buffer to a provider. Sourcing the expected width from
        the job's own env instead would only prove the job agrees with
        itself.
        """
        seen = 0
        for job in self.jobs:
            if not _is_collective(job["op"]):
                self.assertNotIn("n_comm_elems", job["dims"])
                self.assertNotIn("n_comm_stride", job["dims"])
                continue
            row = self.rows[job["node_spec_id"]]
            width = float(row["comm_payload_dtype_bytes"])
            self.assertEqual(job["dims"]["n_comm_elems"] * width,
                             float(row["comm_tensor_bytes"]))
            self.assertEqual(width,
                             dtype_bytes(job["dtype_env"]["activation"]),
                             "the payload is not the width this job's buffer "
                             "would be built at")
            self.assertEqual(job["dims"]["n_comm_ranks"], row["comm_world"])
            self.assertEqual(job["dims"]["n_comm_stride"],
                             row["comm_base_stride"])
            seen += 1
        self.assertTrue(seen)

    def test_the_fused_collective_still_carries_its_group(self):
        """Untouched: the launch shape is one call, the group is read off
        the row, the count is the layer count, and batch stays out of the
        count."""
        fused = [job for job in self.jobs
                 if job["signature"] == "allreduce_fused_rmsnorm"]
        self.assertTrue(fused)
        for job in fused:
            row = self.rows[job["node_spec_id"]]
            self.assertEqual(job["dims"]["n_comm_ranks"], row["comm_world"])
            self.assertEqual(job["dims"]["B"], BATCH)
            self.assertEqual(job["count_per_step"],
                             int(row["count_per_step"]))


if __name__ == "__main__":
    unittest.main()
