"""Vocab parallelism — embedding / lm_head (Megatron / vLLM), active at tp>1.

Pins the three things the modeling adds on top of the vocab-axis shard that
already divides their weight/FLOPs:
  • the collectives — embed all_reduce([B,S,H]) + lm_head all_gather([B,S,V]),
  • lm_head's per-rank column-parallel output [B,S,V/tp] (and tied lm_head's
    FLOPs/act dividing by tp, the gap a tied head left un-sharded),
  • the embedding output staying replicated [B,S,H] (only its weight shards),
and the tp=1 no-op.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import shared_loader
from backend.arch.eval.dim import value_of
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze

SNAP_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"
# Llama-3.2-3B ties its lm_head (the un-sharded-tied-head case); DeepSeek-V3
# unties it and adds a nested MTP head.
_TIED = "meta-llama__Llama-3.2-3B.json"
_UNTIED = "deepseek-ai__DeepSeek-V3.json"


def _run(snap, tp, seq=512):
    resolved = shared_loader().match(json.loads((SNAP_DIR / snap).read_text())["config"],
                                     model_id=Path(snap).stem.replace("__", "/", 1))
    pc = ParallelConfig(tp_attention=tp, tp_dense_ffn=tp, tp_moe_expert=tp)
    return resolved, pc, analyze(resolved_spec=resolved, parallel=pc,
                                 batch_size=1, avg_input_tokens=seq, avg_output_tokens=0, phase="prefill", breakdown=False)


def _lm_head_row(r):
    return next((row for row in r.per_op_timing
                 if (row.get("aggregation_id") or "").endswith("lm_head")), None)


def _vocab_comm_ops(r):
    return sorted(row.get("op") for row in r.per_op_timing
                  if row.get("op") in ("all_reduce", "all_gather")
                  and "vocab_" in str(row.get("aggregation_id", "")))


def _shape(r, sid, *, full=False):
    for program, layer in (
        (r.program, r.full_shapes if full else r.shapes),
        (r.pre_program, r.pre_full_shapes if full else r.pre_shapes),
    ):
        for node in program.nodes():
            if node.aggregation_id == sid and node.uid in layer.values:
                return layer.values[node.uid]
    raise AssertionError(f"shape {sid!r} is missing")


def _dims(axes):
    return [int(value_of(axis)) for axis in axes] if axes is not None else None


def _is_replicated(prn, g) -> bool:
    """A leaf is replicated (former `sharded_axis_names == ()`) when its
    per-rank shape equals the global one on every axis. `shape_in` may be
    None on either side (e.g. a top-level head whose global node carries no
    input shape); normalize before comparing."""
    return (
        _dims(prn.out) == _dims(g.out)
        and [_dims(row) for row in prn.ins] == [_dims(row) for row in g.ins]
    )


class TestVocabParallel(unittest.TestCase):
    def setUp(self):
        for s in (_TIED, _UNTIED):
            if not (SNAP_DIR / s).exists():
                self.skipTest(f"{s} absent")

    def test_collectives_appear_at_tp_gt_1(self):
        for snap in (_TIED, _UNTIED):
            _, _, r1 = _run(snap, 1)
            _, _, r8 = _run(snap, 8)
            self.assertEqual(_vocab_comm_ops(r1), [], f"{snap}: comm at tp=1")
            self.assertEqual(_vocab_comm_ops(r8), ["all_gather", "all_reduce"],
                             f"{snap}: missing embed AR / lm_head AG at tp=8")

    def test_lm_head_flops_and_output_shard_by_tp(self):
        # Holds for the TIED head too — the gap where a tied lm_head kept full
        # FLOPs/act because it is not a spec leaf.
        for snap in (_TIED, _UNTIED):
            _, _, r1 = _run(snap, 1)
            _, _, r8 = _run(snap, 8)
            f1 = _lm_head_row(r1)["flops_per_rank"]
            f8 = _lm_head_row(r8)["flops_per_rank"]
            self.assertAlmostEqual(f1 / f8, 8.0, places=3, msg=f"{snap}: lm_head flops")
            sh = _shape(r8, "lm_head")
            sh1 = _shape(r1, "lm_head")
            self.assertEqual(_dims(sh.out)[-1] * 8, _dims(sh1.out)[-1],
                             f"{snap}: lm_head output not V/tp")

    def test_embedding_output_replicated(self):
        # Vocab-parallel embedding is row-parallel on V: weight shards, but the
        # masked lookup is all_reduced back to a replicated [B,S,H].
        for snap in (_TIED, _UNTIED):
            _, _, r8 = _run(snap, 8)
            e1 = _shape(_run(snap, 1)[2], "embed_tokens")
            e8 = _shape(r8, "embed_tokens")
            self.assertEqual(_dims(e8.out), _dims(e1.out),
                             f"{snap}: embedding output should stay replicated")
            # sharded_axis_names == () ⟺ per-rank shape == global shape.
            g8 = _shape(r8, "embed_tokens", full=True)
            self.assertTrue(_is_replicated(e8, g8),
                            f"{snap}: embedding output must be replicated (== global)")

    def test_lm_head_runs_on_sample_positions_not_full_seq(self):
        # vLLM/sglang select logits_indices before the projection: the last
        # token per sequence in prefill (1 position), so the lm_head output is
        # [B, 1, V] — not [B, S, V]. Sized in the engine, so total_flops agrees.
        _, _, r = _run(_UNTIED, 1, seq=512)
        lh = _shape(r, "lm_head")
        self.assertEqual(_dims(lh.out)[-2], 1, "prefill lm_head should sample 1 position, not S")

    def test_spec_decode_verifies_multiple_positions(self):
        # Speculative decode verifies the whole proposed window per sequence.
        resolved = shared_loader().match(
            json.loads((SNAP_DIR / _UNTIED).read_text())["config"],
            model_id=Path(_UNTIED).stem.replace("__", "/", 1))
        r = analyze(resolved_spec=resolved, parallel=ParallelConfig(tp_attention=1),
                    batch_size=1, avg_input_tokens=512, avg_output_tokens=0, phase="decode", s_verify=4, breakdown=False)
        # The per-rank forward-transform tree sizes lm_head over the verified
        # window (`s_verify=4`), the SAME as the global engine run — both ask
        # `step_eval.sampled_logit_positions` rather than deciding for themselves.
        # At tp=1 per-rank == global (the identity guard).
        lh = _shape(r, "lm_head")
        self.assertEqual(_dims(lh.out)[-2], 4, "spec decode should verify K+1 positions")
        self.assertEqual(_dims(lh.out), _dims(_shape(r, "lm_head", full=True).out),
                         "tp=1: per-rank lm_head must equal global (spec-decode positions)")

    def test_tp1_is_noop(self):
        # At tp=1 the lm_head output is full vocab and no vocab comm exists.
        _, _, r1 = _run(_TIED, 1)
        sh = _shape(r1, "lm_head")
        # tp=1 → per-rank shape byte-equal to global (sharded_axis_names == ()).
        self.assertTrue(_is_replicated(sh, _shape(r1, "lm_head", full=True)))
        self.assertEqual(_vocab_comm_ops(r1), [])


if __name__ == "__main__":
    unittest.main()
