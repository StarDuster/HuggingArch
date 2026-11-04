"""The per-rank FLOPs column of the per-op breakdown is a second WALK.

`estimate_phase` builds `flops_by_module` from the global forward walk, then
runs the same walk again on the sharded dims and pairs the two row-for-row.
It does not divide, and the tests below are the reasons why:

  * a scalar cannot answer for a leaf the cluster does not split — a
    replicated norm, an MLA latent norm and an MoE router each do their WHOLE
    work on every rank, and `flops / cluster_size` billed them at a fraction
    of it;
  * `cluster_size` folds in `dp_replicas`, which divides REQUESTS rather than
    work, and takes the MAX region group, so it is not even the right factor
    for the leaves the cluster does split.

The pairing is positional and gated on structure: the two walks are the same
graph at the same binding, so a row that fails to line up is an error, never
something to reconcile. `test_row_alignment_is_structural` is that gate.
"""
from __future__ import annotations

import sys
import unittest

from tests.snapshots import resolve_snapshot
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.parallel_config import ParallelConfig
from backend.inference.phases import (
    evaluate_phase_step, flops_fold,
)




def _resolved(model_name):
    return resolve_snapshot(model_name)


def _rows(model_name: str, parallel: ParallelConfig, phase: str = "prefill",
          num_gpus: int = 8):
    resolved = _resolved(model_name)
    if resolved is None:
        return None
    step, _bind = evaluate_phase_step(
        resolved, parallel,
        phase=phase, batch=1,
        avg_input_tokens=1024, avg_output_tokens=256,
    )
    return [r for r in flops_fold(step).rows if r.get("flops_full")]


# Models spanning the shapes that break a scalar: dense GQA, MLA + MoE (a
# latent norm and a router that shard by nothing), and a model whose KV head
# count sits below the degree.
MODELS = [
    "meta-llama__Llama-3.2-1B",
    "deepseek-ai__DeepSeek-V2-Lite",
    "Qwen__Qwen3-30B-A3B",
]


class TestIdentityAtTpOne(unittest.TestCase):
    """One rank holds the whole model, so per-rank IS full — every row."""

    def test_every_row_is_unchanged_at_degree_one(self):
        for m in MODELS:
            with self.subTest(model=m):
                rows = _rows(m, ParallelConfig(), num_gpus=1)
                bad = [(r["name"], r["flops_full"], r["flops_per_rank"])
                       for r in rows if r["flops_per_rank"] != r["flops_full"]]
                self.assertEqual(bad[:8], [],
                                 f"{m}: {len(bad)} rows moved at tp=1")


class TestNoRowShedsMoreThanItHas(unittest.TestCase):
    """A rank never does less than zero and never more than the whole.

    The upper bound is the one a divisor could not violate and a second walk
    can: it catches a per-rank walk that sharded an axis the global walk left
    whole (the `logit_positions` defect — the head was billed at one position
    per rank against `s_query` globally, a ratio of `tp × s_query`).
    """

    def test_per_rank_is_between_zero_and_full(self):
        for m in MODELS:
            with self.subTest(model=m):
                rows = _rows(m, ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                                               tp_moe_expert=2))
                bad = [(r["name"], r["flops_full"], r["flops_per_rank"])
                       for r in rows
                       if not 0 <= r["flops_per_rank"] <= r["flops_full"]]
                self.assertEqual(bad[:8], [], f"{m}: rows outside [0, full]")


class TestReplicatedWorkIsNotDivided(unittest.TestCase):
    """The leaves a scalar got wrong, named.

    DeepSeek-V2-Lite carries all three: the per-block norms, MLA's latent
    layernorms (`d_kv_lora` / `d_q_lora`, no head axis for a tp group to
    split), and the MoE router (its output is `n_experts`, a routing
    denominator no world divides). Every rank does each of them whole.
    """

    NAMES = ("input_layernorm", "post_attention_layernorm",
             "kv_a_layernorm", "q_a_layernorm", "gate")

    def test_norms_and_router_do_whole_work_on_every_rank(self):
        rows = _rows("deepseek-ai__DeepSeek-V2-Lite",
                     ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                                    tp_moe_expert=2))
        checked = 0
        for r in rows:
            if r["name"] not in self.NAMES:
                continue
            checked += 1
            self.assertEqual(
                r["flops_per_rank"], r["flops_full"],
                f"{r['name']}: replicated work was divided "
                f"({r['flops_full']} → {r['flops_per_rank']})")
        self.assertGreater(checked, 0, "no replicated rows found to check — "
                                       "the spec's leaf names moved")


class TestShardedWorkIsDivided(unittest.TestCase):
    """The other half of the claim: the column is not simply a copy.

    An attention output projection and an expert projection both carry an axis
    their world splits, so at degree 2 each rank does half. Asserted as an
    exact half rather than merely `< full`, because "smaller" is what a wrong
    divisor also produces.
    """

    def test_projections_halve_at_degree_two(self):
        rows = _rows("deepseek-ai__DeepSeek-V2-Lite",
                     ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                                    tp_moe_expert=2))
        checked = 0
        for r in rows:
            if r["name"] not in ("o_proj", "kv_b_proj"):
                continue
            checked += 1
            self.assertEqual(
                r["flops_per_rank"] * 2, r["flops_full"],
                f"{r['name']}: expected half the work per rank, got "
                f"{r['flops_full']} → {r['flops_per_rank']}")
        self.assertGreater(checked, 0, "no sharded rows found to check")


class TestRowAlignmentIsStructural(unittest.TestCase):
    """Every row carries a per-rank value, and it came from its own twin.

    A missing `flops_per_rank` would mean the attach silently declined — which
    is exactly how the earlier version failed, joining a POST-fusion per-rank
    tree onto these PRE-fusion rows and quietly dropping the ones fusion had
    rewritten.
    """

    def test_every_row_has_a_per_rank_value(self):
        for m in MODELS:
            with self.subTest(model=m):
                rows = _rows(m, ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                                               tp_moe_expert=2))
                missing = [r["name"] for r in rows if "flops_per_rank" not in r]
                self.assertEqual(missing[:8], [],
                                 f"{m}: {len(missing)} rows without a per-rank walk")


if __name__ == "__main__":
    unittest.main()
