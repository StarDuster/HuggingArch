"""MoE all-to-all gather: the per-expert effective batch must include the
DP-attention gather factor.

Under DP-attention + EP, each attention-DP rank runs its own `B·S` tokens
through attention, then the expert-parallel all-to-all pools tokens from ALL
`dp_attention` streams onto every rank's experts. So a rank's experts see
`B·S·dp_attention` tokens, and the per-expert effective batch is

    M = ⌈B·S·K·dp_attention / E⌉

Without the `dp_attention` factor the MoE decode/prefill compute (and its
activation bytes) is under-counted by `dp_attention`, which pushes the
compute-bound knee out by the same factor. Logical callers (validators /
kv_topology / golden shape tests) leave `dp_attention` unset → factor 1 → the
plain per-token model is unchanged.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.projection import project_model
from tests.snapshots import shared_loader

SNAPSHOT_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _resolve(name):
    cfg = json.loads((SNAPSHOT_DIR / name).read_text())["config"]
    return shared_loader().match(cfg, model_id=Path(name).stem.replace("__", "/", 1))


class TestMoEDpGather(unittest.TestCase):
    def setUp(self):
        # DeepSeek-V3: E=256 routed experts, K=8 active per token.
        self.r = _resolve("deepseek-ai__DeepSeek-V3.json")

    def _M(self, B, dp_attention=None):
        mr = project_model(
            self.r, batch=B, input_tokens=1, logit_positions=1,
            dp_attention=dp_attention or 1)

        def find(nodes):
            # `M` is bound where it is derived: on the permute, from the
            # token tensor that arrived there.
            for n in nodes:
                if n.children:
                    hit = find(n.children)
                    if hit is not None:
                        return hit
                elif n.op == "permute":
                    return n
            return None

        for _t, _c, block in mr.block_results:
            hit = find(block.nodes)
            if hit is not None:
                return (hit.op_dims or {}).get("M")
        raise AssertionError("no permute found")

    def test_default_is_logical_no_gather(self):
        """No `dp_attention` set (logical callers) → M = B·K/E exactly."""
        # E=256, K=8: M = 1000*8/256 = 31.25 — the exact average; no
        # ceiling, so the bank total stays E*M = B*K precisely.
        self.assertEqual(float(self._M(1000)), 31.25)
        self.assertEqual(float(self._M(1000, dp_attention=1)), 31.25)

    def test_gather_scales_M_with_dp_attention(self):
        """M carries the ×dp_attention all-to-all gather."""
        # 1000*8*8/256 = 250 exactly.
        self.assertEqual(float(self._M(1000, dp_attention=8)), 250)
        # EXACTLY dp_attention× the logical M — no ceil rounding to forgive.
        m1, m8 = self._M(1000, dp_attention=1), self._M(1000, dp_attention=8)
        self.assertEqual(float(m8) / float(m1), 8.0)

    def test_decode_moe_compute_rises_with_dp_attention(self):
        """End-to-end: a DP-attention + EP MoE decode phase bills MORE expert
        compute than the same per-GPU batch without DP-attention — the gather
        makes each rank's experts process `dp_attention`× the tokens."""
        from backend.inference.capacity import estimate_capacity
        from backend.inference.parallel_config import ParallelConfig
        cfg = json.loads(
            (SNAPSHOT_DIR / "deepseek-ai__DeepSeek-V3.json").read_text())["config"]

        resolved = _resolve("deepseek-ai__DeepSeek-V3.json")

        def tc(dp):
            r = estimate_capacity(
                 config=cfg, resolved=resolved, gpu_sku="B300", num_gpus=8,
                avg_input_tokens=1024, avg_output_tokens=256,
                parallel=ParallelConfig(ep_moe=8, dp_attention=dp),
                decode_batch_size=1000, selection_mode="theoretical_only")
            return (r["phases"]["decode"] or {}).get("t_compute_s", 0.0)

        # dp_attention=8 gathers 8× the tokens onto each rank's experts, so
        # decode compute is materially higher than the (buggy) per-GPU count.
        self.assertGreater(tc(8), tc(1) * 1.3,
                           "decode MoE compute must rise with the dp_attention gather")


if __name__ == "__main__":
    unittest.main()
