"""Who is inside a MoE dispatch zone, as opposed to where one starts.

`SpecIRNode.in_dispatch_zone` is per-graph and deliberately non-cumulative: it
marks the node where the zone is ENTERED, because that is where the evaluator
narrows `B`/`s_ctx` to one token per expert row and installs the routing ratio
for the whole subtree. A descendant carrying the same flag would apply both
again, so the flag is correct exactly as it is.

What was wrong was asking it leaf-by-leaf. DeepSeek-V3 has 193 IR nodes and 2
of them carry the flag — both the `swiglu_ffn` component — while every leaf
that performs a routed GEMM sits one level below and reads False. A bank
packed into one custom op (gpt-oss `mxfp4_fused_linear`) is its own entry node
AND its own leaf, so a leaf-level test found 12 members there and looked like
it worked. It did not: the straggler factor keyed off that set, which means
`ep_load_factor` had no leaf to scale on any wrapper-style bank — the whole
imbalance model was unreachable on almost the entire corpus, silently, while
reporting balanced-routing numbers as if they were the estimate.

So this pins the accumulated form, on both bank shapes and against a dense
control, plus the two properties that make it worth having: that the set drives
the arithmetic, and that a balanced default is announced rather than assumed.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

SNAPSHOT_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"

#: (model, ep, num_gpus). The first two are WRAPPER banks — the expert body is
#: a `swiglu_ffn` component and the routed GEMMs are its children, which is the
#: shape the old leaf-level predicate missed entirely. gpt-oss is a PACKED bank
#: (entry node and leaf are the same node) and is the shape that hid the bug.
ROUTED = (
    ("deepseek-ai__DeepSeek-V3", 8, 8),
    ("Qwen__Qwen3-30B-A3B", 4, 4),
    ("openai__gpt-oss-20b", 4, 4),
)

#: Nothing routes here, so nothing is assumed about routing.
DENSE = (("meta-llama__Llama-3.2-3B", 2),)


def _capacity(model: str, num_gpus: int, ep_load_factor: float = 1.0, **pk):
    from backend.inference.capacity import estimate_capacity
    from backend.inference.parallel_config import ParallelConfig

    config = json.loads(
        (SNAPSHOT_DIR / f"{model}.json").read_text())["config"]
    return estimate_capacity(
         config=config, gpu_sku="H200", num_gpus=num_gpus,
        avg_input_tokens=4096, avg_output_tokens=256,
        model_id=model.replace("__", "/", 1),
        parallel=ParallelConfig(**pk), ep_load_factor=ep_load_factor)


def _zone_uids(model: str, num_gpus: int, **pk):
    """The set `compute_per_op_timing` actually scales, via `analyze`."""
    from backend.arch.loader import get_default_loader
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.pipeline import analyze

    config = json.loads(
        (SNAPSHOT_DIR / f"{model}.json").read_text())["config"]
    spec = get_default_loader().match(config, model_id=model.replace("__", "/", 1))
    return analyze(
        spec,
        batch_size=1,
        avg_input_tokens=4096, avg_output_tokens=0,
        parallel=ParallelConfig(**pk),
        breakdown=False,
    ).dispatch_zone_uids


class TestDispatchZoneMembership(unittest.TestCase):

    maxDiff = None

    def test_a_routed_bank_has_members_whatever_shape_it_is(self):
        for model, ep, ngpu in ROUTED:
            with self.subTest(model=model):
                uids = _zone_uids(model, ngpu, ep_moe=ep)
                self.assertTrue(
                    uids,
                    "no leaf is inside the dispatch zone of a model that "
                    "routes — the straggler factor has nothing to scale, so "
                    "`ep_load_factor` is dead input and the estimate reports "
                    "perfectly balanced routing without saying so")

    def test_a_dense_model_has_none(self):
        for model, ngpu in DENSE:
            with self.subTest(model=model):
                self.assertEqual(
                    _zone_uids(model, ngpu, tp_attention=ngpu),
                    frozenset(),
                )

    def test_the_membership_reaches_the_arithmetic(self):
        """A knob nothing reads is not a knob.

        A leaf in it runs on the busiest rank's routed rows, so its time
        reads the batch's skew. Pinning "1.3 changes the step time" rather
        than a specific millisecond keeps this a wiring test: the size of the
        effect belongs to `phases`, the fact that there IS one belongs here.
        """
        for model, ep, ngpu in ROUTED:
            with self.subTest(model=model):
                bal = _capacity(model, ngpu, ep_moe=ep, ep_load_factor=1.0)
                skew = _capacity(model, ngpu, ep_moe=ep, ep_load_factor=1.3)
                self.assertGreater(
                    skew["phases"]["prefill"]["t_compute_s"],
                    bal["phases"]["prefill"]["t_compute_s"],
                    "a hot expert costs nothing — the busiest rank's rows "
                    "never reached the per-op timing")

    def test_the_balanced_default_is_announced_and_a_measurement_is_not(self):
        """`ep_load_factor = 1.0` is a claim about the world, not a neutral.

        It says every expert receives exactly the mean token load, which is
        false of every published routing trace, and it makes the MoE step
        times a lower bound rather than an estimate. `feedback_mfu_no_defaults`
        rules out inventing a better number, so the deployment says what it
        assumed instead — and stops saying it the moment someone supplies a
        measured factor, because then it is an input and not an assumption.
        """
        for model, ep, ngpu in ROUTED:
            with self.subTest(model=model):
                ids = [a["id"] for a in
                       _capacity(model, ngpu, ep_moe=ep).get("assumptions", [])]
                self.assertIn("moe_routing_balance", ids)

                measured = [a["id"] for a in _capacity(
                    model, ngpu, ep_moe=ep, ep_load_factor=1.3
                ).get("assumptions", [])]
                self.assertNotIn("moe_routing_balance", measured)

        for model, ngpu in DENSE:
            with self.subTest(model=model):
                ids = [a["id"] for a in _capacity(
                    model, ngpu, tp_attention=ngpu).get("assumptions", [])]
                self.assertNotIn(
                    "moe_routing_balance", ids,
                    "a dense model is not assuming anything about routing; "
                    "telling readers it is trains them to skip the list")


if __name__ == "__main__":
    unittest.main()
