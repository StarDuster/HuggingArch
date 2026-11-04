"""The FLOPs breakdown sums to the FLOPs headline it explains.

Three independent walks describe one step: the table (`phases._flops_rows`
over `project_model`), the per-op rows (`pipeline.analyze`), and the per-rank
cost tree. The phase's `flops_per_step` is a fold of those rows — so the table
the UI draws underneath that number must add up to it. It did not, and every way it failed was a difference
of CONTENTS rather than of arithmetic, which is why this is a test and not a
tolerance:

  * the LM head was billed at every prefill position instead of at the
    positions the sampler reads (8192× on a full-length prefill step);
  * aux stacks — MTP, vision, audio — were charged to a step that does not
    run them;
  * `max(count, 1)` billed one layer of each template a checkpoint declares
    and then instantiates zero times (GLM-5.1 declares two);
  * only the LM head was emitted as the model-level tail, so the final norm
    was in the per-op sum and not in the table.

`invariants._inv_flops_closure` is the runtime form of this check and is what
the tests below call, so the two cannot drift apart.
"""
from __future__ import annotations

import sys
import unittest

from tests.snapshots import shared_loader, snapshot_config
import pytest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.capacity import estimate_capacity
from backend.inference.invariants import _inv_flops_closure
from backend.inference.parallel_config import ParallelConfig
from backend.inference.phases import (
    evaluate_phase_step, flops_fold,
)
from backend.inference.step_eval import sampled_logit_positions

pytestmark = pytest.mark.regression  # walks every snapshot


SNAPSHOT_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _config(model_name):
    return snapshot_config(model_name)


def _resolved(model_name: str):
    cfg = _config(model_name)
    return None if cfg is None else shared_loader().match(
        cfg, model_id=model_name.replace("__", "/", 1))


def _table(model_name: str, phase: str, *, s_verify: int = 1,
           parallel: ParallelConfig = None):
    resolved = _resolved(model_name)
    if resolved is None:
        return None
    step, _bind = evaluate_phase_step(
        resolved, parallel or ParallelConfig(),
        phase=phase, batch=1,
        avg_input_tokens=4096, avg_output_tokens=256, s_verify=s_verify,
    )
    return flops_fold(step).rows


# One dense model, one MoE, one carrying an MTP stack, and GLM-5.1 — a
# per-layer assignment map with a dense prefix.
CLOSURE_MODELS = [
    "meta-llama__Llama-3.2-3B",
    "Qwen__Qwen3-30B-A3B",
    "deepseek-ai__DeepSeek-V3",
    "zai-org__GLM-5.1",
]


def _tp(n: int, pp: int = 1) -> ParallelConfig:
    return ParallelConfig(pp=pp, tp_attention=n, tp_dense_ffn=n,
                          tp_moe_expert=n)


#: The smallest one-stage H200 deployment that holds each of the big two.
#: DeepSeek-V3 carries 627 GiB of weights and GLM-5.1 1.39 TiB, so neither
#: admits a request unsharded — and a card that admits none publishes no
#: per-op rows to weigh, so their cases run where a batch actually fits. The
#: small two fit anywhere.
_HELD = {
    "deepseek-ai__DeepSeek-V3": _tp(8),
    "zai-org__GLM-5.1": _tp(16),
}
_STAGED = {
    "deepseek-ai__DeepSeek-V3": _tp(2, pp=4),
    "zai-org__GLM-5.1": _tp(4, pp=4),
}


class TestClosureAcrossModels(unittest.TestCase):
    """Σ depth-0 rows == the phase's published `flops_per_step`."""

    def _check(self, model_name: str, parallel: ParallelConfig,
               require: tuple = ()):
        cfg = _config(model_name)
        res = estimate_capacity(
             config=cfg, gpu_sku="H200", resolved=_resolved(model_name),
            num_gpus=max(8, parallel.cluster_size),
            avg_input_tokens=4096, avg_output_tokens=256, parallel=parallel)
        checks = _inv_flops_closure(res)
        names = [c.name for c in checks]
        # A closure with no phase to check passes vacuously, so the count is
        # part of the assertion: the estimate has to have produced the per-op
        # rows the table is being weighed against.
        self.assertTrue(names, f"{model_name}: the estimate produced no per-op "
                               f"rows, so this test asserted nothing")
        for want in require:
            self.assertIn(want, names, f"{model_name}: {want} was not checked")
        for c in checks:
            self.assertTrue(c.passed, f"{model_name}: {c.name}: {c.detail}")

    def test_closure_holds_on_one_stage(self):
        """No pipeline cut: the phase's rows are the model's, once.

        The big two cannot run this unsharded — no single rank holds them —
        so their case is the smallest one-stage deployment that does.
        """
        for m in CLOSURE_MODELS:
            with self.subTest(model=m):
                self._check(m, _HELD.get(m, ParallelConfig()),
                            require=("flops_closure[prefill]",))

    def test_closure_holds_when_sharded(self):
        """The table carries `flops_per_rank` here, and the closure reads that
        column instead — a per-rank walk that sharded an axis the per-op rows
        left whole shows up as a broken sum rather than as a plausible number.

        """
        for m in CLOSURE_MODELS:
            with self.subTest(model=m):
                self._check(m, _HELD.get(m, _tp(8)),
                            require=("flops_closure[prefill]",))

    def test_closure_holds_on_a_decode_step(self):
        """Decode named explicitly, because it is the phase the check can miss:
        `require=` is what makes a vacuous pass fail, and without naming decode
        a run that only produced a prefill table would still be green.
        """
        for m in ("meta-llama__Llama-3.2-3B", "Qwen__Qwen3-30B-A3B"):
            with self.subTest(model=m):
                self._check(m, ParallelConfig(tp_attention=8, tp_dense_ffn=8,
                                              tp_moe_expert=8),
                            require=("flops_closure[decode]",))

    def test_closure_holds_where_the_deployment_admits_a_batch(self):
        """The big models on a config that actually holds them.

        DeepSeek-V3 and Kimi-K2 need 673 GB and 1029 GB per rank unsharded,
        and a deployment that admits no request publishes no step at all —
        the closure was only ever weighed on the no-capacity path for
        MLA+MoE. It has to hold on the ordinary one too,
        and `max_batch_per_gpu` is asserted here rather than assumed, because a
        capacity regression that took these back to zero would otherwise turn
        this into a duplicate of the sharded case above.

        Measured at 8×H200: DeepSeek-V3 85.14 GB/rank admits 167, Kimi-K2
        129.75 GB/rank admits 21.
        """
        tp8 = ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8)
        for m in ("deepseek-ai__DeepSeek-V3", "moonshotai__Kimi-K2-Instruct"):
            with self.subTest(model=m):
                cfg = _config(m)
                res = estimate_capacity(
                     config=cfg, resolved=_resolved(m), gpu_sku="H200",
                    num_gpus=tp8.cluster_size,
                    avg_input_tokens=4096, avg_output_tokens=256,
                    parallel=tp8, selection_mode="theoretical_only")
                self.assertGreater(
                    int(res.get("max_batch_per_gpu") or 0), 0,
                    f"{m}: this deployment no longer admits a request, so the "
                    f"case it was added to cover is not being covered")
                for phase in ("prefill", "decode"):
                    self.assertIsNone(
                        (res["phases"][phase].get("time_unpublished_reason")
                         or None),
                        f"{m} {phase}: time withheld on a fitting deployment")
                self._check(m, tp8, require=("flops_closure[prefill]",
                                             "flops_closure[decode]"))

    def test_closure_holds_across_pipeline_stages(self):
        """Under `pp` both walks are restricted to one stage's layers, and the
        restriction has to be the same one. A stage holding zero layers of a
        template is the `max(count, 1)` defect's other habitat.
        """
        for m in CLOSURE_MODELS:
            with self.subTest(model=m):
                self._check(m, _STAGED.get(m, ParallelConfig(pp=4)),
                            require=("flops_closure[prefill]",))


class TestHeadRunsWhereTheSamplerReads(unittest.TestCase):
    """The LM head's own row, at the positions `sampled_logit_positions` gives.

    Prefill samples one position no matter how long the prompt is; a
    speculative decode step verifies `s_verify` of them. The table used to
    read the head at `s_query`, which on a 4096-token prefill is the whole
    prompt.
    """

    def _head(self, rows):
        head = [r for r in rows
                if r["depth"] == 0 and r.get("op") == "linear"
                and "lm_head" in (r["name"] or "")]
        self.assertEqual(len(head), 1, f"expected one lm_head row, got {head}")
        return head[0]

    def test_prefill_samples_one_position(self):
        rows = _table("meta-llama__Llama-3.2-3B", "prefill")
        self.assertEqual(self._head(rows)["shape_out"][-2], 1)

    def test_decode_verifies_the_speculative_window(self):
        for k in (1, 4):
            with self.subTest(s_verify=k):
                rows = _table("meta-llama__Llama-3.2-3B", "decode", s_verify=k)
                self.assertEqual(self._head(rows)["shape_out"][-2],
                                 sampled_logit_positions("decode", k))


class TestTheTailIsEveryModelLevelNode(unittest.TestCase):
    """Not just the head. The final norm lives at model level too, and it was
    the difference the head fix left behind."""

    def test_final_norm_is_a_row(self):
        rows = _table("meta-llama__Llama-3.2-3B", "prefill")
        tail = [r["name"] for r in rows
                if r["depth"] == 0 and not r["template_name"]]
        self.assertTrue(any("norm" in n for n in tail),
                        f"model-level tail carries no norm: {tail}")

    def test_the_refold_equals_the_declared_fold_on_every_model(self):
        """The report total and `Program.contributions` both fold the model as
        `Σ count × block + tail`. They describe one model at one anchor, so
        they agree exactly or one fold is holding a node the other is not.

        The whole corpus, not a hand-picked few: the tail used to be the single
        node whose `boundary_role` is `output_head`, and on a text-only model
        that drops one final norm — 1e-7 of the total, invisible under any
        tolerance and invisible on the four models this file otherwise checks.
        On a multimodal model it drops the entire front end. Measured against
        the head-by-name tail: gemma-4-12B-it lost 358,528,057,344 FLOPs
        (3.8e-3) across `embed_audio.embedding_projection`,
        `embed_vision.embedding_projection` and four `vision_embedder` ops;
        MiMo-V2.5 lost 799,990,087,680 (5.8e-3) across the audio encoder's
        projection MLP and the visual merger. All 39 snapshots were
        undercounting; the corpus is the only place the second kind shows.
        """
        from backend.arch.eval.program import Program
        from backend.arch.eval.transforms import Deployment
        from backend.arch.eval.projection import project_analysis

        snapshots = sorted(SNAPSHOT_DIR.glob("*.json"))
        self.assertGreater(len(snapshots), 30,
                           "the corpus is what this test is for — a handful of "
                           "snapshots would not reach the multimodal case")
        seen = 0
        for path in snapshots:
            with self.subTest(model=path.stem):
                resolved = _resolved(path.stem)
                analysis = Deployment(Program.of(resolved)).at(
                    batch=1, input_tokens=4096, logit_positions=1).analyze()
                walk = float(analysis.total().flops)
                tree = float(project_analysis(analysis).total_flops)
                seen += 1
                # Exact, not near: both sides fold the same per-node numbers,
                # so any difference is a whole node, never a rounding.
                self.assertLessEqual(
                    abs(tree - walk), max(1.0, abs(walk) * 1e-12),
                    f"{path.stem}: the re-fold says {tree:,.0f} and the "
                    f"declared fold says {walk:,.0f} — a node one of them "
                    f"holds and the other does not")
        self.assertGreater(seen, 30, "every model skipped; nothing was checked")


class TestZeroLayersCostsZero(unittest.TestCase):
    """A stack whose declared repeat resolves to zero is not in the model.

    Kimi-K2 publishes `num_nextn_predict_layers: 0` under the DeepSeek-V3
    spec, whose MTP group repeats `n_mtp_layers`. `max(count, 1)` once read a
    zero as a missing count and billed one layer of the stack — 1.81e13 of
    7.36e14 FLOPs at 8192 prefill tokens on GLM-5.1, against per-op rows that
    correctly charged nothing. The enumerator now skips the stack, so no
    consumer sees a zero-count template at all: no header row, no FLOPs.
    """

    ZERO = "moonshotai__Kimi-K2-Instruct"
    ONE = "deepseek-ai__DeepSeek-V3"

    def test_a_zero_repeat_stack_is_absent(self):
        cfg = snapshot_config(self.ZERO)
        self.assertEqual(cfg["num_nextn_predict_layers"], 0,
                         "Kimi-K2 no longer publishes zero MTP layers — this "
                         "test lost its subject, point it at one that does")
        self.assertEqual(snapshot_config(self.ONE)["num_nextn_predict_layers"], 1)
        # The same spec family declares the stack; the count is what differs.
        from backend.arch.spec_tree import iter_blocks
        one = [b for b in iter_blocks(_resolved(self.ONE)) if b.aux_kind == "mtp"]
        zero = [b for b in iter_blocks(_resolved(self.ZERO)) if b.aux_kind == "mtp"]
        self.assertEqual([b.count for b in one], [1])
        self.assertEqual(zero, [], "a repeat of 0 still enumerated a block")
        for b in iter_blocks(_resolved(self.ZERO)):
            self.assertGreater(b.count, 0)

    def test_no_header_row_is_billed_for_zero_layers(self):
        rows = _table(self.ZERO, "prefill")
        headers = [r for r in rows
                   if r["depth"] == 0 and r.get("op") == "<template>"]
        self.assertTrue(headers)
        for r in headers:
            self.assertGreater(r["layer_count"], 0, r["name"])
            self.assertNotIn("mtp", r["name"])


class TestAuxStacksDoNotRunInAStep(unittest.TestCase):
    """MTP / vision / audio blocks are not part of a prefill or decode step,
    as declared by Program stack semantics."""

    def test_aux_templates_are_absent_from_the_table(self):
        from backend.arch.eval.program import BlockStage, Program
        resolved = _resolved("deepseek-ai__DeepSeek-V3")
        program = Program.of(resolved)
        aux = frozenset(
            stage.template.name
            for stack in program.stacks if stack.is_aux
            for stage in stack.stages if isinstance(stage, BlockStage)
        )
        self.assertGreater(len(aux), 0,
                           "DeepSeek-V3 no longer carries an aux stack — this "
                           "test lost its subject")
        rows = _table("deepseek-ai__DeepSeek-V3", "prefill")
        billed = sorted({r["template_name"] for r in rows
                         if r["template_name"] in aux})
        self.assertEqual(billed, [], f"aux stacks billed to the step: {billed}")


if __name__ == "__main__":
    unittest.main()
