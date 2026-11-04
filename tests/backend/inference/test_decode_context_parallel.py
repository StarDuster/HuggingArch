"""Decode context parallelism is a SUBGROUP, and it splits the cache only.

Two claims, both from vLLM's own source, and both of which this model got
wrong in opposite directions before.

The first is about ranks. `config/parallel.py` puts prefill context
parallelism in the cluster product (`world_size = pp * tp * pcp`) and says of
the decode one: "DCP does not expand the process world size. Without PCP, DCP
reuses TP ranks", validating `tp % dcp == 0`. So the two are not one knob read
in two phases — one buys ranks and the other re-cuts ranks already bought.
Modelling DCP as another factor asks a user for `dcp x` the GPUs their
deployment actually runs, which is the direction of error that looks prudent
and is simply wrong.

The second is about weights, and is the claim task #14 was opened on. A decode
context-parallel group DOES all_gather the query and reduce_scatter the output
on the head axis, and that looks like a head shard — but `deepseek_v2.py`
builds `o_proj` as a plain `RowParallelLinear` over `num_heads // tp_size` in
every mode, `q_proj` is a plain `ColumnParallelLinear`, and the opt-in
`VLLM_DCP_Q_REPLICATE` path (default OFF, `envs.py`) makes q_proj LARGER, not
smaller. The collectives are comm around the attention core. Inferring a
weight divisor from a head-axis collective is the mistake; per-rank weights
being INDEPENDENT of dcp is the pin.

What DCP does buy is the whole point of it: the cache costs `1/dcp` per rank,
so the same GPUs hold `dcp x` the context — but only out of the copies TP left
behind. `config/model.py` refuses DCP for GQA unless `tp > total_num_kv_heads`
and caps it at `tp // total_num_kv_heads`, because a cache already cut one head
per rank has nothing left to cut. That is why DCP is an MLA / MQA feature, and
why the third claim here is a REFUSAL: the same `kv_replication_factor` that
prices the budget is what says how far the split can go.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

SNAPSHOT_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"

#: (model, tp, dcp). One MLA model and one GQA model — the cache geometries
#: differ enough that a divisor landing in the wrong place shows up on one and
#: not the other. Each `dcp` is the largest the model's own replication allows,
#: which is the point of carrying it in the table rather than sharing one
#: constant: DeepSeek-V3's latent is head-free so all 8 TP ranks hold a copy,
#: while Qwen3-30B's 4 KV heads over tp=8 leave exactly 2 — the two ends of the
#: rule, not two spellings of one number.
#:
#: The TP width is also whatever makes the WEIGHTS fit on H200, so the KV
#: budget is a real number: DeepSeek-V3 at tp=4 has none left over and every
#: capacity answer degenerates to 0, which compares equal for the wrong reason.
MODELS = (("deepseek-ai__DeepSeek-V3", 8, 8), ("Qwen__Qwen3-30B-A3B", 8, 2))


def _config(model: str) -> dict:
    return json.loads((SNAPSHOT_DIR / f"{model}.json").read_text())["config"]


def _resolved(model: str):
    """Resolve the spec once so capacity never re-matches (single entry)."""
    from tests.snapshots import shared_loader
    return shared_loader().match(_config(model), model_id=model.replace("__", "/", 1))


def _capacity(model: str, pc):
    from backend.inference.capacity import estimate_capacity
    return estimate_capacity(
         config=_config(model), resolved=_resolved(model), gpu_sku="H200",
        num_gpus=pc.cluster_size,
        avg_input_tokens=4096, avg_output_tokens=256, parallel=pc)


class TestDecodeContextParallelIsASubgroup(unittest.TestCase):

    def test_it_adds_no_ranks(self):
        from backend.inference.parallel_config import ParallelConfig
        plain = ParallelConfig(tp_attention=8, tp_dense_ffn=8,
                               tp_moe_expert=8)
        with_dcp = ParallelConfig(tp_attention=8, tp_dense_ffn=8,
                                  tp_moe_expert=8, dcp=4)
        # `cluster_size` is the rank count — the ONE answer, on the config
        # that owns the mesh. There is no second one on the eval `Parallelism`
        # to keep in step with it: that layer resolves TP per WORLD through
        # `world_degrees` and never holds a single `tp` to multiply.
        self.assertEqual(with_dcp.cluster_size, plain.cluster_size)
        self.assertEqual(with_dcp.cluster_size, 8)

    def test_prefill_context_parallel_does_add_ranks(self):
        """The control: `cp` is the one that grows the cluster."""
        from backend.inference.parallel_config import ParallelConfig
        self.assertEqual(
            ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8,
                           cp=4, dp_attention=1).cluster_size, 32)

    def test_its_degree_must_be_a_group_the_mesh_can_form(self):
        from backend.inference.parallel_config import ParallelConfig
        errs = ParallelConfig(tp_attention=8, tp_dense_ffn=8,
                              tp_moe_expert=8, dcp=3).validate()
        self.assertTrue(any("dcp" in e for e in errs), errs)
        self.assertEqual(
            ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8,
                           dcp=8).validate(), [])

    def test_it_composes_with_the_prefill_one_without_multiplying_it(self):
        """Both can be > 1 — vLLM allows `dcp in (1, pcp, tp*pcp)` — and the
        cache is still spread `dcp` ways, not `cp * dcp` ways.

        The product was this model's reading of `s_kv` as the axis both context
        parallelisms divide, and vLLM contradicts it in three places:
        `attention/pcp.py::_gather_prefill_cache_inputs` all_gathers the
        prefill K/V across the PCP group BEFORE the cache write,
        `worker/block_table.py` passes `TOTAL_CP_WORLD_SIZE=dcp_world_size` to
        the slot-mapping kernel, and `kv_cache_interface.max_memory_usage_bytes`
        divides `max_model_len` by `dcp_world_size` alone. Prefill context
        parallelism splits the tokens a rank PROCESSES; every rank keeps the
        whole cache.
        """
        from backend.inference.parallel_config import ParallelConfig
        base = dict(tp_attention=2, tp_dense_ffn=2, tp_moe_expert=2,
                    ep_moe=4, cp=4, dp_attention=1)
        self.assertEqual(ParallelConfig(**base, dcp=4).validate(), [])
        for dcp, want in ((1, 1), (4, 4), (8, 8)):
            with self.subTest(dcp=dcp):
                degrees = ParallelConfig(**base, dcp=dcp).axis_degrees(
                    phase="decode")
                self.assertEqual(degrees.get("s_kv", 1), want)

    def test_prefill_context_parallelism_does_not_touch_the_cache(self):
        """The claim above, priced: `cp` moves the per-request KV bytes not at
        all, while `dcp` moves them by its degree. Measured on the head-free
        cache, where TP has no say either and the two knobs are the only
        things that could move the number."""
        from backend.inference.parallel_config import ParallelConfig
        model = "deepseek-ai__DeepSeek-V3"
        alone = _capacity(model, ParallelConfig(
            tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8))
        per_req = "per_request_kv"
        whole = alone["formulas"][per_req]["result_bytes"]
        for tp, cp, dcp, div in ((4, 2, 1, 1), (2, 4, 1, 1),
                                 (8, 1, 2, 2), (2, 4, 4, 4), (2, 4, 8, 8)):
            with self.subTest(tp=tp, cp=cp, dcp=dcp):
                pc = ParallelConfig(tp_attention=tp, tp_dense_ffn=tp,
                                    tp_moe_expert=tp, cp=cp, ep_moe=cp,
                                    dcp=dcp)
                self.assertEqual(pc.validate(), [])
                got = _capacity(model, pc)["formulas"][per_req]["result_bytes"]
                self.assertAlmostEqual(got, whole / div, delta=1.0)


class TestDecodeContextParallelSplitsTheCacheOnly(unittest.TestCase):

    def test_the_cache_shrinks_and_capacity_grows_by_the_degree(self):
        from backend.inference.parallel_config import ParallelConfig
        for model, tp, dcp in MODELS:
            with self.subTest(model=model):
                base = dict(tp_attention=tp, tp_dense_ffn=tp,
                            tp_moe_expert=tp)
                one = _capacity(model, ParallelConfig(**base))
                split = _capacity(model, ParallelConfig(**base, dcp=dcp))
                per_req = "per_request_kv"
                self.assertAlmostEqual(
                    one["formulas"][per_req]["result_bytes"] / dcp,
                    split["formulas"][per_req]["result_bytes"],
                    delta=1.0,
                    msg="the cache is what decode context parallelism splits")
                self.assertGreater(split["formulas"]["max_tokens"]["result"],
                                   one["formulas"]["max_tokens"]["result"])

    def test_the_projections_are_not_head_sharded_by_it(self):
        """The pin on the inference this task was opened on.

        A head-axis all_gather / reduce_scatter around the attention core is
        comm; the weights either side keep their `n_heads / tp_attention`
        split. If a future change divides them by `dcp` as well, per-rank
        weights fall and this fails — which is the whole point of writing it
        down, because the number would still look plausible.
        """
        from backend.inference.parallel_config import ParallelConfig
        for model, tp, dcp in MODELS:
            with self.subTest(model=model):
                base = dict(tp_attention=tp, tp_dense_ffn=tp,
                            tp_moe_expert=tp)
                one = _capacity(model, ParallelConfig(**base))
                split = _capacity(model, ParallelConfig(**base, dcp=dcp))
                self.assertEqual(
                    one["formulas"]["weight_per_rank"]["result_bytes"],
                    split["formulas"]["weight_per_rank"]["result_bytes"])


class TestItCannotSplitPastTheCopiesTPLeft(unittest.TestCase):
    """The refusal, and the reason DCP is an MLA / MQA feature.

    `config/model.py:1385-1405` gates DCP on `tp > total_num_kv_heads` and caps
    it at `tp // total_num_kv_heads`. Both are the same statement read from the
    cache end: `get_num_kv_heads` is `max(1, total // tp)`, and that `max(1,
    ...)` is where TP stops splitting and starts REPLICATING. DCP re-cuts the
    replicas; where there are none it is asking for bytes no rank holds twice.

    We do not reimplement vLLM's head arithmetic to say so. `kv_replication_
    factor` — the attention ranks over the split the cache actually took — is
    already computed to price the budget, and an axis that BUYS ranks cancels
    out of it, so only a subgroup can drive it below 1. The cap falls out.
    """

    def _reject(self, model: str, tp: int, dcp: int):
        from backend.inference.parallel_config import ParallelConfig
        with self.assertRaises(ValueError) as cm:
            _capacity(model, ParallelConfig(
                tp_attention=tp, tp_dense_ffn=tp, tp_moe_expert=tp, dcp=dcp))
        self.assertIn("dcp", str(cm.exception))
        return str(cm.exception)

    def test_gqa_whose_kv_heads_already_fill_tp_refuses_any_split(self):
        """Llama-3.2-3B has 8 KV heads; at tp=4 every rank owns 2 of them and
        holds no copy of anyone else's. vLLM raises for the same config
        because `tp (4) <= total_num_kv_heads (8)`."""
        self._reject("meta-llama__Llama-3.2-3B", tp=4, dcp=2)

    def test_it_stops_exactly_where_the_replication_does(self):
        """Qwen3-30B-A3B has 4 KV heads over tp=8: a factor of 2 replicated,
        so dcp=2 is legal and dcp=4 is not — vLLM's `max_dcp_size = 8 // 4`."""
        from backend.inference.parallel_config import ParallelConfig
        base = dict(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8)
        _capacity("Qwen__Qwen3-30B-A3B", ParallelConfig(**base, dcp=2))
        self._reject("Qwen__Qwen3-30B-A3B", tp=8, dcp=4)

    def test_the_head_free_cache_has_no_cap_below_tp(self):
        """The control. DeepSeek-V3's MLA latent carries no head axis, so TP
        replicates it across all 8 ranks and dcp=8 splits every one of them —
        vLLM likewise applies no head cap to MLA."""
        from backend.inference.parallel_config import ParallelConfig
        base = dict(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8)
        for dcp in (2, 4, 8):
            with self.subTest(dcp=dcp):
                _capacity("deepseek-ai__DeepSeek-V3",
                          ParallelConfig(**base, dcp=dcp))


class TestTheTwoPhasesSplitTheSameAxisTheSameWay(unittest.TestCase):
    """`s_kv` divides by `dcp` in both phases, and by nothing else in either.

    There was a phase correction here — prefill undid the PREFILL split of
    `s_kv` on the reasoning that Ulysses gathers the whole context and pcp
    streams it — and it was a read degree patching a degree table that should
    not have divided the axis at all. vLLM's PCP all_gathers the K/V before the
    cache write, so a rank that attends the whole context also KEEPS it; and
    what `dcp` splits it splits in prefill too (the chunked-context planner
    gives each rank `1/dcp` of the context rows). Owned and read agree, in
    both phases, which is why one degree per axis is enough.

    The token axes `cp` DOES divide are `s_query` and `s_ctx`, and they are
    tested here beside `s_kv` — the two being different axes is the whole
    content of "prefill and decode parallelism are distinct knobs".
    """

    def _degrees(self, *, cp: int, dcp: int, phase: str):
        from backend.arch.eval.transforms import Parallelism, Program
        from tests.snapshots import shared_loader
        resolved = shared_loader().match(_config("deepseek-ai__DeepSeek-V3"), model_id="deepseek-ai/DeepSeek-V3")
        return Parallelism(tp=1, cp=cp, dcp=dcp).policy(
            Program.of(resolved), phase=phase)

    def test_prefill_context_parallelism_never_divides_the_attended_context(self):
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                self.assertEqual(
                    self._degrees(cp=4, dcp=1, phase=phase).get("s_kv", 1), 1)

    def test_decode_context_parallelism_divides_it_in_both_phases(self):
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                self.assertEqual(
                    self._degrees(cp=4, dcp=2, phase=phase).get("s_kv", 1), 2)

    def test_the_query_tokens_are_the_axis_prefill_cp_does_divide(self):
        """The other half of the distinction. `cp` reaches `s_ctx` — the tokens
        a rank processes — and stops there."""
        degrees = self._degrees(cp=4, dcp=1, phase="prefill")
        self.assertEqual(degrees.get("s_ctx", 1), 4)
        self.assertEqual(degrees.get("s_kv", 1), 1)


class TestCachedTensorsLeaveTheProducerWhole(unittest.TestCase):
    """Prefill CP gathers each cached tensor at its producer, once.

    Window and union then see whole positions — a gather anchored at
    `kv_stream_concat` would mean the cut leaked past the producer.
    """

    def test_v4_pro_has_no_union_cp_gather_and_gathers_cached_producers(self):
        from tests.snapshots import shared_loader
        from backend.arch.layout.materializer import \
            materialize_comm_events
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.step_eval import evaluate_step
        resolved = shared_loader().match(_config("deepseek-ai__DeepSeek-V4-Pro"), model_id="deepseek-ai/DeepSeek-V4-Pro")
        step = evaluate_step(
            resolved,
            ParallelConfig(tp_attention=1, tp_dense_ffn=1, tp_moe_expert=1,
                           ep_moe=8, cp=8),
            phase="prefill", batch=1, input_tokens=4096, output_tokens=0)
        by_uid = {n.uid: n for n in step.analysis.program.nodes()}
        ops = []
        for event in materialize_comm_events(step.analysis):
            if (event.op, event.world) != ("all_gather", "cp"):
                continue
            anchor = by_uid[event.report_anchor_uid]
            ops.append(anchor.op)
        self.assertNotIn("kv_stream_concat", ops)
        self.assertTrue(ops)
        self.assertIn("concat", ops)

    def test_prefix_under_dcp_is_refused(self):
        from tests.snapshots import shared_loader
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.step_eval import evaluate_step
        resolved = shared_loader().match(_config("meta-llama__Llama-3.2-1B"), model_id="meta-llama/Llama-3.2-1B")
        with self.assertRaisesRegex(ValueError, "cached prefix"):
            evaluate_step(
                resolved,
                ParallelConfig(tp_attention=4, dcp=2),
                phase="prefill", batch=1, input_tokens=4096, output_tokens=0, s_prefix_cached=3072)


if __name__ == "__main__":
    unittest.main()
