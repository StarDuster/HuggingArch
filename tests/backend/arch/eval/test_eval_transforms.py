"""Transforms partitioned by what they change, and `Deployment` as the one
place they compose.

Falsification targets:
  - the phase math is being MOVED from `inference/phases.py`, not reinvented, so
    it must agree with it exactly on every workload shape — prefill with and
    without a prefix cache, decode autoregressive and speculative;
  - `s_ctx` must follow the phase. It is the activation length, the same
    quantity `s_query` names, and a decode step left at the prompt length costs
    4096x too much everywhere except attention;
  - expert parallelism must divide the expert AXIS (`n_expert_copies`) and
    never the dispatch denominator (`n_route_targets`), because dividing the
    denominator changes what fraction of the bank a token activates;
  - a `Deployment` must be reusable: asking it for a second phase must not
    disturb the first, or the composition site has the mutable state it exists
    to remove.
"""
from __future__ import annotations

import sys
import unittest


import sympy as sp
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.dim import Dim, expr_of, names_of, value_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import (
    DECODE, PREFILL, Deployment, Parallelism, Phase, Workload,
)
from tests.snapshots import shared_loader, shared_program, shared_walk

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _program(name: str) -> Program:
    return shared_program(name.replace('__', '/', 1))


def _kv_b_views(program: Program) -> set:
    """The nodes that run the stored `kv_b_proj` without holding it: each
    stores at the `kv_b_proj` address of its own graph (`module:`), and that node is
    the only holder of the tensor."""
    views = set()
    for node in program.nodes():
        if node.tied_to != "kv_b_proj":
            continue
        owner = program.by_uid(node.tied_to_uid)
        assert owner.name == "kv_b_proj" and owner.owns_weight
        assert owner.parent is node.parent
        views.add(node.name)
    return views


def _single_rank_rule():
    """The rule a bare `Evaluator` walk runs under: one rank, no worlds."""
    from backend.arch.layout.parallel_config import ParallelConfig
    from backend.arch.layout.propagation import LayoutPropagation

    return LayoutPropagation.of(ParallelConfig(), "prefill")


class TestPhaseMathMatchesTheModuleItMovedFrom(unittest.TestCase):
    """`inference/phases.phase_binding` is the incumbent. Every workload shape
    it handles must come out identical here, or this is a rewrite pretending to
    be a move."""

    CASES = (
        (PREFILL, 4096, 0, 0, 1),
        (PREFILL, 4096, 256, 1024, 1),      # prefix-cached
        (PREFILL, 8192, 2048, 512, 1),
        (DECODE, 1024, 512, 0, 1),          # autoregressive
        (DECODE, 1024, 512, 0, 4),          # speculative
        (DECODE, 4096, 1024, 0, 8),
    )

    def test_identical_on_every_workload_shape(self):
        from backend.inference.phases import phase_binding

        for kind, s_in, s_out, s_pre, k in self.CASES:
            with self.subTest(phase=kind, S=s_in, prefix=s_pre, verify=k):
                mine = Phase(kind=kind).resolve(Workload(
                    input_tokens=s_in, output_tokens=s_out,
                    s_prefix_cached=s_pre, s_verify=k))
                theirs = phase_binding(
                    phase=kind, avg_input_tokens=s_in, avg_output_tokens=s_out,
                    s_prefix_cached=s_pre, s_verify=k)
                for axis in ("s_query", "s_kv", "s_pairs"):
                    self.assertEqual(mine[axis], theirs[axis], axis)

    def test_the_pairs_keep_the_diagonal(self):
        """`Sq(Sq+1)/2`, not `Sq²/2`: at Sq=Lq=S the difference is `S/2` —
        the diagonal, which the approximation drops."""
        got = Phase(PREFILL).resolve(Workload(input_tokens=4096))
        self.assertEqual(got["s_pairs"], 4096 * 4097 // 2,
                         "exact, so the diagonal survives")
        self.assertNotEqual(got["s_pairs"], 4096 * 4096 // 2)

    def test_autoregressive_decode_has_no_triangle(self):
        got = Phase(DECODE, cache_len=1280).resolve(Workload(input_tokens=1024))
        self.assertEqual((value_of(got["s_query"]), got["s_kv"],
                          got["s_pairs"]),
                         (1, 1280, 1280))

    def test_decode_binds_n_verify_as_the_query_axis(self):
        """Spec decode's query length is n_verify. The walk has to carry
        that identity, or formula says s_ctx while values substitute 6."""
        got = Phase(DECODE, cache_len=1280).resolve(
            Workload(input_tokens=1024, s_verify=6))
        self.assertEqual(value_of(got["n_verify"]), 6)
        self.assertEqual(value_of(got["s_query"]), 6)
        self.assertNotIn("s_ctx", got)
        self.assertIsInstance(got["s_query"], Dim)
        self.assertTrue(got["s_query"].is_("n_verify"))

    def test_prefill_leaves_the_prompt_axis_alone(self):
        got = Phase(PREFILL).resolve(
            Workload(input_tokens=4096, s_verify=6))
        self.assertEqual(value_of(got["s_query"]), 4096)
        self.assertEqual(value_of(got["n_verify"]), 1)


class TestActivationLengthFollowsThePhase(unittest.TestCase):
    def test_s_ctx_lookup_is_s_query(self):
        from backend.arch.eval.binding import Binding
        for kind in (PREFILL, DECODE):
            b = Phase(kind, cache_len=512).bind(
                Binding(), Workload(input_tokens=2048, s_verify=3))
            self.assertIs(b["s_ctx"], b["s_query"], kind)

    def test_spec_decode_walk_writes_attention_flops_in_its_pairs(self):
        """The tree walk binds n_verify as the query axis and the verify
        block's live pairs as `s_pairs`, so a spec-decode score formula
        carries `s_pairs` instead of a folded number that the values line
        cannot line up with."""
        d = Deployment(_program("meta-llama__Llama-3.2-1B")).at(
            batch=2, input_tokens=1024, output_tokens=0, s_verify=6,
        ).during(DECODE, cache_len=1280)
        analysis = shared_walk(d)
        self.assertTrue(analysis.binding["s_query"].is_("n_verify"))
        self.assertEqual(value_of(analysis.binding["n_verify"]), 6)
        score = next(n for n in analysis.program.nodes()
                     if n.op == "attention_score")
        flops = analysis.cost.values[score.uid].flops
        self.assertIn("s_pairs", names_of(flops))
        folded = str(int(value_of(analysis.binding["s_pairs"])))
        self.assertNotIn(folded, str(expr_of(flops)).replace(" ", ""))

    def test_decode_costs_a_token_not_a_prompt(self):
        """Everything but attention scales with the activation length, so a
        decode step whose `s_ctx` stayed at the prompt length is off by the
        prompt length."""
        d = Deployment(_program("meta-llama__Llama-3.2-1B")).at(
            batch=1, input_tokens=4096, output_tokens=0)
        prefill = shared_walk(d).total().flops
        decode = shared_walk(d.during(DECODE, cache_len=4096)).total().flops
        self.assertLess(decode / prefill, 1 / 500,
                        "a decode step is one token wide")
        self.assertGreater(decode / prefill, 1 / 100_000)

    def test_params_do_not_depend_on_the_phase(self):
        d = Deployment(_program("meta-llama__Llama-3.2-1B")).at(input_tokens=512)
        self.assertEqual(shared_walk(d).total().params,
                         shared_walk(d.during(DECODE)).total().params)


class TestPhaseAlsoChangesTheGraph(unittest.TestCase):
    """The half that was missing when `Phase` was filed as a binding. At decode
    an MLA reads its KV in latent space — a different subgraph, shipped as its
    own component — not the same graph at `Sq=1`."""

    def test_decode_rebinds_a_component_that_declares_a_decode_form(self):
        prog = _program("deepseek-ai__DeepSeek-V3")
        self.assertEqual(prog.spec.get_block("mla_attention").decode_variant,
                         "mla_attention_absorbed")
        swapped = Phase(DECODE).substitute(prog)
        self.assertIsNot(swapped, prog)
        self.assertTrue(any("absorb" in (getattr(n, "op", "") or "")
                            or "absorb" in (getattr(n, "name", "") or "")
                            for n in swapped.nodes()))

    def test_prefill_substitutes_nothing(self):
        prog = _program("deepseek-ai__DeepSeek-V3")
        self.assertIs(Phase(PREFILL).substitute(prog), prog)

    def test_a_sparse_attention_prefills_in_its_decode_form(self):
        """A top-k (DSA) attention reads each query's selected latent rows,
        so serving prefills it absorbed, as it decodes; the same weights."""
        for model in ("deepseek-ai__DeepSeek-V3.2", "zai-org__GLM-5.3-Flash"):
            with self.subTest(model=model):
                prog = _program(model)
                swapped = Phase(PREFILL).substitute(prog)
                self.assertEqual(_kv_b_views(swapped),
                                 {"q_absorbed_nope", "v_absorbed"})
                d = Deployment(prog).at(input_tokens=512)
                self.assertEqual(
                    int(shared_walk(d).total().params),
                    int(shared_walk(d.during(DECODE)).total().params))

    def test_a_sparse_prefill_keeps_its_other_components_prefill_forms(self):
        """Only the selecting attention changes form at prefill: a compressor
        inside it compresses the prompt in its prefill form, and binds its
        decode form at decode alone."""
        from backend.arch.eval.transforms import phase_form
        prog = _program("zai-org__GLM-5.3-Flash")
        spec = prog.spec
        compressor = spec.get_block("glm_kpool_compressor")
        self.assertIsNotNone(compressor.decode_variant)
        self.assertIsNone(phase_form(compressor, PREFILL))
        self.assertEqual(phase_form(compressor, DECODE),
                         compressor.decode_variant)
        swapped = Phase(PREFILL).substitute(prog).spec
        self.assertIs(swapped.get_block("glm_kpool_compressor"), compressor)

    def test_the_component_declares_which_phases_run_its_variant(self):
        """`variant_phases` is the whole answer: the same sparse graph
        prefills authored once the declaration is taken off it, and a
        declaration with no variant to run is refused."""
        from backend.arch.eval.transforms import phase_form
        spec = _program("deepseek-ai__DeepSeek-V3.2").spec
        dsa = spec.get_block("mla_dsa_attention")
        self.assertEqual(dsa.variant_phases, ("prefill", "decode"))
        self.assertEqual(phase_form(dsa, PREFILL), dsa.decode_variant)
        undeclared = dsa.model_copy(update={"variant_phases": ("decode",)})
        self.assertIsNone(phase_form(undeclared, PREFILL))
        self.assertEqual(phase_form(undeclared, DECODE), dsa.decode_variant)
        with self.assertRaisesRegex(ValueError, "declares none"):
            type(dsa).model_validate({**dsa.model_dump(),
                                      "decode_variant": None})

    def test_a_model_with_no_declared_variant_is_untouched(self):
        prog = _program("meta-llama__Llama-3.2-1B")
        self.assertIs(Phase(DECODE).substitute(prog), prog)

    def test_the_original_spec_is_not_mutated(self):
        """`SpecLoader` caches one `ResolvedSpec` per model at module level, so
        every request shares it. A mutate-and-restore swap surfaced absorbed
        ops in a concurrent prefill that snapshotted the dict mid-window."""
        prog = _program("deepseek-ai__DeepSeek-V3")
        before = prog.spec.get_block("mla_attention")
        Phase(DECODE).substitute(prog)
        self.assertIs(prog.spec.get_block("mla_attention"), before)

    def test_the_weights_are_the_same_weights(self):
        """Absorbed changes what is COMPUTED, not what is stored — it trades
        FLOPs for KV traffic. A parameter count that moved would mean the
        substitution swapped in a different model."""
        prog = _program("deepseek-ai__DeepSeek-V3")
        d = Deployment(prog).at(input_tokens=512).during(DECODE)
        binding = d.binding()
        from backend.arch.eval.evaluator import Evaluator
        naive = Evaluator(prog, rule=_single_rank_rule(),
                          workload=d.workload).run(binding).total()
        absorbed = Evaluator(d.phase.substitute(prog),
                             rule=_single_rank_rule(),
                             workload=d.workload).run(binding).total()
        self.assertEqual(int(naive.params), int(absorbed.params))
        self.assertNotEqual(int(naive.flops), int(absorbed.flops))

    def test_every_walk_lists_the_files_tensors(self):
        """What is stored does not change with the form: the authored walk,
        the prefill and decode reports (validator, analyzer, compose and
        rendering read these) and the decode checkpoint list the same
        tensors at the same bytes — a dense MLA and a sparse one, FP8."""
        from backend.arch.eval.evaluator import Evaluator
        from backend.arch.eval.projection import project_model

        def stored(analysis):
            return sorted((row.name, tuple(map(int, row.tensor.shape)),
                           row.storage_dtype, int(row.count))
                          for row in analysis.iter_checkpoint())
        for model in ("deepseek-ai__DeepSeek-V3", "zai-org__GLM-5.3"):
            with self.subTest(model=model):
                prog = _program(model)
                d = Deployment(prog).at(batch=1, input_tokens=8)
                authored = Evaluator(prog, rule=_single_rank_rule(),
                                     workload=d.workload).run(d.binding())
                self.assertTrue(any(".kv_b_proj." in row[0]
                                    for row in stored(authored)))
                for phase in ("prefill", DECODE):
                    for walk in (project_model(prog.resolved, input_tokens=8,
                                               phase=phase,
                                               checkpoint=True).analysis,
                                 d.during(phase).checkpoint()):
                        self.assertEqual(stored(walk), stored(authored))
                        self.assertEqual(walk.checkpoint_param_bytes(),
                                         authored.checkpoint_param_bytes())

    def test_the_two_halves_read_the_stored_tensor_once(self):
        """Each per-head GEMM at `kv_b_proj` is billed its share of the
        stored tensor — data and FP8 block scales, tiled once over the
        whole — so the two together read it exactly once."""
        from backend.arch.eval import traffic
        for model in ("deepseek-ai__DeepSeek-V3", "zai-org__GLM-5.3"):
            with self.subTest(model=model):
                an = shared_walk(Deployment(_program(model)).at(
                    batch=1, input_tokens=8).during(DECODE))
                reads = traffic.derive(an).values
                read = {}
                for node in an.program.nodes():
                    if node.tied_to == "kv_b_proj":
                        got = read.setdefault(node.tied_to_uid, [0, 0])
                        got[0] += reads[node.uid].weight_reads
                        got[1] += reads[node.uid].metadata_reads
                self.assertTrue(read)
                for uid, (weight, metadata) in read.items():
                    (row,) = an.tensors_of(uid)
                    rep = row.rep.resident
                    self.assertEqual(
                        float(weight),
                        float(rep.weight_bytes_for(row.tensor.elements)))
                    self.assertEqual(float(metadata),
                                     float(rep.metadata_bytes))

    def test_an_address_with_no_stored_parameter_is_refused(self):
        """Weight holders sharing a `module:` address are one stored tensor;
        without the parameter that declares it there is nothing to read."""
        from backend.arch.loader import SpecLoader
        from backend.arch.eval.program import Program
        from tests.snapshots import resolve_snapshot
        # Its own loader, and NOT the shared layer: this one has a component
        # deleted out of it, so neither it nor what it resolves may be
        # shared with anybody.
        loader = SpecLoader()
        del loader.library_component("mla_attention_absorbed").graph["kv_b_proj"]
        prog = Program.of(resolve_snapshot(
            "deepseek-ai__DeepSeek-V3", loader))
        with self.assertRaisesRegex(ValueError, "store at module 'kv_b_proj'"):
            Phase(DECODE).substitute(prog).nodes()

    def test_a_drafts_decode_form_is_the_draft_the_spec_opted_into(self):
        """GLM-5.3-Flash's NoPE MLA is a draft, so its absorbed form is one
        too, and nothing in the prefill graph calls it: the decode phase
        finds it among the drafts the spec lists, and it is the same
        weights read another way — the stored kv_b_proj run as the two
        per-head absorptions."""
        prog = _program("zai-org__GLM-5.3-Flash")
        swapped = Phase(DECODE).substitute(prog)
        self.assertEqual(_kv_b_views(swapped),
                         {"q_absorbed_nope", "v_absorbed"})
        d = Deployment(prog).at(input_tokens=512)
        self.assertEqual(int(shared_walk(d).total().params),
                         int(shared_walk(d.during(DECODE)).total().params))

    def test_a_draft_is_no_decode_form_of_a_spec_that_does_not_list_it(self):
        """A draft is scoped to the specs that opt into it, at decode as at
        load: a spec that does not list it cannot bind it in."""
        name = "region_softmax_pool_compressor_decode"
        step5 = _program("stepfun-ai__Step-5-Preview").resolved
        dsv3 = _program("deepseek-ai__DeepSeek-V3").resolved
        self.assertIsNotNone(
            step5.declared_by.decode_form(step5.spec.name, name))
        self.assertIsNone(
            dsv3.declared_by.decode_form(dsv3.spec.name, name))


class TestParallelismPolicy(unittest.TestCase):
    """The degree table the layout reads — not a divisor on the binding."""

    def test_tensor_parallelism_uses_the_classified_axes(self):
        from backend.arch.eval.transforms import tp_divides
        prog = _program("deepseek-ai__DeepSeek-V3")
        policy = Parallelism(tp=8).policy(prog, phase=PREFILL)
        self.assertIn("n_heads", tp_divides())
        self.assertEqual(policy["n_heads"], 8)
        self.assertEqual(prog.base_binding["n_heads"],
                         prog.base_binding.raw("n_heads"))

    def test_a_models_own_symbols_are_found(self):
        from backend.arch.eval.transforms import tp_divides
        prog = _program("moonshotai__Kimi-K3")
        self.assertLessEqual({"n_lin_k_heads", "n_lin_v_heads"}, tp_divides())
        policy = Parallelism(tp=8).policy(prog, phase=PREFILL)
        self.assertEqual(policy["n_lin_v_heads"], 8)

    def test_expert_parallelism_shards_copies_not_the_denominator(self):
        prog = _program("deepseek-ai__DeepSeek-V3")
        policy = Parallelism(ep=8).policy(prog, phase=PREFILL)
        self.assertEqual(policy["n_expert_copies"], 8)
        self.assertNotIn("n_experts", policy)
        self.assertEqual(prog.base_binding["n_experts"],
                         prog.base_binding.raw("n_experts"))

    def test_data_parallelism_shards_nothing(self):
        policy = Parallelism(dp=8).policy(phase=PREFILL)
        self.assertTrue(not policy or all(d <= 1 for d in policy.values()))

    def test_dp_attention_owns_the_batch_without_dividing_it(self):
        prog = _program("deepseek-ai__DeepSeek-V3")
        p = Parallelism(dp_attention=8, tp=8)
        self.assertEqual(p.worlds(prog).get("B"), "dp_attention")
        policy = p.policy(prog, phase=PREFILL)
        self.assertNotIn("B", policy)
        self.assertEqual(policy["n_heads"], 8)

    def test_degrees_compose(self):
        prog = _program("deepseek-ai__DeepSeek-V3")
        policy = Parallelism(tp=8, ep=4).policy(prog, phase=PREFILL)
        self.assertEqual(policy["n_heads"], 8)
        self.assertEqual(policy["n_expert_copies"], 4)

    def test_world_specific_tp_degrees(self):
        prog = _program("deepseek-ai__DeepSeek-V3")
        policy = Parallelism(
            tp=8,
            world_degrees={"tp_dense_ffn": 4, "tp_moe_expert": 2},
        ).policy(prog, phase=PREFILL)
        self.assertEqual(policy["n_heads"], 8)
        self.assertEqual(policy["d_ffn"], 4)
        self.assertEqual(policy["d_ffn_expert"], 2)


class TestDeploymentComposes(unittest.TestCase):
    def setUp(self):
        self.d = Deployment(_program("deepseek-ai__DeepSeek-V3")).at(
            batch=2, input_tokens=1024)

    def test_asking_twice_gives_the_same_answer(self):
        """The composition site must have no state to disturb — that is what it
        exists to remove."""
        first = shared_walk(self.d).total()
        shared_walk(self.d.during(DECODE, cache_len=2048))
        shared_walk(self.d.on(tp=8))
        again = shared_walk(self.d).total()
        self.assertEqual((first.flops, first.params),
                         (again.flops, again.params))

    def test_variations_do_not_mutate_the_original(self):
        self.assertEqual(self.d.on(tp=8).config.tp_attention, 8)
        self.assertIsNone(self.d.config)

    def test_run_scoped_dims_are_immutable_and_override_phase(self):
        changed = self.d.with_dims(s_query=7, dp_attention=4)
        self.assertEqual(changed.binding()["s_query"], 7)
        self.assertEqual(changed.binding()["dp_attention"], 4)
        self.assertEqual(self.d.binding()["s_query"], 1024)

    def test_symbolic_is_a_rebinding(self):
        """Not a subclass — the same walk over the workload's counts: the
        value half is a law in the counts, the identity half one in the
        model's names."""
        from dataclasses import replace
        walk = shared_walk(replace(self.d, workload=self.d.workload.symbolic()[0]))
        got = walk.total()
        self.assertFalse(sp.sympify(value_of(got.flops)).is_number)
        self.assertIn("s_pairs", str(expr_of(got.flops)))

    def test_aux_stacks_are_their_own_bucket(self):
        by_stack = shared_walk(self.d).by_stack()
        self.assertIn("", by_stack, "the backbone")
        self.assertIn("mtp", by_stack)
        self.assertGreater(int(by_stack[""].params), int(by_stack["mtp"].params))

    def test_rewrite_is_identity_when_nothing_was_fused(self):
        self.assertTrue(shared_walk(self.d).rewrite.is_identity())

    def test_analyze_uses_layout_propagation(self):
        from backend.arch.layout.propagation import LayoutPropagation

        analysis = shared_walk(self.d)
        self.assertIsInstance(analysis.rule, LayoutPropagation)
        self.assertTrue(analysis.rule.is_trivial)

        sharded = shared_walk(self.d.on(tp=8))
        self.assertIsInstance(sharded.rule, LayoutPropagation)
        self.assertFalse(sharded.rule.is_trivial)


class TestHeadSamplingIsAView(unittest.TestCase):
    """Serving slices the residual to the positions the sampler reads, then
    the head sees that tensor. Binding `s_query` stays the activation length.
    """

    def _head_analysis(self, logit_positions):
        analysis = shared_walk(Deployment(_program("meta-llama__Llama-3.2-1B")).at(
            batch=1, input_tokens=128, logit_positions=logit_positions,
        ))
        head = next(n for n in analysis.program.nodes()
                    if n.boundary_role == "output_head")
        return analysis, head

    def test_head_output_is_the_sampled_axis(self):
        from backend.arch.eval.dim import value_of

        analysis, head = self._head_analysis(1)
        seq = analysis.shapes.values[head.uid].out[1]
        self.assertTrue(seq.is_("s_sampled"))
        self.assertEqual(value_of(seq), 1)

    def test_binding_s_query_is_still_the_prompt(self):
        """The overlay used to write `s_sampled` onto Binding `s_ctx` for the
        head's frame, so `node_dims` reported the sampled length under the
        context name. Cost follows the arriving tensor; the frame does not
        have to lie about the step's query length."""
        from backend.arch.eval.dim import Dim, value_of

        analysis, head = self._head_analysis(1)
        self.assertEqual(value_of(analysis.binding["s_query"]), 128)
        seq = analysis.node_dims.values[head.uid]["s_query"]
        self.assertEqual(value_of(seq), 128)
        if isinstance(seq, Dim):
            self.assertTrue(seq.is_("s_query"))

    def test_head_flops_follow_the_sampled_edge(self):
        from backend.arch.eval.dim import value_of

        one, head_one = self._head_analysis(1)
        four, head_four = self._head_analysis(4)
        self.assertEqual(
            value_of(four.cost.values[head_four.uid].flops),
            4 * value_of(one.cost.values[head_one.uid].flops),
        )

    def test_sample_is_a_graph_node(self):
        analysis, head = self._head_analysis(1)
        sample = next(n for n in analysis.program.nodes()
                      if n.op == "sample")
        self.assertTrue(
            analysis.shapes.values[sample.uid].out[1].is_("s_sampled"))
        self.assertEqual(analysis.cost.values[sample.uid].flops, 0)
        ins = analysis.shapes.values[head.uid].ins
        self.assertTrue(ins and ins[0][1].is_("s_sampled"))


if __name__ == "__main__":
    unittest.main()


class TestIsolationAtACallSite(unittest.TestCase):
    """An isolated component lives where its model calls it. Under that call
    every tensor has the name, role-picked scheme and format it has in the
    whole model — DeepSeek-V3's MLA q_a_proj is FP8 there — and the
    component receives what the call hands it. A model that calls the
    component in several places is evaluated at the call the caller names;
    one that does not call it at all, at the call it takes the place of."""

    def test_tensors_match_the_whole_models(self):
        from backend.arch.eval.transforms import CallSite, analyze_isolated

        resolved = _program("deepseek-ai__DeepSeek-V3").resolved
        first = next(site for site in CallSite.calls(resolved)
                     if site.component == "mla_attention")
        site = CallSite.of(resolved, "mla_attention", call=first.call)
        self.assertEqual((site.address, site.role), ("model.layers.X.self_attn.",
                                                     "attention"))
        whole = {row.name: (row.storage_dtype, row.storage_bytes)
                 for row in shared_walk(Deployment(Program.of(resolved)))
                 .iter_checkpoint()}
        isolated = list(analyze_isolated(resolved, "mla_attention",
                                         site=site).iter_checkpoint())
        self.assertTrue(isolated)
        for row in isolated:
            self.assertEqual(whole[row.name],
                             (row.storage_dtype, row.storage_bytes), row.name)

    def test_a_component_called_in_several_places_needs_its_call_named(self):
        from backend.arch.eval.transforms import CallSite, analyze_isolated

        resolved = _program("deepseek-ai__DeepSeek-V3").resolved
        calls = [site.call for site in CallSite.calls(resolved)
                 if site.component == "mla_attention"]
        self.assertGreater(len(calls), 1)
        with self.assertRaises(ValueError) as refused:
            analyze_isolated(resolved, "mla_attention")
        for call in calls:
            self.assertIn(call, str(refused.exception))

    def test_a_component_the_model_does_not_call_takes_the_named_calls_place(self):
        from backend.arch.eval.transforms import CallSite

        resolved = _program("Qwen__Qwen3-8B").resolved
        spec = resolved.spec
        resolved = resolved.with_spec(spec.model_copy(update={"components": {
            **spec.components,
            "mla_attention": shared_loader().component("mla_attention")}}))
        with self.assertRaisesRegex(ValueError, "does not call"):
            CallSite.of(resolved, "mla_attention")
        (attention,) = (site for site in CallSite.calls(resolved)
                        if site.component == "gqa_qknorm_attention")
        site = CallSite.of(resolved, "mla_attention", call=attention.call)
        self.assertFalse(site.invoked)
        self.assertEqual(site.role, "attention")
        self.assertEqual(site.address, attention.address)
        self.assertEqual(site.formals, attention.formals)
