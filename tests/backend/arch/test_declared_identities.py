"""Identities the code used to read off names or stand-ins, now asked of
declarations.

  * a model is MoE when its graph declares a dispatch zone (`permute` →
    `combine`), not when `n_experts > 1` happens to bind;
  * a weight under an `ffn_moe` component is the router because its node
    declares `role: router`, the bank because it runs in the zone, and
    every-token work otherwise — the child's name is not consulted;
  * a backbone is hybrid when its caches grow under more than one law,
    not when two display labels differ;
  * an alias adapter is a one-node graph whose op resolves to another
    component — `_type_contracts` and the loader agree on it because both
    ask `BlockGraph.alias_target`;
  * a model card's layer groups are the backbone stack's blocks, its sizes
    the model-level dims the spec declares (`Program.headline`).

Corpus facts are pinned on named snapshots so a spec that loses its
`role: router` fails here, by name, rather than silently relabelling a
router as a shared expert.
"""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import shared_loader
from backend.arch.eval.program import Program
from backend.arch.eval.cache_view import cache_view_from_analysis
from backend.arch.eval.transforms import Deployment
from backend.arch.spec_tree import iter_leaves
from backend.arch.structure import iter_reachable_components
from backend.arch.quant_schemes import SIBLING_KINDS, QUANT_SCHEMES as _SCHEMES
from backend.inference.sharding import moe_child_class
from backend.arch import roles

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _resolved(stem: str):
    config = json.loads((SNAPSHOTS / f"{stem}.json").read_text())["config"]
    resolved = shared_loader().match(config, model_id=stem.replace("__", "/", 1))
    assert resolved is not None, stem
    return resolved


def _classes(stem: str) -> dict:
    """display_path → moe class for every leaf whose role is MoE work."""
    resolved = _resolved(stem)
    program = Program.of(resolved)
    out = {}
    for lt in iter_leaves(resolved):
        if lt.role == "router" or "ffn_moe" in roles.lineage(lt.role):
            out[lt.display_path] = moe_child_class(
                lt.role, program.is_routed(lt.ir_node.uid))
    return out


class TestDispatchZoneIsTheMoeFact(unittest.TestCase):
    def test_a_routed_model_declares_a_zone_and_a_dense_one_does_not(self):
        self.assertTrue(Program.of(_resolved("deepseek-ai__DeepSeek-V3")).has_dispatch_zone)
        self.assertFalse(Program.of(_resolved("Qwen__Qwen3-8B")).has_dispatch_zone)

    def test_the_zone_reaches_the_bank_body_and_its_parameters(self):
        """DSv3 declares the zone on the `experts` invocation; the body's
        leaves inherit it. gpt-oss's expert biases are `parameter_vector`
        sources outside the dataflow span, read by a routed kernel."""
        dsv3 = _classes("deepseek-ai__DeepSeek-V3")
        self.assertEqual(dsv3["mlp.experts.gate_proj"], "moe_routed")
        self.assertEqual(dsv3["mlp.shared_experts.gate_proj"], "moe_shared")
        self.assertEqual(dsv3["mlp.gate.linear"], "moe_router")
        self.assertEqual(dsv3["mlp.gate.e_score_correction_bias"], "moe_router")
        oss = _classes("openai__gpt-oss-20b")
        self.assertEqual(oss["mlp.experts.gate_up_proj"], "moe_routed")
        self.assertEqual(oss["mlp.experts.gate_up_proj_bias"], "moe_routed")
        self.assertEqual(oss["mlp.router"], "moe_router")

    def test_every_moe_model_has_a_router_and_a_bank(self):
        """The two declarations the classifier reads. A component that
        forgets `role: router` shows up here as a model with no router."""
        missing = []
        for snap in sorted(SNAPSHOTS.glob("*.json")):
            resolved = shared_loader().match(json.loads(snap.read_text())["config"],
                                             model_id=snap.stem.replace("__", "/", 1))
            if resolved is None or not Program.of(resolved).has_dispatch_zone:
                continue
            classes = set(_classes(snap.stem).values())
            if not {"moe_router", "moe_routed"} <= classes:
                missing.append((snap.stem, sorted(classes)))
        self.assertEqual(missing, [])


class TestHybridIsAGrowthLawFact(unittest.TestCase):
    def _analysis(self, stem):
        return Deployment(Program.of(_resolved(stem))).at(
            batch=1, input_tokens=8).analyze()

    def test_two_laws_is_hybrid_one_law_factored_is_not(self):
        self.assertTrue(cache_view_from_analysis(self._analysis("openai__gpt-oss-20b")).mixed_growth_laws)
        # 3 dense + 58 MoE layers, one MLA Σ: `growth_formula` factors
        # n_layers over both, and the badge must not call that hybrid.
        self.assertFalse(cache_view_from_analysis(self._analysis("deepseek-ai__DeepSeek-V3")).mixed_growth_laws)


class TestHybridIsTheBackbonesMixers(unittest.TestCase):
    """Hybrid is the backbone running more than one attention / sequence
    mixer; an auxiliary stack's attention (a vision tower, an MTP head) is not
    the backbone's. Whether the cache grows by more than one law is a
    different question (`mixed_growth_laws`)."""

    def test_an_aux_tower_is_not_a_mixer_of_the_backbone(self):
        program = Program.of(_resolved("stepfun-ai__Step-3.7-Flash"))
        self.assertEqual(program.attention_mix, ("GQA", "SWA"))
        self.assertTrue(program.is_hybrid_attention)

    def test_a_linear_mixer_beside_attention_is_hybrid_with_one_growth_law(self):
        program = Program.of(_resolved("Qwen__Qwen3-Coder-Next"))
        self.assertTrue(program.is_hybrid_attention)
        analysis = Deployment(program).at(batch=1, input_tokens=8).analyze()
        self.assertFalse(cache_view_from_analysis(analysis).mixed_growth_laws)

    def test_one_family_over_dense_and_moe_layers_is_not_hybrid(self):
        self.assertFalse(
            Program.of(_resolved("deepseek-ai__DeepSeek-V3")).is_hybrid_attention)


class TestServingReadsSpecDeclaredWeights(unittest.TestCase):
    """A3's cost, pinned: gpt-oss's MXFP4 expert banks are weights, so the
    model is MXFP4-dominant and mixed-precision on the serving side. The
    label is MXFP4, not the generic FP4: `mxfp4.yaml` declares
    `numeric_format: mx_e2m1` (96c2108e3), a different format from NVFP4's
    `fp_e2m1`."""

    def test_gpt_oss_is_fp4_dominant_and_mixed(self):
        analysis = Deployment(Program.of(_resolved("openai__gpt-oss-20b"))).at(
            batch=1, input_tokens=8).analyze()
        self.assertEqual(analysis.dominant_weight_dtype(), "MXFP4")
        self.assertFalse(analysis.weight_dtype_is_uniform())


class TestAliasAdapterIsOneQuestion(unittest.TestCase):
    def test_type_contracts_and_the_loader_agree_on_every_component(self):
        """The loader skips its cache-family check for components with no
        weight leaf of their own, `_type_contracts` for alias adapters of an
        attention component. Both sets must contain every alias adapter."""
        from backend.arch.ops import OP_REGISTRY
        for snap in sorted(SNAPSHOTS.glob("*.json")):
            resolved = shared_loader().match(json.loads(snap.read_text())["config"],
                                             model_id=snap.stem.replace("__", "/", 1))
            if resolved is None:
                continue
            def _component(op):
                try:
                    return resolved.spec.get_block(op)
                except KeyError:
                    return None
            for name, bg in iter_reachable_components(resolved):
                target = bg.alias_target(_component)
                if target is None:
                    continue
                with self.subTest(model=snap.stem, component=name):
                    has_own_weight = any(
                        n.op in OP_REGISTRY and OP_REGISTRY[n.op].has_weight()
                        for n in bg.graph.values())
                    self.assertFalse(has_own_weight,
                                     "an alias adapter holds no weight of its own")
                    self.assertIsNot(target, bg)


class TestSiblingKindsAreDeclared(unittest.TestCase):
    def test_every_scheme_sibling_declares_a_kind(self):
        for scheme in _SCHEMES.values():
            for sibling in scheme.metadata_siblings:
                with self.subTest(scheme=scheme.name, suffix=sibling.suffix):
                    self.assertIn(sibling.kind, SIBLING_KINDS)
                    self.assertEqual(scheme.sibling_kind(sibling.suffix), sibling.kind)

    def test_the_scale_dtype_is_read_off_the_declared_scale(self):
        analysis = Deployment(Program.of(_resolved("deepseek-ai__DeepSeek-V3"))).at(
            batch=1, input_tokens=8).analyze()
        self.assertEqual(analysis.dominant_weight_scale_dtype(), "fp32")


class TestLayerGroupsAreTheBackbones(unittest.TestCase):
    """A model card's layer groups are the backbone's blocks; an MTP head or
    a vision tower is a stack of its own, whatever its `assignments:`."""

    def test_an_mtp_head_is_not_a_backbone_layer(self):
        program = Program.of(_resolved("nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"))
        backbone = program.stack_blocks()
        self.assertEqual(sum(t.count for t in backbone), 108)
        self.assertEqual(
            sum(t.count for t in backbone if t.ffn_type() == "moe"), 48)
        # Its attention and Mamba layers run no FFN at all: not dense ones.
        self.assertFalse([t for t in backbone if t.ffn_type() == "dense"])
        self.assertEqual(len(program.stack_blocks("mtp")), 2)

    def test_a_vision_tower_is_not_a_backbone_layer(self):
        program = Program.of(_resolved("XiaomiMiMo__MiMo-V2.5"))
        self.assertEqual(sum(t.count for t in program.stack_blocks()), 48)
        self.assertEqual(
            {t.name for t in program.stack_blocks("vision")},
            {"mimo_v2_5_vision_block_full", "mimo_v2_5_vision_block_sink"})


class TestHeadlineIsTheDeclaredDims(unittest.TestCase):
    """The model card's sizes are the model-level dims as declared."""

    def test_kv_heads_are_the_declared_count_not_the_largest_block(self):
        # num_key_value_heads = 4; its 39 sliding-window layers bind 8.
        headline = Program.of(_resolved("XiaomiMiMo__MiMo-V2.5")).headline
        self.assertEqual((headline.heads, headline.kv_heads), (64, 4))

    def test_intermediate_is_the_dense_width_not_an_experts(self):
        headline = Program.of(_resolved("deepseek-ai__DeepSeek-V3")).headline
        self.assertEqual(headline.intermediate, 18432)

    def test_a_model_with_no_dense_layer_has_no_dense_width(self):
        # MiniMax-M2.7 binds `d_ffn`, Qwen3-VL-235B's config carries an
        # `intermediate_size`: no layer of either runs a dense FFN.
        for stem in ("MiniMaxAI__MiniMax-M2.7",
                     "Qwen__Qwen3-VL-235B-A22B-Instruct",
                     "nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"):
            with self.subTest(stem):
                headline = Program.of(_resolved(stem)).headline
                self.assertEqual(headline.dense_layers, ())
                self.assertIsNone(headline.intermediate)

    def test_a_dense_ffn_beside_the_experts_is_a_dense_layer(self):
        # LongCat-Flash runs two SwiGLUs of ffn_hidden_size = 12288 in every
        # one of its 28 layers, next to the MoE.
        headline = Program.of(
            _resolved("meituan-longcat__LongCat-Flash-Thinking-2601")).headline
        self.assertEqual(headline.dense_layers, tuple(range(28)))
        self.assertEqual(headline.intermediate, 12288)

    def test_a_shared_expert_is_not_a_dense_layer(self):
        # Step-3.7-Flash's MoE layers run a shared expert outside the
        # dispatch zone; only layers 0-2 run the dense FFN.
        headline = Program.of(_resolved("stepfun-ai__Step-3.7-Flash")).headline
        self.assertEqual(headline.dense_layers, (0, 1, 2))

    def test_dense_layers_are_the_declared_layer_groups(self):
        # DeepSeek-V3: first_k_dense_replace = 3. Step-3.7-Flash:
        # moe_layers_enum starts at 3. MiMo-V2-Flash: moe_layer_freq[0] = 0.
        expected = {"deepseek-ai__DeepSeek-V3": (0, 1, 2),
                    "stepfun-ai__Step-3.7-Flash": (0, 1, 2),
                    "XiaomiMiMo__MiMo-V2-Flash": (0,),
                    "XiaomiMiMo__MiMo-V2.5-Pro": (0,)}
        for stem, layers in expected.items():
            with self.subTest(stem):
                headline = Program.of(_resolved(stem)).headline
                self.assertEqual(headline.dense_layers, layers)
                self.assertIsNotNone(headline.intermediate)

    def test_a_spec_declaring_no_vocab_is_refused_naming_it(self):
        from backend.arch.eval.binding import Binding
        from backend.arch.eval.refusal import SpecRefusal
        resolved = _resolved("Qwen__Qwen3-8B")
        program = copy.copy(Program.of(resolved))
        program.__dict__.pop("headline", None)
        program.__dict__["base_binding"] = Binding(_tracked=True).child_map(
            {k: v for k, v in resolved.dim_values.items() if k != "n_vocab"})
        with self.assertRaises(SpecRefusal) as caught:
            _ = program.headline
        self.assertIn("n_vocab", caught.exception.fix)


class TestAttentionFamiliesReadTheirOwnDims(unittest.TestCase):
    """Each attention family carries the dims its call reads, at their
    values there; a model-level dim no call reads belongs to none."""

    def _analysis(self, stem):
        return Deployment(Program.of(_resolved(stem))).analyze()

    def _families(self, stem):
        return {family.name: dict(family.dims)
                for family in self._analysis(stem).attention_families}

    def test_mla_reads_its_widths_and_no_kv_head_count(self):
        # DeepSeek-V3 config: qk_nope_head_dim 128, qk_rope_head_dim 64,
        # v_head_dim 128, kv_lora_rank 512, num_key_value_heads 128.
        families = self._families("deepseek-ai__DeepSeek-V3")
        self.assertEqual(set(families), {"MLA"})
        mla = families["MLA"]
        self.assertEqual(
            (mla["d_qk_nope"], mla["d_qk_rope"], mla["d_v_head"],
             mla["d_kv_lora"]), (128, 64, 128, 512))
        self.assertNotIn("n_kv_heads", mla)
        self.assertNotIn("d_head", mla)
        # Bound from the config, read by no call.
        headline = Program.of(_resolved("deepseek-ai__DeepSeek-V3")).headline
        self.assertEqual(headline.kv_heads, 128)

    def test_a_hybrids_linear_heads_are_not_mlas(self):
        # Ling-2.6-1T: 70 GLA linear-attention layers read the KV-head
        # count (num_kv_heads_for_linear_attn); its 10 MLA layers read the
        # latent widths and V width 128, and no KV-head count.
        families = self._families("inclusionAI__Ling-2.6-1T")
        self.assertEqual(set(families), {"GLA-Linear", "MLA"})
        self.assertEqual(families["GLA-Linear"]["n_kv_heads"], 64)
        self.assertNotIn("n_kv_heads", families["MLA"])
        self.assertEqual(families["MLA"]["d_v_head"], 128)
        self.assertEqual(families["MLA"]["d_kv_lora"], 512)

    def test_a_layer_assignment_names_the_dim_its_layers_read(self):
        # Step-3.7-Flash's full-attention layers run 64 heads: their
        # assignment binds `n_heads: n_heads_full`, a dim read from the
        # config's `num_attention_heads`, so the family reads that dim — not
        # a rebound `n_heads` with no config field. The sliding-window
        # backbone binds 96.
        analysis = self._analysis("stepfun-ai__Step-3.7-Flash")
        by_name = {family.name: family
                   for family in analysis.attention_families}
        self.assertEqual(dict(by_name["GQA"].dims)["n_heads_full"], 64)
        self.assertEqual(by_name["GQA"].rebound, frozenset())
        self.assertEqual(dict(by_name["SWA"].dims)["n_heads"], 96)


#: The snapshots whose IR roles the params-bound role changed: an attention
#: bound to `mrope`, or the MTP head's inner `pre_norm` bound by params.
_BOUND_ROLE_SNAPSHOTS = (
    "Qwen__Qwen3-VL-235B-A22B-Instruct", "Qwen__Qwen3.5-35B-A3B",
    "Qwen__Qwen3.8-27B", "deepseek-ai__DeepSeek-R1", "deepseek-ai__DeepSeek-V3",
    "deepseek-ai__DeepSeek-V3.1", "deepseek-ai__DeepSeek-V3.2",
    "deepseek-ai__DeepSeek-V3.2-Exp", "zai-org__GLM-4.5", "zai-org__GLM-4.5-Air",
    "zai-org__GLM-5", "zai-org__GLM-5.1", "zai-org__GLM-5.2",
    "zai-org__GLM-5.3-Flash",
)
_MROPE_ATTENTION = ("Qwen__Qwen3-VL-235B-A22B-Instruct", "Qwen__Qwen3.5-35B-A3B",
                    "Qwen__Qwen3.8-27B")


class TestABoundComponentKeepsItsRole(unittest.TestCase):
    """A component bound with params (`{template: gqa_qknorm_attention,
    params: {rope: mrope}}`) carries the role its template declares."""

    def test_an_attention_bound_to_mrope_is_attention(self):
        program = Program.of(_resolved("Qwen__Qwen3-VL-235B-A22B-Instruct"))
        calls = [node for template in program.stack_blocks()
                 for node in template.roots if node.name == "self_attn"]
        self.assertTrue(calls)
        self.assertEqual({node.role for node in calls}, {"attention"})

    def test_the_attention_bucket_is_the_attention_calls(self):
        """Each backbone block's attention FLOPs and params are those of
        its attention calls (found by cache identity, not role), and the
        block's leaves still sum to its total."""
        from backend.arch.eval.cache import identity
        from backend.arch.eval.projection import project_model

        def calls(program, nodes):
            out = []
            for node in nodes:
                ir = node.ir_node
                if ir is not None and ir.children and identity(program, ir):
                    out.append(node)
                else:
                    out.extend(calls(program, node.children or ()))
            return out

        def leaves(node):
            if not node.children:
                return [node]
            return [leaf for child in node.children for leaf in leaves(child)]

        for stem in _MROPE_ATTENTION:
            report = project_model(_resolved(stem), batch=1, input_tokens=2048)
            program = report.analysis.program
            backbone = {t.name for t in program.stack_blocks()}
            for name, _count, block in report.block_results:
                if name not in backbone:
                    continue
                with self.subTest(model=stem, block=name):
                    found = calls(program, block.nodes)
                    self.assertTrue(found)
                    self.assertEqual(block.attention_flops,
                                     sum(n.flops for n in found))
                    self.assertGreater(block.attention_flops, 0)
                    self.assertEqual(block.attention_params,
                                     sum(n.params for n in found))
                    self.assertEqual(
                        block.total_flops,
                        sum(leaf.flops for n in block.nodes for leaf in leaves(n)))
                    self.assertLessEqual(block.attention_flops + block.ffn_flops,
                                         block.total_flops)

    def test_a_per_role_dtype_reaches_a_bound_attention(self):
        """`per_role.attention` quantizes the mrope-bound attention's
        matrices: the role selects the format."""
        import tempfile
        from backend.arch.loader import SpecLoader, overlay_kwargs
        stem = "Qwen__Qwen3-VL-235B-A22B-Instruct"
        text = (PROJECT_ROOT / "backend" / "arch" / "models"
                / f"{stem}.yaml").read_text()
        text += ("\nquant_context:\n  per_role:\n"
                 "    attention: { method: fp8, block_h: 128, block_w: 128 }\n"
                 "    default: { method: none }\n  default_dtype: BF16\n")
        config = json.loads((SNAPSHOTS / f"{stem}.json").read_text())["config"]
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "models").mkdir()
            (Path(d) / "components" / "drafts").mkdir(parents=True)
            (Path(d) / "models" / f"{stem}.yaml").write_text(text)
            resolved = SpecLoader(**overlay_kwargs(d)).match(
                config, model_id=stem.replace("__", "/", 1))
        analysis = Deployment(Program.of(resolved)).analyze()
        rows = [row for row in analysis.tensors()
                if row.name.startswith("model.language_model.layers.")
                and ".self_attn." in row.name and row.name.endswith("proj.weight")]
        self.assertTrue(rows)
        self.assertEqual({(row.storage_dtype, row.selected_by) for row in rows},
                         {("F8_E4M3", "role:attention")})

    def test_stored_dtypes_are_the_checkpoints(self):
        """Every stored tensor of the snapshots whose roles changed keeps the
        dtype its checkpoint header states."""
        from backend.hf.header import SafetensorsHeader
        for stem in _BOUND_ROLE_SNAPSHOTS:
            header = SafetensorsHeader.from_json(json.loads(
                (SNAPSHOTS / f"{stem}.json").read_text())["metadata"]["header"])
            analysis = Deployment(Program.of(_resolved(stem))).analyze()
            checked = 0
            for row in analysis.iter_checkpoint():
                if row.is_buffer:
                    continue
                selection = header.select(row.name)
                if not selection:
                    continue
                checked += 1
                dtypes = set(selection.dtypes)
                with self.subTest(model=stem, tensor=row.name):
                    self.assertIn(row.storage_dtype, dtypes)
            self.assertGreater(checked, 0, stem)


if __name__ == "__main__":
    unittest.main()
