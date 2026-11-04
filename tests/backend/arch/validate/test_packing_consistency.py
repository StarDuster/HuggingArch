"""Packing / quant-scheme consistency check.

The byte anchor compares only the aggregate `Σ(count × width)`, which is
packing-agnostic: a 2× logical-count error cancelled by a ½× dtype-width error
leaves the byte sum untouched. `run_packing_consistency_check` decomposes the
product and anchors each factor (stored element count, stored dtype) against the
safetensors header, so a compensating packing error can't hide.

Tests synthesize checkpoint rows and pass them as an analysis stub.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))


from backend.arch.schema.quant import QuantContext
from backend.arch.validate._packing_consistency import (
    run_packing_consistency_check,
    run_quant_method_check,
    _canon_dtype,
)


def _row(name, dtype, storage_bytes, method=None, is_metadata=False,
         count=1):
    return SimpleNamespace(
        name=name,
        storage_dtype=dtype,
        storage_bytes=storage_bytes,
        method=method,
        is_metadata=is_metadata,
        count=count,
    )


class _Analysis:
    def __init__(self, rows):
        self._rows = rows

    def iter_checkpoint(self):
        return iter(self._rows)


def _header(stored):
    """A header holding `{name: (shape, dtype, count)}`: `count` tensors of
    that name, its first numeric segment taking indices 0 … count-1."""
    from backend.hf.header import SafetensorsHeader
    tensors = {}
    for name, (shape, dtype, count) in stored.items():
        for i in range(count):
            tensors[name.replace("{}", str(i), 1).replace("{}", "0")] = (dtype, shape)
    return SafetensorsHeader.from_tensors(tensors)


_W_ELEMS = 3072 * 3584
_S_ELEMS = 3072 * 224
_HEADER = _header({
    "layers.{}.ffn.experts.{}.w1.weight": ([3072, 3584], "I8", 1),
    "layers.{}.ffn.experts.{}.w1.scale": ([3072, 224], "F8_E8M0", 1),
})


def _run(rows):
    return run_packing_consistency_check(_Analysis(rows), _HEADER)


class TestPackingConsistency(unittest.TestCase):
    def test_correct_fp4_expert_passes(self):
        rows = [
            _row("layers.X.ffn.experts.X.w1.weight", "I8", _W_ELEMS, method="fp4"),
            _row("layers.X.ffn.experts.X.w1.scale", "F8_E8M0FNU", _S_ELEMS,
                 is_metadata=True),
        ]
        self.assertEqual(_run(rows), [])

    def test_compensating_count_error_fires(self):
        rows = [
            _row("layers.X.ffn.experts.X.w1.weight", "I8", _W_ELEMS // 2,
                 method="fp4"),
            _row("layers.X.ffn.experts.X.w1.scale", "F8_E8M0FNU", _S_ELEMS,
                 is_metadata=True),
        ]
        errs = _run(rows)
        self.assertTrue(errs)
        self.assertTrue(any("element-count mismatch" in e.detail for e in errs))

    def test_wrong_scheme_scale_fingerprint_fires(self):
        rows = [
            _row("layers.X.ffn.experts.X.w1.weight", "I8", _W_ELEMS, method="int8"),
            _row("layers.X.ffn.experts.X.w1.scale", "F16", _S_ELEMS,
                 is_metadata=True),
        ]
        errs = _run(rows)
        self.assertTrue(any("Stored-dtype mismatch" in e.detail and "scale" in e.detail
                            for e in errs))

    def test_full_precision_dtype_divergence_is_ignored(self):
        header = _header({"lm_head.weight": ([32000, 4096], "F16", 1)})
        rows = [_row("lm_head.weight", "BF16", 32000 * 4096 * 2)]
        self.assertEqual(
            run_packing_consistency_check(_Analysis(rows), header), [])

    def test_packed_leading_axis_is_carried_by_count(self):
        experts, out_dim, in_dim = 512, 4096, 8192
        header = _header({
            "mtp.layers.{}.mlp.experts.gate_up_proj":
                ([experts, out_dim, in_dim], "BF16", 1),
        })
        rows = [_row(
            "mtp.layers.X.mlp.experts.gate_up_proj",
            "BF16",
            out_dim * in_dim * 2,
            count=experts,
        )]
        self.assertEqual(
            run_packing_consistency_check(_Analysis(rows), header), [])

    def test_repeated_tensor_uses_header_count(self):
        layers, out_dim, in_dim = 92, 4096, 8192
        header = _header({
            "model.layers.{}.proj.weight": ([out_dim, in_dim], "BF16", layers),
        })
        rows = [_row(
            "model.layers.X.proj.weight",
            "BF16",
            out_dim * in_dim * 2,
            count=layers,
        )]
        self.assertEqual(
            run_packing_consistency_check(_Analysis(rows), header), [])

    def test_e8m0_alias(self):
        self.assertEqual(_canon_dtype("F8_E8M0FNU"), _canon_dtype("F8_E8M0"))
        self.assertNotEqual(_canon_dtype("I8"), _canon_dtype("F8_E4M3"))


class TestUnknownQuantMethod(unittest.TestCase):
    """A checkpoint whose quant_method names no registered scheme is a
    finding the writer repairs in quant_context, not a stop."""

    def test_a_spec_naming_no_method_is_told_how_to_describe_it(self):
        for label, context in (("absent", None), ("empty", QuantContext())):
            with self.subTest(context=label):
                (finding,) = run_quant_method_check(context, "fbgemm_fp8")
                self.assertEqual(finding.kind, "quant_method_unknown")
                self.assertEqual(finding.extra["quant_method"], "fbgemm_fp8")
                self.assertIn("fbgemm_fp8", finding.detail)
                self.assertIn("fp8", finding.detail)

    def test_a_named_method_is_left_to_the_byte_gates(self):
        for label, context in (
                ("per_role fp8", QuantContext(per_role={"default": {
                    "method": "fp8", "block_h": 128, "block_w": 128}})),
                ("override none", QuantContext(overrides=[{
                    "tensors_re": "lm_head", "dtype": "BF16",
                    "method": "none"}]))):
            with self.subTest(context=label):
                self.assertEqual(
                    run_quant_method_check(context, "fbgemm_fp8"), [])

    def test_a_registered_method_is_no_finding(self):
        self.assertEqual(run_quant_method_check(QuantContext(), None), [])


if __name__ == "__main__":
    unittest.main()


class TestQuantRoleGranularity(unittest.TestCase):
    """A deployment chooses weight precision by role and stack, so within one
    role and stack the checkpoint stores every weight matrix at one precision.
    A role the checkpoint splits is refused, naming the finer role to
    declare. (Every shipped snapshot passing it is
    `test_validator_golden.TestCoverageGate`.)"""

    def test_a_split_role_is_refused_with_the_repair(self):
        """An override pinning Kimi-K2.5's routed `down_proj` to BF16 beside
        the INT4 `gate_proj` / `up_proj` splits the `ffn_moe` role."""
        from backend.arch.eval.program import Program
        from backend.arch.eval.transforms import Deployment
        from backend.arch.schema.quant import OverrideEntry
        from backend.arch.validate._packing_consistency import (
            run_quant_role_check)
        from tests.snapshots import resolve_snapshot, shared_walk

        resolved = resolve_snapshot("moonshotai__Kimi-K2.5")
        quant = resolved.spec.quant_context.model_copy(deep=True)
        quant.overrides.append(OverrideEntry(
            tensors_re=r"mlp\.experts\.X\.down_proj\.weight$",
            dtype="BF16"))
        spec = resolved.spec.model_copy(update={"quant_context": quant})
        analysis = shared_walk(Deployment(Program.of(resolved.with_spec(spec))).at(
            batch=1, input_tokens=8))
        (finding,) = run_quant_role_check(analysis)
        self.assertEqual(finding.kind, "quant_role_mixed_precision")
        self.assertEqual((finding.extra["role"], finding.extra["stack"],
                          finding.extra["dtypes"]),
                         ("ffn_moe", "backbone", ["BF16", "INT4"]))
        self.assertIn("down_proj", finding.detail)
        self.assertIn("refines 'ffn_moe'", finding.detail)

    def test_a_finer_role_with_its_own_rule_is_quiet(self):
        """Kimi-K2.5's BF16 shared experts sit in `ffn_moe_shared`, which
        refines `ffn_moe` and has its own `per_role` entry."""
        from backend.arch.eval.transforms import Deployment
        from backend.arch.validate._packing_consistency import (
            run_quant_role_check)
        from tests.snapshots import shared_program, shared_walk

        analysis = shared_walk(Deployment(shared_program(
            "moonshotai__Kimi-K2.5")).at(batch=1, input_tokens=8))
        self.assertEqual(run_quant_role_check(analysis), [])

    def test_a_layer_scope_spanning_a_node_s_whole_schedule_is_still_a_pin(
            self):
        """The layer scope must not become a way around the gate. Scoped to
        every layer the node runs at, it says nothing a bare name predicate
        did not: the node is stored one way throughout, its siblings
        another, and a finer role is still the declaration that is
        missing."""
        from backend.arch.eval.program import Program
        from backend.arch.eval.transforms import Deployment
        from backend.arch.schema.quant import OverrideEntry, OverrideSelector
        from backend.arch.validate._packing_consistency import (
            run_quant_role_check)
        from tests.snapshots import resolve_snapshot, shared_walk

        resolved = resolve_snapshot("moonshotai__Kimi-K2.5")
        quant = resolved.spec.quant_context.model_copy(deep=True)
        quant.overrides.insert(0, OverrideEntry(
            select=OverrideSelector(
                layers={"range": ["0", "n_layers - 1"]},
                tensors_re=r"mlp\.experts\.X\.down_proj\.weight$"),
            dtype="BF16"))
        spec = resolved.spec.model_copy(update={"quant_context": quant})
        analysis = shared_walk(Deployment(Program.of(resolved.with_spec(spec))).at(
            batch=1, input_tokens=8))
        (finding,) = run_quant_role_check(analysis)
        self.assertEqual(finding.kind, "quant_role_mixed_precision")
        self.assertEqual((finding.extra["role"], finding.extra["dtypes"]),
                         ("ffn_moe", ["BF16", "INT4"]))
        self.assertIn("a finer role", finding.detail)

    def test_a_role_expressed_split_across_layers_stays_quiet(self):
        """GLM-5.3-Flash stores `o_proj` BF16 in its linear-attention layers
        and FP8 in its full-attention ones, and says so with roles
        (`attention_linear` beside `attention`). Two nodes, each stored one
        way, each with its own `per_role` entry — which is the declaration
        the gate asks for, so it has nothing to say."""
        from backend.arch.eval.transforms import Deployment
        from backend.arch.validate._packing_consistency import (
            run_quant_role_check)
        from tests.snapshots import shared_program, shared_walk

        analysis = shared_walk(Deployment(shared_program(
            "zai-org__GLM-5.3-Flash")).at(
                batch=1, input_tokens=8))
        self.assertEqual(run_quant_role_check(analysis), [])

    def test_one_node_split_across_its_own_layers_passes(self):
        """Step-5's shape: the routed experts of three of 61 layers are BF16
        and the rest FP8, with the same role, stack, component and graph.
        No assignment of roles to nodes reproduces that, so the gate's
        question — could a finer role have expressed this? — answers no, and
        the layer-scoped row is the declaration."""
        from backend.arch.eval.program import Program
        from backend.arch.eval.transforms import Deployment
        from backend.arch.schema.quant import OverrideEntry, OverrideSelector
        from backend.arch.validate._packing_consistency import (
            run_quant_role_check)
        from tests.snapshots import resolve_snapshot, shared_walk

        resolved = resolve_snapshot("deepseek-ai__DeepSeek-V3")
        context = resolved.spec.quant_context
        row = OverrideEntry(
            select=OverrideSelector(
                layers={"range": ["n_layers - 3", "n_layers - 1"]},
                tensors_re=r"mlp\.experts\.X\.(gate|up|down)_proj\.weight$"),
            dtype="BF16")
        spec = resolved.spec.model_copy(update={
            "quant_context": context.model_copy(update={
                "overrides": [row, *context.overrides]})})
        analysis = shared_walk(Deployment(Program.of(resolved.with_spec(spec))).at(
            batch=1, input_tokens=8))
        self.assertEqual(run_quant_role_check(analysis), [])

    def test_a_decode_twin_reading_a_tensor_at_another_format_is_refused(self):
        """DeepSeek-V4's one-token compressor (`decode_variant`) holds the
        same `wkv` the prefill compressor does. Without the role its prefill
        node carries it reads the BF16 tensor as FP8 attention."""
        from backend.arch.eval.transforms import Deployment
        from backend.arch.loader import SpecLoader
        from backend.arch.validate._packing_consistency import (
            run_phase_format_check)
        from backend.arch.eval.program import Program
        from tests.snapshots import resolve_snapshot, shared_program

        # Its own loader, and NOT the shared layer: a role is deleted out of
        # this component, so nothing resolved against it may be shared.
        loader = SpecLoader()
        twin = loader.library_component("dsv4_compressor_overlap_decode")
        twin.graph["wkv"].role = None
        deployment = Deployment(Program.of(resolve_snapshot(
            "deepseek-ai__DeepSeek-V4-Flash", loader))).at(
                batch=1, input_tokens=1)
        prefill = deployment.checkpoint()
        decode = deployment.during("decode").checkpoint()
        findings = run_phase_format_check(prefill, decode)
        self.assertEqual(
            {f.extra["tensor"] for f in findings},
            {"layers.X.attn.compressor.wkv.weight",
             "layers.X.attn.indexer.compressor.wkv.weight"})
        self.assertIn("decode_variant", findings[0].detail)
        self.assertEqual(run_phase_format_check(
            *(Deployment(shared_program(
                "deepseek-ai__DeepSeek-V4-Flash")).at(batch=1, input_tokens=1)
              .during(phase).checkpoint() for phase in ("prefill", "decode"))),
            [])

    def test_a_derived_weight_is_held_to_its_source_tensors_format(self):
        """Absorbed MLA keeps the stored `kv_b_proj` and runs it as the
        per-head `q_absorbed_nope` / `v_absorbed` (`module: kv_b_proj`), so it
        resolves through the rules that select it in either form — Ling's
        MTP override names `kv_b_proj` alone. A decode form whose
        `kv_b_proj` resolves to another format than the prefill form's is
        refused."""
        from backend.arch.eval.transforms import Deployment
        from backend.arch.loader import SpecLoader
        from backend.arch.validate._packing_consistency import (
            run_phase_format_check)
        from backend.arch.eval.program import Program
        from tests.snapshots import resolve_snapshot, shared_program

        model = "inclusionAI__Ling-2.6-1T"
        deployment = Deployment(shared_program(model)).at(
            batch=1, input_tokens=1)
        decode = deployment.during("decode").checkpoint()
        rows = {}
        for row in decode.tensors():
            node = decode.program.by_uid(row.uid)
            if node.name == "kv_b_proj":
                rows.setdefault(decode.program.stack_of(node), set()).add(
                    (row.name, row.storage_dtype))
        self.assertEqual(rows["mtp"], {
            ("model.layers.X.attention.kv_b_proj.weight", "BF16")})
        backbone = [rows[stack] for stack in rows if stack != "mtp"]
        self.assertEqual(backbone, [{
            ("model.layers.X.attention.kv_b_proj.weight", "F8_E4M3")}])
        self.assertEqual(
            run_phase_format_check(deployment.checkpoint(), decode), [])

        # Its own loader again, for the same reason: this one has a role
        # rewritten on a library component.
        loader = SpecLoader()
        twin = loader.library_component("mla_attention_dense_absorbed")
        twin.graph["kv_b_proj"].role = "router"
        deployment = Deployment(Program.of(resolve_snapshot(
            model, loader))).at(batch=1, input_tokens=1)
        findings = run_phase_format_check(
            deployment.checkpoint(), deployment.during("decode").checkpoint())
        self.assertEqual({f.extra["tensor"] for f in findings},
                         {"model.layers.X.attention.kv_b_proj.weight"})

    def test_glm5_fp8_selects_its_bf16_tensors_by_role(self):
        """`glm_moe_dsa_fp8` (no snapshot of its own) over GLM-5.2's config
        with the FP8 quantization_config: the indexer's weights_proj and the
        router gate are BF16 by role, the rest of attention FP8, and both
        phases pass the gates."""
        from backend.arch.eval.program import Program
        from backend.arch.eval.transforms import Deployment
        from backend.arch.validate._packing_consistency import (
            run_phase_format_check, run_quant_role_check)
        from tests.snapshots import shared_loader, snapshot_config

        config = dict(snapshot_config("zai-org__GLM-5.2"))
        config["quantization_config"] = {
            "quant_method": "fp8", "fmt": "e4m3",
            "weight_block_size": [128, 128]}
        resolved = shared_loader().resolve_by_name("glm_moe_dsa_fp8", config)
        deployment = Deployment(Program.of(resolved)).at(
            batch=1, input_tokens=1)
        prefill = deployment.checkpoint()
        decode = deployment.during("decode").checkpoint()
        stored = {row.name: row.logical_dtype for row in prefill.tensors()}
        self.assertEqual(
            stored["model.layers.X.self_attn.indexer.weights_proj.weight"],
            "BF16")
        self.assertEqual(stored["model.layers.X.mlp.gate.weight"], "BF16")
        self.assertEqual(stored["model.layers.X.self_attn.indexer.wk.weight"],
                         "F8_E4M3")
        self.assertEqual(run_quant_role_check(prefill, decode), [])
        self.assertEqual(run_phase_format_check(prefill, decode), [])
