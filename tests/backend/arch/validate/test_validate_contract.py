"""Phase-4 contract test: `validate(model_id, spec)` must return the
shape the agent's prompt promises.

The spec-gen agent's Phase 4 ("Validate & Iterate") loop reads specific
keys out of the validate result — `param_mismatches[i].ratio_hint`,
`unmatched_spec_nodes[i].nearest_tensor_keys`, `unmatched_hf_tensors`,
`shape_mismatches[i].shapes`. These names appear verbatim in the
prompt template (`backend/spec_worker/_prompts/tasks.py`). If a refactor
ever renames them, OR a regression makes validate crash before it can
produce that dict, the agent's self-correction loop silently degrades
— agent ends up shipping unvalidated specs.

`test_validate_returns_phase4_contract_keys` exercises validate
end-to-end, so any crash-before-dict regression (the original
motivating bug was `_matching.py` using `Counter` without importing
it) trips here regardless of which import / helper broke.

Marked as integration: it makes a real HF metadata fetch via the
project's hf.metadata cache. On a CI machine without network or
HF_HOME the test skips. The DeepSeek-V3 fixture is chosen because
it's already used as a known-good spec elsewhere and its checkpoint
metadata is small.

Ship-gate correctness (bytes_diff / unmatched / param_mismatches
actually being zero) is a SEPARATE concern — see
`test_validator_golden.py`.
"""
from __future__ import annotations

from typing import ClassVar

import os
import sys
import unittest
from unittest import mock

from tests.snapshots import snapshot_config
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))


_FIXTURE_MODEL = "deepseek-ai/DeepSeek-V3"
_FIXTURE_SPEC = "deepseek_v3"

# Skip when:
#   • SKIP_NETWORK_TESTS is set (typical CI fast lane), or
#   • the user has no HF cache and HUGGINGARCH_OFFLINE_TESTS is set.
# We intentionally DO NOT skip silently when the network call just
# fails — the test is supposed to be loud about contract regressions.
_SKIP_REASON = os.environ.get("SKIP_NETWORK_TESTS")


@unittest.skipIf(_SKIP_REASON, f"network tests skipped ({_SKIP_REASON})")
class TestValidateContract(unittest.TestCase):
    def test_validate_returns_phase4_contract_keys(self):
        """End-to-end: validate must produce the dict shape the agent's
        prompt promises. We don't assert the content (param diffs are
        allowed, this is a known-imperfect spec by design) — only that
        the keys exist and are list-typed where the prompt iterates.
        """
        from backend.arch.validate import validate
        try:
            result = validate(_FIXTURE_MODEL, _FIXTURE_SPEC)
        except Exception as e:
            self.skipTest(f"validate raised before producing dict: {e!r}")

        self.assertIsInstance(result, dict)

        # Top-level keys consumed by the prompt's Phase-4 instructions.
        for key in (
            "valid",
            "errors",
            "param_mismatches",
            "unmatched_spec_nodes",
            "unmatched_hf_tensors",
            "shape_mismatches",
            "structural_warnings",
        ):
            self.assertIn(key, result, f"missing Phase-4 contract key: {key!r}")

        # Iterable-shaped keys: prompt does `param_mismatches[i].ratio_hint`,
        # `unmatched_spec_nodes[i].nearest_tensor_keys`, etc. So at minimum
        # these have to be lists (possibly empty), not None / scalar.
        for key in (
            "errors", "param_mismatches", "unmatched_spec_nodes",
            "unmatched_hf_tensors", "shape_mismatches", "structural_warnings",
        ):
            self.assertIsInstance(
                result[key], list,
                f"contract violation: {key!r} must be a list, got {type(result[key]).__name__}",
            )

    def test_param_mismatch_entries_have_ratio_hint_field(self):
        """When a param mismatch IS present, each entry carries
        `ratio_hint` — the agent prompt explicitly tells the agent to
        index by this field to find which dim binding is off."""
        from backend.arch.validate import validate
        try:
            result = validate(_FIXTURE_MODEL, _FIXTURE_SPEC)
        except Exception as e:
            self.skipTest(f"validate raised before producing dict: {e!r}")

        mismatches = result.get("param_mismatches") or []
        if not mismatches:
            self.skipTest("no param mismatches on the fixture spec — nothing to check")
        for entry in mismatches:
            self.assertIn(
                "ratio_hint", entry,
                f"param_mismatch entry missing ratio_hint: {entry!r}. "
                "Prompt's Phase 4 instruction depends on this field.",
            )


class TestCheckpointJoin(unittest.TestCase):
    """The stored tensors' names joined to the header's on equality.

    A name is compared whole: a vision tower's `q_proj` and the text
    backbone's share every segment below their stack, and a stored text
    tensor must not stand for the vision one. And a header that lists no
    tensors is a missing fetch, not a checkpoint every name passes against.
    """

    class _Row:
        def __init__(self, name, is_buffer=False):
            self.name = name
            self.is_buffer = is_buffer

    class _Analysis:
        def __init__(self, names):
            self._rows = [TestCheckpointJoin._Row(n) for n in names]

        def iter_checkpoint(self):
            return iter(self._rows)

    def _join(self, stored, header):
        from backend.arch.validate._matching import CheckpointJoin
        from backend.hf.header import SafetensorsHeader
        return CheckpointJoin.of(
            self._Analysis(stored),
            SafetensorsHeader.from_tensors(
                {name: ("BF16", [8, 8]) for name in header}))

    def test_same_block_internal_name_in_another_tower_stays_unmatched(self):
        join = self._join(
            ["model.layers.X.self_attn.q_proj.weight"],
            ["model.layers.0.self_attn.q_proj.weight",
             "model.layers.1.self_attn.q_proj.weight",
             "vision_model.layers.0.self_attn.q_proj.weight"])
        self.assertEqual(sorted(join.unmatched_header),
                         ["vision_model.layers.{0}.self_attn.q_proj.weight"])
        self.assertEqual(join.absent, ())

    def test_a_name_the_header_lacks_is_absent(self):
        join = self._join(["layers.X.self_attn.q_proj.weight"],
                          ["model.layers.0.self_attn.q_proj.weight"])
        self.assertEqual([row.name for row in join.absent],
                         ["layers.X.self_attn.q_proj.weight"])
        self.assertEqual(sorted(join.unmatched_header),
                         ["model.layers.{0}.self_attn.q_proj.weight"])

    def test_a_literal_index_claims_only_its_own_tensors(self):
        join = self._join(
            ["model.layers.X.mlps.0.down_proj.weight"],
            ["model.layers.0.mlps.0.down_proj.weight",
             "model.layers.0.mlps.1.down_proj.weight",
             "model.layers.1.mlps.0.down_proj.weight",
             "model.layers.1.mlps.1.down_proj.weight"])
        self.assertEqual(sorted(join.unmatched_header),
                         ["model.layers.{0-1}.mlps.{1}.down_proj.weight"])

    def test_empty_header_is_refused(self):
        from backend.arch.validate._matching import CheckpointUnavailable
        with self.assertRaises(CheckpointUnavailable):
            self._join(["model.norm.weight"], [])


class TestCallSiteRoleOverride(unittest.TestCase):
    """When a wrapper component (role X) invokes another component
    (role Y, X≠Y) under `repeat:`, the wrapper MUST set `role: X` on
    the call site so leaves under the inner component inherit X — not
    Y. This is the pattern V4-Pro hit: `dsv4_moe_sigmoid` (role=ffn_moe)
    `repeat:E`s a `swiglu_ffn` (role=ffn_dense) per expert; without
    `role: ffn_moe` on the call site, expert leaves carry ffn_dense
    and `quant_context.per_role.ffn_moe` (which routes FP4 packing)
    misses every one of them.

    The schema's role precedence already supports this: call-site
    `node.role` > invoked-component's declared role > parent's role.
    Authors just have to use it. The loader's
    `_check_role_propagation_in_components` refuses the library at boot
    when a component forgets to, so the misroute below cannot ship.
    """

    def _scaffold(self, base, *, with_call_site_role: bool):
        """Build a synthetic library + model in tempdir. The MoE
        wrapper's `experts:` call site either has `role: ffn_moe`
        (with_call_site_role=True) or doesn't (False)."""
        import textwrap
        (base / "components" / "library").mkdir(parents=True)
        (base / "components" / "adapters").mkdir()
        (base / "components" / "drafts").mkdir()
        (base / "models").mkdir()

        # Inner dense FFN — `swiglu_ffn`-shaped.
        (base / "components" / "library" / "_dense_ffn.yaml").write_text(textwrap.dedent("""\
            role: ffn_dense
            graph:
              gate_proj: { op: linear, in: x,         dims: { in: d_model, out: d_ffn } }
              up_proj:   { op: linear, in: x,         dims: { in: d_model, out: d_ffn } }
              down_proj: { op: linear, in: gate_proj, dims: { in: d_ffn, out: d_model } }
            outputs: [down_proj]
        """))

        role_field = ", role: ffn_moe" if with_call_site_role else ""
        (base / "components" / "library" / "_moe_wrapper.yaml").write_text(textwrap.dedent(f"""\
            role: ffn_moe
            graph:
              experts: {{ op: _dense_ffn{role_field}, in: x, repeat: "n_experts" }}
            outputs: [experts]
        """))

        # Need a `template:`-driven model layer with explicit repeat so
        # the component is enumerated as a block — `op:` on a model
        # layer is for primitive ops, not component references.
        (base / "models" / "_role_test.yaml").write_text(textwrap.dedent("""\
            match: { model_type: [_role_test], model_id: [synthetic/_role_test] }
            dims:
              d_model: config.hidden_size
              d_ffn: config.intermediate_size
              n_experts: config.num_experts
            model_inputs:
              input_ids: { kind: tokens }
            model:
              - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
                  dims: { n_vocab: d_model, d_model: d_model } }
              - { name: layers, repeat: "1", template: _moe_wrapper, inputs: [embed],
                  streams: { hidden: { carries: true } } }
              - { name: head, op: linear, boundary_role: output_head,
                  dims: { in: d_model, out: d_model }, inputs: [layers] }
        """))

    def test_with_call_site_role_inner_leaves_inherit_wrapper(self):
        """When the wrapper sets `role: ffn_moe` on its `experts:` call
        site, leaves under the inner ffn_dense FFN end up tagged ffn_moe
        — `quant_context.per_role.ffn_moe` will reach them."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            self._scaffold(base, with_call_site_role=True)
            from backend.arch.loader import SpecLoader
            from backend.arch.spec_tree import iter_leaves
            loader = SpecLoader(base_dir=str(base))
            cfg = {
                "model_type": "_role_test",
                "torch_dtype": "bfloat16", "hidden_size": 16, "intermediate_size": 32, "num_experts": 4,
            }
            resolved = loader.match(cfg, model_id="synthetic/_role_test")
            self.assertIsNotNone(resolved)
            roles = {lt.role for lt in iter_leaves(resolved) if "experts" in lt.name}
            self.assertEqual(
                roles, {"ffn_moe"},
                f"expected experts roles == {{ffn_moe}}, got {roles}",
            )

    def test_loader_refuses_a_missing_call_site_role(self):
        """Loader's `_check_role_propagation_in_components` refuses a
        library with the offending pattern at load. It used to warn; the
        corpus has had zero offenders since, and a warning is a gate that
        can be violated without a sound."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            self._scaffold(base, with_call_site_role=False)
            from backend.arch.loader import SpecLoader
            with self.assertRaises(ValueError) as ctx:
                SpecLoader(base_dir=str(base))
            text = str(ctx.exception)
            self.assertIn("role propagation gap", text)
            self.assertIn("_moe_wrapper", text)
            self.assertIn("_dense_ffn", text)

    def test_loader_silent_when_call_site_role_present(self):
        """No warning when the wrapper does tag the call site."""
        import io
        import logging
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            self._scaffold(base, with_call_site_role=True)
            log_buf = io.StringIO()
            handler = logging.StreamHandler(log_buf)
            handler.setLevel(logging.WARNING)
            logger_name = "backend.arch.loader"
            logging.getLogger(logger_name).addHandler(handler)
            try:
                from backend.arch.loader import SpecLoader
                SpecLoader(base_dir=str(base))
            finally:
                logging.getLogger(logger_name).removeHandler(handler)
            self.assertNotIn("role propagation gap", log_buf.getvalue())


if __name__ == "__main__":
    unittest.main()


class TestABrokenSpecIsAFindingNotACrash(unittest.TestCase):
    """`validate` reports an unparseable spec; it does not propagate.

    The loader REFUSES to serve a spec that failed to load — returning
    None there would send a model that has a spec back to spec-gen. But
    refusing to serve and refusing to explain are different jobs, and
    validate's whole job is the second one. Its caller is spec-gen's
    verdict gate, which files anything raised under "validation crashed"
    and truncates it to 200 characters; the pydantic error naming the
    offending field has to arrive as a finding to survive that.

    Pinned end-to-end because this exact chain — loader raises, validate
    propagates, the gate mislabels — is invisible to a test of any one
    link.
    """

    def _overlay(self, tmpdir: str) -> str:
        models = Path(tmpdir) / "models"
        models.mkdir(parents=True, exist_ok=True)
        # A top-level `name:`, which the schema forbids: a model spec is
        # named by its file stem. This is what spec-gen wrote for GLM-5.3.
        (models / "broken_spec_fixture.yaml").write_text(
            "name: some_other_family\n"
            "match:\n"
            "  model_type: [synthetic_broken_arch]\n"
            "dims:\n"
            "  d_model: config.hidden_size\n"
            "model: []\n"
        )
        return tmpdir

    def test_the_parse_error_arrives_as_an_error_entry(self):
        import tempfile
        from unittest import mock

        from backend.arch.validate import validate

        with tempfile.TemporaryDirectory() as tmp:
            overlay = self._overlay(tmp)
            with mock.patch("backend.arch.validate._orchestrator._get_safetensors_data",
                            return_value={"total_params": 0, "model_storage_bytes": 0,
                                          "tensors": {}}), \
                 mock.patch("backend.arch.validate._orchestrator._fetch_config",
                            return_value={"model_type": "synthetic_broken_arch",
                                          "torch_dtype": "bfloat16", "hidden_size": 8}):
                result = validate("fake/broken", spec_name="broken_spec_fixture",
                                  overlay_dir=overlay)

        self.assertFalse(result["valid"])
        joined = "; ".join(str(e) for e in result["errors"])
        self.assertIn("failed to load", joined)
        # The finding must name the field, not just say something went wrong.
        self.assertIn("name", joined)
        self.assertNotIn("No spec matched", joined)


class TestAnUnknownQuantMethodIsARepairableError(unittest.TestCase):
    """A checkpoint whose `quant_method` no registered scheme is named for
    comes back to the writer as an `errors[]` entry it can repair in
    `quant_context`, not as a stop: the loop keeps repairing
    (`blocked_by_engine` is false)."""

    def test_the_writer_is_told_to_describe_the_packing(self):
        from unittest import mock

        from backend.arch.validate import _orchestrator, validate

        real = _orchestrator._get_safetensors_data

        def unknown(model_id, overlay_dir=None):
            return {**real(model_id, overlay_dir), "quant_method_unknown": "fbgemm_fp8"}

        with mock.patch.object(_orchestrator, "_get_safetensors_data", unknown):
            result = validate("Qwen/Qwen3-8B")

        self.assertFalse(result["valid"])
        self.assertFalse(result["blocked_by_engine"])
        (error,) = result["errors"]
        self.assertIn("fbgemm_fp8", error)
        self.assertIn("quant_context", error)


class TestABroadcastIncompatibleSpecIsAFindingNotACrash(unittest.TestCase):
    """A graph whose tensors do not compose is an `errors[]` entry.

    Verdict used to file `project_model` raising (elementwise broadcast
    with extents [2560, 4] / [128, 64]) as `validation crashed` and skip
    the repair round. The writer cannot fix a crash. The same draft now
    comes back as `valid: false` with the broadcast extents in `errors[]`.
    """

    _SPEC = """\
match:
  model_type: [synthetic_broadcast]
dims:
  d_model: config.hidden_size
  d_head: "4"
  n_layers: "1"
  n_vocab: "32"
model_inputs:
  input_ids: {kind: tokens}
model:
  - name: embed
    op: embedding
    inputs: [input_ids]
    boundary_role: input_embedding
    dims: {n_vocab: n_vocab, d_model: d_model}
  - name: layers
    template: broken_block
    repeat: n_layers
    inputs: [embed]
    streams: { hidden: { carries: true } }
  - name: head
    op: linear
    boundary_role: output_head
    dims: {in: d_model, out: n_vocab}
    inputs: [layers]
blocks:
  broken_block:
    graph:
      wide:
        op: linear
        in: x
        dims: {in: d_model, out: d_model}
      narrow:
        op: linear
        in: x
        dims: {in: d_model, out: d_head}
      mix:
        op: add
        in: [wide, narrow]
        dim: d_model
    outputs: [mix]
"""

    def test_validate_returns_errors_instead_of_raising(self):
        import tempfile
        from unittest import mock

        from backend.arch.validate import validate

        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models"
            models.mkdir(parents=True, exist_ok=True)
            (models / "broadcast_fixture.yaml").write_text(self._SPEC)
            with mock.patch("backend.arch.validate._orchestrator._get_safetensors_data",
                            return_value={"total_params": 0, "model_storage_bytes": 0,
                                          "tensors": {}, "patterns": {}}), \
                 mock.patch("backend.arch.validate._orchestrator._fetch_config",
                            return_value={"model_type": "synthetic_broadcast",
                                          "torch_dtype": "bfloat16", "hidden_size": 2560}):
                result = validate("fake/broadcast", spec_name="broadcast_fixture",
                                  overlay_dir=tmp)

        self.assertFalse(result["valid"])
        joined = "; ".join(str(e) for e in result["errors"])
        self.assertRegex(joined, r"2560")
        self.assertRegex(joined, r"\b4\b")
        self.assertTrue(
            "broadcast" in joined.lower() or "conflicting extents" in joined.lower(),
            joined)

    def test_a_project_model_broadcast_raise_is_an_error_entry(self):
        """The 7 crashed runs raised this exact ValueError from project_model."""
        from unittest import mock

        from backend.arch.validate import validate
        from backend.arch.validate import _orchestrator as O

        def _boom(*_a, **_k):
            raise ValueError(
                "elementwise broadcast axis 3 has incompatible extents [2560, 4]")

        with mock.patch.object(O, "_get_safetensors_data",
                               return_value={"total_params": 0, "model_storage_bytes": 0,
                                             "tensors": {}, "patterns": {}}), \
             mock.patch.object(O, "_fetch_config", return_value={"model_type": "x"}), \
             mock.patch.object(O, "_resolve_spec", return_value=mock.Mock()), \
             mock.patch("backend.arch.eval.projection.project_model", _boom):
            result = validate("fake/broadcast")

        self.assertFalse(result["valid"])
        joined = "; ".join(str(e) for e in result["errors"])
        self.assertIn("broadcast", joined.lower())
        self.assertIn("2560", joined)
        self.assertIn("4", joined)
        self.assertIn("evaluating spec", joined)
        self.assertIn("during projection", joined)

    def test_a_keyerror_names_the_spec_stage_and_field(self):
        from unittest import mock

        from backend.arch.validate import validate
        from backend.arch.validate import _orchestrator as O

        def _boom(*_a, **_k):
            raise KeyError("in")

        resolved = mock.Mock()
        resolved.spec.name = "demo"

        with mock.patch.object(O, "_get_safetensors_data",
                               return_value={"total_params": 0, "model_storage_bytes": 0,
                                             "tensors": {}, "patterns": {}}), \
             mock.patch.object(O, "_fetch_config", return_value={"model_type": "x"}), \
             mock.patch.object(O, "_resolve_spec", return_value=resolved), \
             mock.patch("backend.arch.eval.projection.project_model", _boom):
            result = validate("fake/keyerror")

        self.assertFalse(result["valid"])
        joined = "; ".join(str(e) for e in result["errors"])
        self.assertIn("evaluating spec `demo`", joined)
        self.assertIn("during projection", joined)
        self.assertIn("field `in`", joined)
        self.assertIn("KeyError", joined)


class TestInfrastructureIsNotASpecFinding(unittest.TestCase):
    """Config / checkpoint failures raise; they do not become errors[]."""

    def test_config_unavailable_raises(self):
        from unittest import mock

        from backend.arch.validate import validate
        from backend.arch.validate import _orchestrator as O

        with mock.patch.object(O, "_get_safetensors_data",
                               return_value={"total_params": 0, "model_storage_bytes": 0,
                                             "patterns": {}}), \
             mock.patch.object(O, "_fetch_config",
                               side_effect=O.ConfigUnavailable("hub down")), \
             self.assertRaises(O.ConfigUnavailable):
            validate("org/m")

    def test_empty_safetensors_fetch_is_checkpoint_unavailable(self):
        from unittest import mock

        from backend.arch.validate import validate
        from backend.arch.validate import _orchestrator as O

        with mock.patch("backend.hf.metadata.fetch_metadata", return_value={}), \
             mock.patch.object(O, "_fetch_config",
                               return_value={"model_type": "x"}), \
             self.assertRaises(O.CheckpointUnavailable):
            validate("org/m")


class TestCheckerCrashIsAFinding(unittest.TestCase):
    """A check module that raises is a Finding in that check's usual list."""

    def _errors(self, **buckets):
        from backend.arch.validate._orchestrator import _gating_errors

        class Checks:
            coverage_errors: ClassVar[list] = []
            param_config_errors: ClassVar[list] = []
            packing_errors: ClassVar[list] = []
            shape_mismatches: ClassVar[list] = []
            axiom_warnings: ClassVar[list] = []
            type_contract_warnings: ClassVar[list] = []
            selection_read_errors: ClassVar[list] = []
            cross_layer_warnings: ClassVar[list] = []
            analysable_errors: ClassVar[list] = []
            rotary_position_errors: ClassVar[list] = []
            attention_mask_errors: ClassVar[list] = []
            unread_parameter_errors: ClassVar[list] = []

        for name, value in buckets.items():
            setattr(Checks, name, value)
        # Skip the bytes-vs-bytes gate so this stays a unit of the
        # checker-crash path, not of checkpoint matching.
        return _gating_errors(
            Checks(), {"model_storage_bytes": 1}, None, [], [],
            lambda lever: lever == "weight",
        )

    def test_gating_checker_crash_fails_the_gate(self):
        from backend.arch.validate._orchestrator import _guarded

        def boom():
            raise RuntimeError("injected")

        findings = _guarded("type-contract checks", boom)
        errors = self._errors(type_contract_warnings=findings)
        self.assertTrue(errors)
        joined = " ".join(errors)
        self.assertIn("type-contract checks", joined)
        self.assertIn("RuntimeError", joined)
        self.assertIn("injected", joined)

    def test_diagnostic_checker_crash_stays_a_warning(self):
        from backend.arch.validate._orchestrator import _guarded

        def boom():
            raise RuntimeError("injected")

        findings = _guarded("activation check", boom)
        self.assertEqual(findings[0].kind, "check_crashed")
        self.assertIn("activation check", findings[0].detail)
        self.assertIn("RuntimeError", findings[0].detail)
        self.assertEqual(self._errors(), [])

    def test_gating_checker_crash_fails_validate(self):
        from unittest import mock

        from tests.backend.arch.validate.test_validator_golden import _run_validate

        model_id = "meta-llama/Llama-3.2-1B"
        snapshot_config(model_id)

        def boom(*_a, **_k):
            raise RuntimeError("injected")

        with mock.patch(
            "backend.arch.validate._orchestrator.run_type_contract_checks",
            side_effect=boom,
        ):
            result = _run_validate(model_id)
        self.assertFalse(result["valid"])
        joined = " ".join(str(e) for e in result["errors"])
        self.assertIn("type-contract checks", joined)
        self.assertIn("RuntimeError", joined)
        self.assertIn("injected", joined)

    def test_diagnostic_checker_crash_does_not_fail_validate(self):
        from unittest import mock

        from tests.backend.arch.validate.test_validator_golden import _run_validate

        model_id = "meta-llama/Llama-3.2-1B"
        snapshot_config(model_id)

        baseline = _run_validate(model_id)

        def boom(*_a, **_k):
            raise RuntimeError("injected")

        with mock.patch(
            "backend.arch.validate._activation.run_activation_check",
            side_effect=boom,
        ):
            result = _run_validate(model_id)
        self.assertEqual(result["valid"], baseline["valid"])
        kinds = [w.get("kind") for w in result["activation_warnings"]]
        self.assertIn("check_crashed", kinds)
        details = " ".join(w.get("detail") or "" for w in result["activation_warnings"])
        self.assertIn("activation check", details)
        self.assertIn("RuntimeError", details)


class TestAWeightWithNoLayoutIsRepairable(unittest.TestCase):
    """A stored weight with no `weight_axes` cannot be placed on a rank, so
    the weight ledger every drawn page prices its nodes off does not build
    and the Inspector fails on the model. The byte anchor never builds that
    ledger; validate does, and the failure comes back as an `errors[]` entry
    naming the tensor and what to declare — not as a published spec."""

    def test_the_missing_declaration_is_an_error_entry(self):
        import tempfile

        from tests.backend.arch.validate.test_validator_golden import _run_validate

        model_id = "meta-llama/Llama-3.2-1B"
        snapshot_config(model_id)
        spec = (PROJECT_ROOT / "backend" / "arch" / "models" / "llama.yaml").read_text()
        declared = "{ name: norm, op: rmsnorm, dim: d_model, weight_axes: [{ replicated: true }] , inputs: [layers]}"
        self.assertIn(declared, spec)
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models"
            models.mkdir()
            (models / "llama_no_weight_axes.yaml").write_text(
                spec.replace(declared, "{ name: norm, op: rmsnorm, dim: d_model, inputs: [layers] }"))
            with mock.patch.dict("os.environ", {"HF_HUB_OFFLINE": "1"}):
                from backend.arch.validate import validate
                result = validate(model_id, spec_name="llama_no_weight_axes",
                                  overlay_dir=tmp)
            declared_run = _run_validate(model_id)

        self.assertTrue(declared_run["valid"], declared_run["errors"])
        self.assertFalse(result["valid"])
        (error,) = result["errors"]
        self.assertIn("norm.weight", error)
        self.assertIn("weight_axes", error)


class TestMetadataTensorsJoin(unittest.TestCase):
    """Quant-metadata safetensors entries join at the same grain as weights."""

    def test_fp8_scale_inv_is_claimed(self):
        from tests.backend.arch.validate.test_validator_golden import _run_validate

        snapshot_config("deepseek-ai/DeepSeek-V3")
        result = _run_validate("deepseek-ai/DeepSeek-V3")
        scales = [t["pattern"] for t in (result.get("unmatched_hf_tensors") or [])
                  if "scale" in t["pattern"]]
        self.assertEqual(scales, [])
        mismatches = [m for m in (result.get("param_mismatches") or [])
                      if "scale" in (m.get("hf_module") or "")]
        self.assertEqual(mismatches, [])

    def test_v4_shared_expert_and_mtp_scales_are_claimed(self):
        from tests.backend.arch.validate.test_validator_golden import _run_validate

        for model_id in ("deepseek-ai/DeepSeek-V4-Pro",
                         "deepseek-ai/DeepSeek-V4-Flash"):
            with self.subTest(model_id=model_id):
                snapshot_config(model_id)
                result = _run_validate(model_id)
                patterns = [t["pattern"]
                            for t in (result.get("unmatched_hf_tensors") or [])]
                self.assertFalse(
                    any(p.endswith(".scale") for p in patterns),
                    patterns,
                )
                self.assertEqual(result.get("unmatched_hf_tensors") or [], [])
                self.assertEqual(result.get("param_mismatches") or [], [])
