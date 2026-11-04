"""Guard the activation soft-fidelity check's classification logic.

The FFN activation (silu / gelu / custom) is param-free and cost-negligible, so
no numeric anchor catches a wrong one. `config.hidden_act` is the author's
declared ground truth; the check compares the spec's expanded FFN gate op
against it. Kimi-K3 is the motivating miss: `hidden_act = "situ"` (a custom
`SituAndMul`) modeled as `silu`.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


class TestActivationClassification(unittest.TestCase):
    def _mod(self):
        from backend.arch.validate import _activation
        return _activation

    def test_config_names_resolve_to_the_op_that_computes_them(self):
        """A config's `hidden_act` names one function. transformers ACT2FN:
        `gelu` is the erf form, `gelu_pytorch_tanh` / `gelu_new` the tanh
        approximation (a different op), `swish` is silu, `relu2` squared
        ReLU. A name no op declares (a custom activation the spec can only
        approximate) resolves to nothing."""
        op_for = self._mod()._op_for_config_act
        self.assertEqual(op_for("silu"), "silu")
        self.assertEqual(op_for("swish"), "silu")
        self.assertEqual(op_for("gelu"), "gelu")
        self.assertEqual(op_for("gelu_pytorch_tanh"), "gelu_tanh")
        self.assertEqual(op_for("gelu_new"), "gelu_tanh")
        self.assertEqual(op_for("gelu_python"), "gelu")        # erf, in Python
        self.assertEqual(op_for("gelu_accurate"), "gelu_tanh")  # tanh form
        self.assertEqual(op_for("relu2"), "relu_squared")
        self.assertEqual(op_for("situ"), "situ")
        self.assertIsNone(op_for("quick_gelu"))
        self.assertIsNone(op_for("xielu"))
        self.assertIsNone(op_for(None))

    def test_a_config_name_declared_by_two_ops_is_refused_at_registration(self):
        """The reverse map is built as ops register; a second op claiming
        `silu` is an error there, and the registry is left as it was."""
        import dataclasses
        from backend.arch.ops_base import OP_REGISTRY, register_op, op_for_config_name
        clash = dataclasses.replace(OP_REGISTRY["gelu"], name="clash_probe",
                                    config_names=("silu",))
        with self.assertRaisesRegex(ValueError, "already declares"):
            register_op(clash)
        self.assertNotIn("clash_probe", OP_REGISTRY)
        self.assertEqual(op_for_config_name("silu"), "silu")

    def test_the_language_config_is_where_the_spec_binds_d_model(self):
        """A multimodal config keeps the language model's fields in a sub
        config; the spec says which by binding `d_model` there, so a field
        stated at both levels is read where the backbone's is, not by a
        probe order."""
        from backend.arch.dim_resolution import language_config
        config = {"hidden_act": "gelu", "hidden_size": 16,
                  "text_config": {"hidden_act": "situ", "hidden_size": 8}}
        nested = {"d_model": "config.text_config.hidden_size"}
        self.assertEqual(language_config(nested, config),
                         ("text_config", config["text_config"]))
        self.assertEqual(language_config({"d_model": "config.hidden_size"}, config),
                         ("", config))
        # `??` is read in its own order, as dims resolve it.
        either = {"d_model": "config.hidden_size ?? config.text_config.hidden_size"}
        self.assertEqual(language_config(either, config["text_config"] | {}),
                         ("", config["text_config"]))
        self.assertEqual(language_config(either, {"text_config": {"hidden_size": 8}}),
                         ("text_config", {"hidden_size": 8}))
        m = self._mod()
        fields = language_config(nested, config)[1]
        self.assertEqual(m._config_hidden_act(fields), "situ")
        self.assertEqual(m._config_hidden_act({"hidden_activation": "gelu"}), "gelu")
        self.assertIsNone(m._config_hidden_act({"no_act_here": 1}))

    def test_a_d_model_reading_no_config_field_is_refused_with_the_fix(self):
        from backend.arch.dim_resolution import language_config
        with self.assertRaisesRegex(ValueError, "Bind d_model"):
            language_config({"d_model": '"4096"'}, {"hidden_size": 4096})

    def test_no_hidden_act_no_warning(self):
        """Without a declared activation there's nothing to check against."""
        import copy
        from tests.snapshots import shared_loader, snapshot_config
        config = copy.deepcopy(snapshot_config("Qwen/Qwen3-8B"))
        resolved = shared_loader().match(config, model_id="Qwen/Qwen3-8B")
        config.pop("hidden_act")
        self.assertEqual(self._mod().run_activation_check(resolved, config), [])

    def test_router_scoring_dig_and_map(self):
        m = self._mod()
        # the router scoring nonlinearity under its vendor spellings
        self.assertEqual(m._config_scoring_func({"scoring_func": "sigmoid"}), "sigmoid")
        self.assertEqual(
            m._config_scoring_func({"moe_router_activation": "sigmoid"}), "sigmoid")
        self.assertIsNone(m._config_scoring_func({"topk_method": "greedy"}))
        # config scoring string → expected spec op (attention `softmax` is NOT
        # the router op — router softmax is `softmax_act`)
        self.assertEqual(m._SCORING_TO_OP["softmax"], "softmax_act")
        self.assertEqual(m._SCORING_TO_OP["sigmoid"], "sigmoid")
        self.assertEqual(m._SCORING_TO_OP["sqrtsoftplus"], "softplus")


if __name__ == "__main__":
    unittest.main()


class TestGdnGateActFollowsConfig(unittest.TestCase):
    """vLLM and sglang build the GDN RMSNormGated with
    `activation = config.output_gate_type` (silu when absent, `swish` = silu).
    The library GDN components take it as `gate_act` and declare the field
    (`config: [output_gate_type]`), so a config naming a different gate than
    the bound one is a refusal that names the binding to write."""

    MODEL = "Qwen/Qwen3.5-35B-A3B"

    def _check(self, gate):
        import copy
        from tests.snapshots import shared_loader, snapshot_config
        from backend.arch.validate._activation import run_param_config_check
        config = copy.deepcopy(snapshot_config(self.MODEL))
        if gate is not None:
            config.setdefault("text_config", config)["output_gate_type"] = gate
        resolved = shared_loader().match(config, model_id=self.MODEL)
        return run_param_config_check(resolved, config)

    def test_a_config_without_the_field_binds_the_default(self):
        self.assertEqual(self._check(None), [])

    def test_swish_is_silu(self):
        self.assertEqual(self._check("swish"), [])

    def test_a_sigmoid_gate_left_at_silu_is_refused_with_the_fix(self):
        findings = self._check("sigmoid")
        self.assertTrue(findings)
        for f in findings:
            self.assertEqual(f.kind, "param_config_mismatch")
            self.assertEqual(f.extra["bound"], ["silu"])
            self.assertIn("params: { gate_act: sigmoid }", f.detail)
