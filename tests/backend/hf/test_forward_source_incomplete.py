"""Guard the "incomplete forward source" detector.

Motivating case: `moonshotai/Kimi-K3`'s `KimiK3ForConditionalGeneration` does
`from .modeling_kimi_linear import KimiLinearForCausalLM`, but the auto_map
fetch only pulled the config + vision tower — `modeling_kimi_linear.py`, where
the decoder forward (AttnRes / LatentMoE / attention) actually lives, was never
fetched. The spec-gen agent then guessed the block wiring from config + tower.
The detector must flag that so the prompt tells the agent to go pull the real
decoder source instead of inventing a topology.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


class TestIncompleteForwardSource(unittest.TestCase):
    def _mod(self):
        from backend.hf import forward_source
        return forward_source

    def test_fires_when_decoder_imported_from_unfetched_sibling(self):
        fs = self._mod()
        files = [
            {"path": "modeling_kimi_k3.py",
             "text": "from .modeling_kimi_linear import KimiLinearForCausalLM\n"
                     "class KimiK3ForConditionalGeneration: pass\n"},
            {"path": "configuration_kimi_k3.py", "text": "class KimiK3Config: pass\n"},
        ]
        unresolved = fs._unresolved_decoder_imports(files)
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0]["module"], "modeling_kimi_linear")
        self.assertIn("KimiLinearForCausalLM", unresolved[0]["names"])
        self.assertTrue(fs._incomplete_source_banner(files))

    def test_silent_when_sibling_is_present(self):
        fs = self._mod()
        files = [
            {"path": "modeling_kimi_k3.py",
             "text": "from .modeling_kimi_linear import KimiLinearForCausalLM\n"},
            {"path": "modeling_kimi_linear.py",
             "text": "class KimiLinearForCausalLM: pass\n"},
        ]
        self.assertEqual(fs._unresolved_decoder_imports(files), [])
        self.assertEqual(fs._incomplete_source_banner(files), "")

    def test_silent_on_pretrained_base_import(self):
        """A relative import of a `*PreTrainedModel` base is not the decoder —
        must not trip the flag (the `Model` shape is otherwise too broad)."""
        fs = self._mod()
        files = [{"path": "m.py", "text": "from .foo import BarPreTrainedModel\n"}]
        self.assertEqual(fs._incomplete_source_banner(files), "")

    def test_silent_on_transformers_framework_imports(self):
        """A `local_transformers` source (inspect.getsource of a native module)
        carries relative imports of transformers factories / bases / output
        dataclasses — `AutoModel`, `BaseModelOutputWithPast`, `PreTrainedModel`.
        None name a MISSING decoder; the detector must ignore them, or every
        transformers-native model false-fires `forward_source_incomplete`
        (the Kimi-K2.5 / gemma-4 bug)."""
        fs = self._mod()
        self.assertTrue(fs._is_framework_name("AutoModel"))
        self.assertTrue(fs._is_framework_name("AutoModelForCausalLM"))
        self.assertTrue(fs._is_framework_name("BaseModelOutputWithPast"))
        self.assertTrue(fs._is_framework_name("LlamaPreTrainedModel"))
        self.assertFalse(fs._is_framework_name("KimiLinearForCausalLM"))
        files = [{"path": "m.py", "text": (
            "from ...auto import AutoModel\n"
            "from ...modeling_outputs import BaseModelOutputWithPast\n"
            "from ...modeling_utils import PreTrainedModel\n")}]
        self.assertEqual(fs._unresolved_decoder_imports(files), [])
        self.assertEqual(fs._incomplete_source_banner(files), "")

    def test_banner_is_ast_safe(self):
        """The banner is prepended to source the structural validator parses,
        so it must be pure comments."""
        fs = self._mod()
        files = [{"path": "m.py",
                  "text": "from .modeling_x import XForCausalLM\n"}]
        banner = fs._incomplete_source_banner(files)
        self.assertTrue(banner)
        ast.parse(banner + "\n\nx = 1\n")  # must not raise


class TestStrategyCrashDegrades(unittest.TestCase):
    """The strategy contract is "returns [] on any failure"; a strategy that
    RAISES instead (httpx2.ConnectError on an offline/sandboxed host) must
    degrade to the next strategy — forward source feeds only structural
    warnings, which never gate, so it must never kill a validation run."""

    def test_a_raising_strategy_is_skipped(self):
        from unittest import mock
        from backend.hf import forward_source as fs

        good = [{"path": "m.py", "text": "class X: pass\n"}]
        with mock.patch.object(fs, "_try_snapshot", side_effect=OSError("down")), \
                mock.patch.object(fs, "_try_custom_model", return_value=good), \
                mock.patch.object(fs, "_try_local_transformers", return_value=[]), \
                mock.patch.object(fs, "_try_model_info_cache", return_value=[]), \
                mock.patch.object(fs, "_try_hf_modeling_files", return_value=[]):
            out = fs.describe_forward_source("acme/x", None)
        self.assertEqual(out["source"], "custom_model")
        self.assertIn("class X", out["text"])

    def test_a_failed_repo_fetch_marks_the_stand_in_degraded(self):
        """The repo could not be read, so the transformers module that answers
        is a stand-in for the repo's code; the payload and the text say so."""
        from unittest import mock
        from backend.hf import forward_source as fs

        native = [{"path": "modeling_x.py", "text": "class X: pass\n"}]
        with mock.patch.object(fs, "_try_snapshot", return_value=[]), \
                mock.patch.object(fs, "_try_custom_model", return_value=[]), \
                mock.patch.object(fs, "_try_hf_modeling_files",
                                  side_effect=OSError("offline")), \
                mock.patch.object(fs, "_try_local_transformers", return_value=native):
            out = fs.describe_forward_source("acme/x", None)
        self.assertEqual(out["source"], "local_transformers")
        self.assertIn("OSError: offline", out["degraded"])
        self.assertTrue(out["text"].startswith("# DEGRADED SOURCE"))
        ast.parse(out["text"])

    def test_a_repo_without_code_is_not_degraded(self):
        from unittest import mock
        from backend.hf import forward_source as fs

        native = [{"path": "modeling_x.py", "text": "class X: pass\n"}]
        with mock.patch.object(fs, "_try_snapshot", return_value=[]), \
                mock.patch.object(fs, "_try_custom_model", return_value=[]), \
                mock.patch.object(fs, "_try_hf_modeling_files", return_value=[]), \
                mock.patch.object(fs, "_try_local_transformers", return_value=native):
            out = fs.describe_forward_source("acme/x", None)
        self.assertEqual((out["source"], out["degraded"]), ("local_transformers", ""))

    def test_the_snapshot_refresh_never_stands_in_for_the_repo(self):
        """A refresh that cannot read the repo fails; it does not freeze the
        transformers module as the model's forward."""
        from unittest import mock
        from backend.hf import forward_source as fs
        from backend.hf import snapshot

        with mock.patch.object(fs, "_try_custom_model", return_value=[]), \
                mock.patch.object(fs, "_try_hf_modeling_files",
                                  side_effect=OSError("offline")), \
                mock.patch.object(fs, "_try_local_transformers",
                                  side_effect=AssertionError("stand-in used")), \
                self.assertRaises(OSError):
            snapshot._fetch_forward_source_raw("acme/x", {"model_type": "llama"})

    def test_every_strategy_raising_yields_the_empty_payload(self):
        from unittest import mock
        from backend.hf import forward_source as fs

        with mock.patch.object(fs, "_try_snapshot", side_effect=OSError("down")), \
                mock.patch.object(fs, "_try_custom_model", side_effect=OSError("down")), \
                mock.patch.object(fs, "_try_local_transformers", side_effect=OSError("down")), \
                mock.patch.object(fs, "_try_model_info_cache", side_effect=OSError("down")), \
                mock.patch.object(fs, "_try_hf_modeling_files", side_effect=OSError("down")):
            out = fs.describe_forward_source("acme/x", None)
        self.assertEqual(out["source"], "")
        self.assertEqual(out["files"], [])


if __name__ == "__main__":
    unittest.main()
