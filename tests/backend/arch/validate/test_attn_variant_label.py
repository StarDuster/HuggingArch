"""The display label `attn_display = attn_variant or cache.family` has no other
gate: attn_variant moves no bytes/shapes, so a DSA layer left labelled "MLA" or
a KDA layer left "GatedDeltaNet" passes every other check. `_check_attn_variant_label`
makes that omission make a sound — but only on unambiguous flavor evidence, so
genuine plain-MLA / real GatedDeltaNet are never flagged.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.loader import SpecLoader
from backend.arch.structure import iter_reachable_components
from backend.arch.validate._axioms import (
    _check_attn_variant_label, _config_declares_kda)

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _resolved(stem: str):
    config = json.loads((SNAPSHOTS / f"{stem}.json").read_text())["config"]
    return SpecLoader().match(config, model_id=stem.replace("__", "/", 1))


def _strip_attn_variant(resolved) -> list[str]:
    stripped = []
    for name, bg in iter_reachable_components(resolved):
        if bg.role == "attention" and bg.attn_variant is not None:
            object.__setattr__(bg, "attn_variant", None)
            stripped.append(name)
    return stripped


class TestAttnVariantLabel(unittest.TestCase):
    def test_dsa_without_attn_variant_fires(self):
        resolved = _resolved("deepseek-ai__DeepSeek-V3.2")
        stripped = _strip_attn_variant(resolved)
        self.assertTrue(stripped, "expected a labelled DSA component to strip")
        findings = _check_attn_variant_label(resolved)
        self.assertTrue(findings)
        self.assertTrue(all(f.kind == "attn_variant_missing" for f in findings))
        self.assertTrue(any("DSA" in f.detail for f in findings))

    def test_labelled_dsa_is_clean(self):
        # As shipped, deepseek_v3_2's DSA component carries attn_variant: DSA.
        self.assertEqual(_check_attn_variant_label(_resolved("deepseek-ai__DeepSeek-V3.2")), [])

    def test_plain_mla_is_clean(self):
        # No DSA ops, no kda_layers -> nothing to flag.
        self.assertEqual(_check_attn_variant_label(_resolved("deepseek-ai__DeepSeek-V3")), [])

    def test_kimi_k3_kda_is_clean_when_labelled(self):
        # config declares kda_layers, but the component now carries attn_variant: KDA.
        resolved = _resolved("moonshotai__Kimi-K3")
        self.assertTrue(_config_declares_kda(
            json.loads((SNAPSHOTS / "moonshotai__Kimi-K3.json").read_text())["config"]))
        self.assertEqual(_check_attn_variant_label(resolved), [])

    def test_kda_without_attn_variant_fires(self):
        resolved = _resolved("moonshotai__Kimi-K3")
        _strip_attn_variant(resolved)  # drop the KDA label from the linear layers
        findings = _check_attn_variant_label(resolved)
        self.assertTrue(any("KDA" in f.detail for f in findings),
                        "kda_layers config + GatedDeltaNet family + no variant must flag KDA")

    def test_config_declares_kda_discriminates(self):
        # Kimi/GLM name KDA layers; generic GatedDeltaNet (Qwen3-Next) uses linear_* keys.
        self.assertTrue(_config_declares_kda(
            {"text_config": {"linear_attn_config": {"kda_layers": [0, 1, 2]}}}))
        self.assertFalse(_config_declares_kda(
            {"text_config": {"linear_num_key_heads": 16, "linear_key_head_dim": 128}}))


if __name__ == "__main__":
    unittest.main()
