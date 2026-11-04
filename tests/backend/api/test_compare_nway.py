"""N-way /compare diff helpers: each field row carries a `values` list aligned
to the model order + a `same` flag. Regression for the 2-model -> 2..5 change.

Unit-level (no network / no HF fetch): exercises the diff-widening helpers and
the CompareRequest validator directly.
"""
from __future__ import annotations

import sys
import unittest
import copy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.api.routes.analyze import (
    _all_same,
    _metric_pairs,
    _modality_diffs,
)
from backend.api._models import (
    CompareRequest,
    MODALITY_REGISTRY,
    _checked_modalities,
)


class TestCompareNWay(unittest.TestCase):
    def test_modality_registry_refuses_an_unknown_modality(self):
        """A registry snapshot with `speech` is refused while it loads: no
        model layer can declare that aux kind, so its card is unreachable."""
        entries = copy.deepcopy(MODALITY_REGISTRY)
        entries["speech"] = copy.deepcopy(entries["audio"])
        with self.assertRaisesRegex(
                ValueError,
                r"modality 'speech' is not an `aux_kind` any stack can "
                "declare"):
            _checked_modalities(entries)

    def test_modality_registry_refuses_an_unregistered_dim(self):
        """A registry snapshot with `Mela` and `Stra` rows is refused while
        it loads: both names point to no registered dim, so every lookup for
        those rows would return None."""
        entries = copy.deepcopy(MODALITY_REGISTRY)
        entries["audio"] = {
            **entries["audio"],
            "dims": [*entries["audio"]["dims"],
                     ("Mela", "Mel bins"), ("Stra", "Stride / hop")],
        }
        with self.assertRaisesRegex(
                ValueError,
                r"the 'audio' card names dim\(s\) \['Mela', 'Stra'\], "
                "which backend/arch/dims.yaml does not register"):
            _checked_modalities(entries)

    def test_all_same(self):
        self.assertTrue(_all_same([1, 1, 1]))
        self.assertTrue(_all_same(["mla", "mla"]))
        self.assertFalse(_all_same([1, 2, 1]))
        # a missing side (None) counts as a difference
        self.assertFalse(_all_same([1, None, 1]))

    def test_metric_pairs_nway(self):
        dashboards = [
            {"main_total_params": 100, "num_layers": 32, "attention_type": "MLA"},
            {"main_total_params": 200, "num_layers": 32, "attention_type": "GQA"},
            {"main_total_params": 100, "num_layers": 32, "attention_type": "MLA"},
        ]
        rows = _metric_pairs([{"dashboard": d} for d in dashboards])
        by_key = {r["key"]: r for r in rows}
        # values aligned to model order, length == model count
        self.assertEqual(by_key["main_total_params"]["values"], [100, 200, 100])
        self.assertFalse(by_key["main_total_params"]["same"])
        # non-distinguishing row flagged `same`
        self.assertEqual(by_key["num_layers"]["values"], [32, 32, 32])
        self.assertTrue(by_key["num_layers"]["same"])
        # rows no model has are dropped
        self.assertNotIn("vocab_size", by_key)
        # every row's values length matches the model count
        for r in rows:
            self.assertEqual(len(r["values"]), 3)

    def test_metric_rows_are_quantities_of_the_model(self):
        # No FLOPs row (a FLOPs figure belongs to a batch and length); the KV
        # rows come from each model's composition, with the dtype priced at.
        def totals(n, dtype):
            held = {"weight_bytes": n,
                    "storage_by_dtype": [{"dtype": dtype, "bytes": n}]}
            return {"lm": held, "with_encoders": held}
        results = [
            {"dashboard": {"total_flops": 9, "weight_totals": totals(4, "BF16")},
             "composition": {"kv_native_dtype": "BF16", "totals": {
                 "kv_per_token_bytes": 2048.0, "kv_constant_bytes": 0.0,
                 "kv_dtypes": [{"dtype": "BF16", "source": "native"},
                               {"dtype": "F8_E4M3", "source": "declared"}]}}},
            {"dashboard": {"total_flops": 7, "weight_totals": totals(3, "F32")},
             "composition": {"kv_native_dtype": "F32", "totals": {
                 "kv_per_token_bytes": 4096.0, "kv_constant_bytes": 64.0,
                 "kv_dtypes": [{"dtype": "F32", "source": "native"}]}}},
        ]
        by_key = {r["key"]: r for r in _metric_pairs(results)}
        self.assertFalse([k for k, r in by_key.items() if r["kind"] == "flops"])
        self.assertEqual(by_key["kv_per_token_bytes"]["values"], [2048.0, 4096.0])
        # Every dtype the cache is priced at, not only the native default.
        self.assertEqual(by_key["kv_dtypes"]["values"], ["BF16 · F8_E4M3", "F32"])
        self.assertFalse(by_key["storage_by_dtype"]["same"])

    def test_params_are_the_main_models_and_bytes_carry_both_scopes(self):
        """Total / Active params are the main model's in every view (no MTP,
        no encoder; the single view's headline), whatever weight scope the
        page shows; MTP and encoder parameters are their own rows. Weight
        bytes are the language model's (`values`) with the whole
        checkpoint's beside them (`values_with_encoders`)."""
        results = [
            {"dashboard": {
                "weight_totals": {"lm": {"weight_bytes": 60},
                                  "with_encoders": {"weight_bytes": 70}},
                "main_total_params": 28, "main_active_params": 9,
                "aux_total_params": 2, "encoder_total_params": 5}},
            {"dashboard": {
                "weight_totals": {"lm": {"weight_bytes": 60},
                                  "with_encoders": {"weight_bytes": 60}},
                "main_total_params": 28, "main_active_params": 9,
                "aux_total_params": 2, "encoder_total_params": 0}},
        ]
        by_key = {r["key"]: r for r in _metric_pairs(results)}
        self.assertEqual(by_key["main_total_params"]["label"], "Total params")
        self.assertEqual(by_key["main_active_params"]["label"], "Active params")
        self.assertEqual(by_key["main_active_params"]["values"], [9, 9])
        for key in ("main_total_params", "main_active_params"):
            self.assertNotIn("values_with_encoders", by_key[key])
        self.assertNotIn("total_params", by_key)
        self.assertEqual(by_key["weight_bytes"]["values"], [60, 60])
        self.assertEqual(by_key["weight_bytes"]["values_with_encoders"], [70, 60])
        self.assertEqual(by_key["aux_total_params"]["label"], "MTP params")
        self.assertEqual(by_key["aux_total_params"]["values"], [2, 2])
        self.assertEqual(by_key["encoder_total_params"]["values"], [5, 0])

    def test_modality_diffs_nway(self):
        modality = next(iter(MODALITY_REGISTRY), None)
        if modality is None:
            self.skipTest("no registered modality")
        key0 = MODALITY_REGISTRY[modality]["dims"][0][0]
        dims_list = [{key0: 14}, {key0: 16}, {}]
        dashboards = [
            {"encoder_stacks": {"tower": modality},
             "encoder_params": {"tower": 100}},
            {"encoder_stacks": {"visual": modality},
             "encoder_params": {"visual": 200}},
            {},
        ]
        out = _modality_diffs(dims_list, dashboards)
        m = next((x for x in out if x["name"] == modality), None)
        self.assertIsNotNone(m)
        self.assertEqual(m["present"], [True, True, False])
        self.assertEqual(m["params"], [100, 200, None])
        for row in m["rows"]:
            self.assertEqual(len(row["values"]), 3)

    def test_a_modality_is_the_encoders_declared_kind_not_a_group_name(self):
        # An MTP head is an aux group too; only an encoder stack is shown,
        # and under the modality its stack declares, whatever it is named.
        modality = next(iter(MODALITY_REGISTRY), None)
        if modality is None:
            self.skipTest("no registered modality")
        dashboards = [
            {"aux_params": {modality: 7, "mtp": 3},
             "encoder_stacks": {}, "encoder_params": {}},
            {"aux_params": {"visual": 5},
             "encoder_stacks": {"visual": modality},
             "encoder_params": {"visual": 5}},
        ]
        out = _modality_diffs([{}, {}], dashboards)
        (m,) = [x for x in out if x["name"] == modality]
        self.assertEqual(m["present"], [False, True])
        self.assertEqual(m["params"], [None, 5])

    def test_request_validator(self):
        # blanks stripped; 2..5 accepted
        self.assertEqual(CompareRequest(models=["a", "b", ""]).models, ["a", "b"])
        self.assertEqual(len(CompareRequest(models=["a", "b", "c", "d", "e"]).models), 5)
        with self.assertRaises(ValueError):     # pydantic ValidationError
            CompareRequest(models=["a"])            # < 2
        with self.assertRaises(ValueError):
            CompareRequest(models=["a", "b", "c", "d", "e", "f"])  # > 5


if __name__ == "__main__":
    unittest.main()


class TestCompareResolvesWithModelId(unittest.TestCase):
    def test_compare_passes_model_id_to_match(self):
        """Regression: /compare must resolve each spec WITH model_id so custom:/
        overlay-only models bind (else their column is blank). Mirrors the
        per-model /analyze path."""
        import asyncio
        import json
        from unittest import mock
        import backend.api.routes.analyze as A

        captured = []

        class FakeLoader:
            def match(self, cfg, model_id=None, header=None):
                captured.append(model_id)
                return None

        def fake_analyze(sub):
            return {"success": True, "dashboard": {"total_params": 1}, "model_info": {}}

        req = CompareRequest(models=["custom:foo", "meta-llama/Llama-3.2-3B"])
        with mock.patch.object(A, "SpecLoader", FakeLoader), \
             mock.patch.object(A, "analyze_payload", fake_analyze), \
             mock.patch("backend.analyzer.fetch_config",
                        lambda m, overlay_dir=None: {"model_type": "x"}):
            out = json.loads(asyncio.run(A.compare(req)).body)

        self.assertEqual(captured, ["custom:foo", "meta-llama/Llama-3.2-3B"])
        self.assertTrue(out.get("success"))
