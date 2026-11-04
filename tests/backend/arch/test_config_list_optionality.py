"""Whether a per-layer config list may be absent is declared on the selector.

`from: config.layer_types` selected nothing when the config lacked the
list and quietly fell through to the sibling `stride:` / `rest:`
generator — the intended behaviour for Qwen3-Next (HF emits either
spelling) and GLM-5.1 (no `indexer_types`), and also exactly what a
misspelt field name looks like. `?? absent` says which one is meant.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.loader import SpecLoader
from backend.arch.spec_tree import iter_blocks

SPEC = """\
match: {{ model_type: [{model_type}], model_id: [{model_type}] }}
dims:
  d_model: config.hidden_size
  n_layers: config.num_hidden_layers
blocks:
  a:
    graph:
      n: {{ op: rmsnorm, in: x, dim: d_model }}
    outputs: [n]
  b:
    graph:
      n: {{ op: rmsnorm, in: x, dim: d_model }}
    outputs: [n]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: d_model, d_model: d_model }} }}
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: {{ hidden: {{ carries: true }} }}
    per_layer:
      assign:
        - {{ type: special, from: "{source}", is: special }}
        - {{ type: plain, rest: true }}
      map:
        special: {{ template: a }}
        plain: {{ template: b }}
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: d_model }}, inputs: [layers] }}
"""


def _resolve(model_type: str, source: str, config_extra: dict):
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / f"{model_type}.yaml").write_text(
            SPEC.format(model_type=model_type, source=source))
        loader = SpecLoader(base_dir=str(base))
        config = {"model_type": model_type, "hidden_size": 8,
                  "num_hidden_layers": 4, **config_extra}
        resolved = loader.match(config, model_id=model_type)
        return resolved, iter_blocks(resolved)


class TestAbsenceIsDeclared(unittest.TestCase):
    def test_a_present_list_selects_its_layers(self):
        _r, groups = _resolve("test_cl_present", "config.kinds",
                              {"kinds": ["special", "plain", "special", "plain"]})
        counts = {m.template_name: m.count for m in groups if m.count}
        self.assertEqual(counts, {"a": 2, "b": 2})

    def test_a_declared_absent_list_falls_through(self):
        _r, groups = _resolve("test_cl_absent_ok", "config.kinds ?? absent", {})
        counts = {m.template_name: m.count for m in groups if m.count}
        self.assertEqual(counts, {"b": 4})

    def test_a_nested_list_is_read_like_a_nested_dim(self):
        """A multimodal config keeps the text schedule under `text_config`;
        `config.text_config.kinds` reads it there, as `dims:` would."""
        nested = {"text_config": {"kinds": ["special", "plain", "plain",
                                            "special"]}}
        _r, groups = _resolve("test_cl_nested", "config.text_config.kinds",
                              nested)
        counts = {m.template_name: m.count for m in groups if m.count}
        self.assertEqual(counts, {"a": 2, "b": 2})
        _r, groups = _resolve("test_cl_nested_opt",
                              "config.text_config.kinds ?? absent", nested)
        counts = {m.template_name: m.count for m in groups if m.count}
        self.assertEqual(counts, {"a": 2, "b": 2})
        # The exhaustiveness gate reads the same nested list: a value no
        # selector handles is reported, not absorbed by `rest:`.
        from backend.arch.validate._schedule_exhaustiveness import (
            run_layer_schedule_check)
        resolved, _groups = _resolve(
            "test_cl_nested_unhandled", "config.text_config.kinds",
            {"text_config": {"kinds": ["special", "other", "plain",
                                       "special"]}})
        self.assertEqual([e.kind for e in run_layer_schedule_check(resolved)],
                         ["schedule_value_unhandled"])

    def test_an_undeclared_absent_list_is_refused(self):
        with self.assertRaisesRegex(Exception, "config.kinds"):
            _resolve("test_cl_absent_bad", "config.kinds", {})


if __name__ == "__main__":
    unittest.main()
