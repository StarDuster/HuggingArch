"""Per-layer selectors intersect (AND), and schedule exhaustiveness is scoped
to the container the config list drives.

A hybrid backbone crosses two independent axes — DeepSeek-V4 picks attention
from `config.compress_ratios` AND MoE routing from the first `num_hash_layers`
layers. Neither axis alone names a block kind; their INTERSECTION does. A single
selector (the common case) intersects to itself, so pre-existing one-selector
specs are unaffected — asserted here alongside the new AND behaviour and the
span-scoped exhaustiveness that lets a shared schedule array carry a trailing
entry for a sibling stack (V4's MTP `0`).
"""
from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.document import PerLayerAssign
from backend.arch.loader import SpecLoader
from backend.arch.spec_tree import iter_blocks
from backend.arch.validate._schedule_exhaustiveness import run_layer_schedule_check


SPEC = """\
match: {{ model_type: [{model_type}], model_id: [{model_type}] }}
dims:
  d_model: config.hidden_size
  n_layers: config.num_hidden_layers
  n_hash_layers: config.num_hash_layers
blocks:
  a:
    graph: {{ n: {{ op: rmsnorm, in: x, dim: d_model }} }}
    outputs: [n]
  b:
    graph: {{ n: {{ op: rmsnorm, in: x, dim: d_model }} }}
    outputs: [n]
  c:
    graph: {{ n: {{ op: rmsnorm, in: x, dim: d_model }} }}
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
{per_layer}
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: d_model }}, inputs: [layers] }}
"""


def _resolve(model_type: str, per_layer: str, config_extra: dict):
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / f"{model_type}.yaml").write_text(
            SPEC.format(model_type=model_type,
                        per_layer=textwrap.indent(per_layer, " " * 6)))
        loader = SpecLoader(base_dir=str(base))
        config = {"model_type": model_type, "hidden_size": 8, **config_extra}
        resolved = loader.match(config, model_id=model_type)
        return resolved, iter_blocks(resolved)


# Cross-product: `hot` layers split into a prefix (also within the first
# n_hash_layers) vs the rest; everything else is `cold`.
CROSS = """\
assign:
  - { type: hot_prefix, from: config.kinds, is: hot, range: ["0", "n_hash_layers - 1"] }
  - { type: hot_rest,   from: config.kinds, is: hot }
  - { type: cold,       rest: true }
map:
  hot_prefix: { template: a }
  hot_rest:   { template: b }
  cold:       { template: c }
"""

# Single selector, two kinds — the historical shape.
SINGLE = """\
assign:
  - { type: hot,  from: config.kinds, is: hot }
  - { type: cold, rest: true }
map:
  hot:  { template: a }
  cold: { template: b }
"""

# Every in-span value is claimed by a config_list generator (not `rest`), so the
# schedule check has nothing to flag unless a value it does not handle appears
# inside the stack's own span.
FULLY_HANDLED = """\
assign:
  - { type: hot,    from: config.kinds, is: hot }
  - { type: cold,   from: config.kinds, is: cold }
  - { type: filler, rest: true }
map:
  hot:    { template: a }
  cold:   { template: b }
  filler: { template: c }
"""


class TestSelectorIntersection(unittest.TestCase):
    def test_two_selectors_intersect(self):
        _r, groups = _resolve(
            "test_isect", CROSS,
            {"num_hidden_layers": 6, "num_hash_layers": 3,
             "kinds": ["hot", "hot", "cold", "hot", "cold", "hot"]})
        idx = {m.template_name: set(m.layer_indices) for m in groups if m.count}
        self.assertEqual(idx["a"], {0, 1})        # hot AND in first 3
        self.assertEqual(idx["b"], {3, 5})        # hot, after the prefix
        self.assertEqual(idx["c"], {2, 4})        # cold (rest)

    def test_partition_follows_config_across_sizes(self):
        # A differently shaped checkpoint: indices track config, never a baked-in
        # list — the property that fixes V4-Flash.
        _r, groups = _resolve(
            "test_isect_small", CROSS,
            {"num_hidden_layers": 4, "num_hash_layers": 2,
             "kinds": ["hot", "cold", "hot", "cold"]})
        idx = {m.template_name: set(m.layer_indices) for m in groups if m.count}
        self.assertEqual(idx["a"], {0})           # hot AND in first 2
        self.assertEqual(idx["b"], {2})           # hot, after the prefix
        self.assertEqual(idx["c"], {1, 3})

    def test_single_selector_is_unchanged(self):
        # One selector intersects to itself — historical behaviour is byte-exact.
        _r, groups = _resolve(
            "test_isect_one", SINGLE,
            {"num_hidden_layers": 4, "num_hash_layers": 0,
             "kinds": ["hot", "cold", "hot", "cold"]})
        idx = {m.template_name: set(m.layer_indices) for m in groups if m.count}
        self.assertEqual(idx["a"], {0, 2})
        self.assertEqual(idx["b"], {1, 3})


class TestScheduleExhaustivenessSpan(unittest.TestCase):
    def test_trailing_value_outside_container_span_is_ok(self):
        # kinds carries one extra entry beyond n_layers (a sibling stack's slot,
        # like V4's MTP `0`); its value is unhandled here but is out of span.
        resolved, _g = _resolve(
            "test_span_ok", FULLY_HANDLED,
            {"num_hidden_layers": 4, "num_hash_layers": 2,
             "kinds": ["hot", "cold", "hot", "cold", "mtp_only"]})
        errors = run_layer_schedule_check(resolved)
        self.assertEqual(errors, [], f"unexpected: {[str(e) for e in errors]}")

    def test_unhandled_value_within_span_still_fires(self):
        # An unhandled value INSIDE the stack the list drives must still error —
        # `rest:` does not launder it.
        resolved, _g = _resolve(
            "test_span_bad", FULLY_HANDLED,
            {"num_hidden_layers": 4, "num_hash_layers": 2,
             "kinds": ["hot", "warm", "hot", "cold"]})
        errors = run_layer_schedule_check(resolved)
        joined = " ".join(str(e) for e in errors)
        self.assertIn("warm", joined)


# The config lists the layer numbers themselves (MiMo's
# `fullatt_block_indexes`; Qwen3.8's 1-based `ple_layer_ids`).
NUMBERED = """\
assign:
  - { type: listed, from: config.listed }
  - { type: other,  rest: true }
map:
  listed: { template: a }
  other:  { template: b }
"""

ONE_BASED = """\
assign:
  - { type: listed, from: config.listed, base: 1 }
  - { type: other,  rest: true }
map:
  listed: { template: a }
  other:  { template: b }
"""

WITHIN = """\
assign:
  - { type: listed_cold, from: config.kinds, is: cold, within: { from: config.listed, base: 1 } }
  - { type: hot,  from: config.kinds, is: hot }
  - { type: cold, from: config.kinds, is: cold }
map:
  listed_cold: { template: a }
  hot:         { template: b }
  cold:        { template: c }
"""


class TestLayerNumberList(unittest.TestCase):
    def test_list_of_layer_numbers_selects_those_layers(self):
        resolved, groups = _resolve(
            "test_numbered", NUMBERED,
            {"num_hidden_layers": 6, "num_hash_layers": 0,
             "listed": [0, 3, 5]})
        idx = {m.template_name: set(m.layer_indices) for m in groups if m.count}
        self.assertEqual(idx["a"], {0, 3, 5})
        self.assertEqual(idx["b"], {1, 2, 4})
        # A list of layer numbers is not a per-layer schedule.
        self.assertEqual(run_layer_schedule_check(resolved), [])

    def test_one_based_list_counts_from_its_base(self):
        _r, groups = _resolve(
            "test_one_based", ONE_BASED,
            {"num_hidden_layers": 6, "num_hash_layers": 0, "listed": [2, 6]})
        idx = {m.template_name: set(m.layer_indices) for m in groups if m.count}
        self.assertEqual(idx["a"], {1, 5})

    def test_a_number_past_the_stack_is_refused(self):
        with self.assertRaisesRegex(Exception, r"\[99\].*outside this stack's 6 layers"):
            _resolve("test_past_span", NUMBERED,
                     {"num_hidden_layers": 6, "num_hash_layers": 0,
                      "listed": [0, 3, 99]})
        with self.assertRaisesRegex(Exception, r"\[6\].*outside this stack's 6 layers"):
            _resolve("test_past_span_base", ONE_BASED,
                     {"num_hidden_layers": 6, "num_hash_layers": 0,
                      "listed": [2, 7]})

    def test_within_intersects_a_second_config_list(self):
        """A layer kind that is the product of two config lists: the type
        list picks the attention, a list of layer numbers picks the add-on."""
        _r, groups = _resolve(
            "test_within", WITHIN,
            {"num_hidden_layers": 6, "num_hash_layers": 0,
             "kinds": ["hot", "cold", "hot", "cold", "hot", "cold"],
             "listed": [2, 4]})
        idx = {m.template_name: set(m.layer_indices) for m in groups if m.count}
        self.assertEqual(idx["a"], {1, 3})   # cold AND listed (1-based 2, 4)
        self.assertEqual(idx["b"], {0, 2, 4})
        self.assertEqual(idx["c"], {5})

    def test_number_below_the_base_is_refused(self):
        with self.assertRaisesRegex(Exception, "precede its first layer"):
            _resolve("test_below_base", ONE_BASED,
                     {"num_hidden_layers": 6, "num_hash_layers": 0,
                      "listed": [0, 2]})

    def test_base_only_beside_from_alone(self):
        with self.assertRaisesRegex(Exception, "`base`"):
            PerLayerAssign(type="x", **{"from": "config.kinds", "is": "hot"},
                           base=1)
        with self.assertRaisesRegex(Exception, "`base`"):
            PerLayerAssign(type="x", range=["0", "1"], base=1)

    def test_is_needs_from(self):
        with self.assertRaisesRegex(Exception, "`from`"):
            PerLayerAssign(type="x", **{"is": "hot"})


class TestValidatorAcceptsIntersection(unittest.TestCase):
    def test_config_list_plus_range_is_accepted(self):
        a = PerLayerAssign(type="x", **{"from": "config.kinds", "is": "hot"},
                           range=["0", "n_hash_layers - 1"])
        self.assertEqual(a.from_, "config.kinds")
        self.assertEqual(a.range, ["0", "n_hash_layers - 1"])

    def test_rest_cannot_be_combined(self):
        with self.assertRaisesRegex(Exception, "exclusive catch-all"):
            PerLayerAssign(type="x", rest=True, range=["0", "1"])

    def test_no_selector_is_refused(self):
        with self.assertRaisesRegex(Exception, "at least one selector"):
            PerLayerAssign(type="x")


if __name__ == "__main__":
    unittest.main()
