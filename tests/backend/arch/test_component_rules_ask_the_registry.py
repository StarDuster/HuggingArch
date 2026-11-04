"""Two component-authoring rules that used to consult their own op lists.

An attention component that stores any tensor must declare its cache
family; a zero-input op may not declare `in:`. Both questions are the
op's own (`has_weight`, `arity`), and the lists that restated them had
already forgotten ops (`linear_param`, `parameter_tensor`).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.spec_parsing import _parse_block_graph


class TestAttentionComponentNeedsACacheFamily(unittest.TestCase):
    def _attention(self, op: str, **node):
        return {
            "role": "attention",
            "graph": {"w": {"op": op, **node}},
            "outputs": ["w"],
        }

    def test_a_buffer_counts_as_a_stored_tensor(self):
        """`rotary_inv_freq_buffer` was not on the old list, so an attention
        component holding only a buffer slipped past the rule."""
        with self.assertRaisesRegex(ValueError, "cache"):
            _parse_block_graph(self._attention("rotary_inv_freq_buffer", dim="d_head // 2"),
                               identity="attn_with_buffer")

    def test_a_bare_parameter_linear_counts_too(self):
        with self.assertRaisesRegex(ValueError, "cache"):
            _parse_block_graph(self._attention("linear_param", **{"in": "x"},
                                               dims={"in": "d_model", "out": "d_model"}),
                               identity="attn_with_linear_param")


class TestZeroInputOpsRefuseInputs(unittest.TestCase):
    def test_parameter_tensor_with_in_is_refused(self):
        """The old check knew `parameter_vector` only."""
        with self.assertRaisesRegex(ValueError, "zero-input"):
            _parse_block_graph({
                "graph": {
                    "w": {"op": "parameter_tensor", "in": "x", "shape": ["d_model", "d_model"]},
                    "y": {"op": "linear_external_weight", "in": ["x", "w"],
                          "dims": {"in": "d_model", "out": "d_model"}},
                },
                "outputs": ["y"],
            }, identity="param_with_in")


if __name__ == "__main__":
    unittest.main()
