"""A row re-evaluated at another batch reads its scales at the weights' coverage."""
from __future__ import annotations

import unittest

from backend.inference.comm_cost import Fabric
from backend.inference.row_roofline import (
    LawBook, RowCard, build_row_exprs, law_columns)


class TestMetadataTotal(unittest.TestCase):
    def test_scales_follow_the_covered_weights(self):
        book = LawBook()
        row = {"is_comm": False, **law_columns(build_row_exprs(
            count=1, flops=0, act_in=0, act_out=0, demand=100,
            weight_total=1000, metadata_total=80), book)}
        at = book.row_at(
            row, {"B": 1, "s_input": 1, "s_output": 0, "s_prefix": 0,
                  "s_verify": 1, "rho": 1},
            RowCard(None, 1.0, Fabric(None, None, None)))
        # 10% coverage → 100 weight + 8 metadata
        self.assertEqual(at.times.t_memory, 100 + 8)
