"""An expert bank reads its weights for the rows it receives, not for the
step's own requests.

The saturating read (`RowRoofline.stored_read`) counts the rows the LEAF's
kernel runs on. Those are not the step's tokens for a routed bank: the
dispatch's mean rows per expert name `dp_attention`, because a bank receives
every DP rank's tokens while the attention beside it receives only its own.
While the saturation read the step's count instead, a bank under DP attention
read its weights a factor of `dp_attention` too few.

A bank leaf's operand is `[E_local, M, d]`: the rank's experts and the MEAN
rows each receives, so the rank's routed rows are `M x E_local` and the
model's `B.s.k.E/n_route_targets` — the weighted fraction, not `B.s.k`. LongCat-Flash is the model where the two differ (768 route
targets including 256 zero-compute experts over 512 copies).
"""
from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from backend.inference.row_roofline import RowRoofline
from tests.snapshots import resolve_snapshot

pytestmark = pytest.mark.regression


def _decode(model: str, batch: int, **parallel):
    resolved = resolve_snapshot(model)
    return resolved, analyze(
        resolved, parallel=ParallelConfig.from_dict(parallel) if parallel
        else ParallelConfig(),
        batch_size=batch, phase="decode",
        avg_input_tokens=2048, avg_output_tokens=0, breakdown=False)


def _operand(row):
    shape = row["shape_in_exact"]
    return shape[0] if isinstance(shape[0], list) else shape


def _bank_rows(result):
    """`(row, E_local, M)` for every weight-bearing routed expert-bank leaf:
    its operand is `[E_local, M, d]`."""
    program = result.program
    out = []
    for row in result.per_op_timing:
        if not row.get("weight_bytes_per_rank") or not row.get("moe_block_member"):
            continue
        node = next((n for n in program.nodes()
                     if str(n.aggregation_id) == row["aggregation_id"]), None)
        if node is None or not program.node_is_routed(node):
            continue
        operand = _operand(row)
        out.append((row, int(Fraction(str(operand[0]))),
                    Fraction(str(operand[1]))))
    return out


class TestBankReadsItsRows:
    """The read is `min(rows_received, E_local)` experts' weights, where the
    rows are the ones the bank's own extents state."""

    @pytest.mark.parametrize("model,parallel", [
        ("Qwen/Qwen3-30B-A3B", {}),
        ("moonshotai/Kimi-K3", {}),
        ("moonshotai/Kimi-K3", {"ep_moe": 8}),
        # The disagreement: the bank receives all 8 DP ranks' tokens.
        ("moonshotai/Kimi-K3", {"ep_moe": 8, "dp_attention": 8}),
        ("deepseek-ai/DeepSeek-V3", {"ep_moe": 2, "dp_attention": 2}),
        # Copies are not the destinations: 512 copies, 768 route targets.
        ("meituan-longcat/LongCat-Flash-Omni", {}),
    ])
    @pytest.mark.parametrize("batch", [1, 8, 64])
    def test_experts_read_are_the_rows_received(self, model, parallel, batch):
        _resolved, result = _decode(model, batch, **parallel)
        banks = _bank_rows(result)
        assert banks, f"{model}: no routed expert bank with weights"
        for row, local, per_expert_rows in banks:
            total = int(row["weight_bytes_per_rank"])
            layers = int(row["count_per_step"])
            per_expert = total / (layers * local)
            # The rows ONE expert receives, as the walk states them on this
            # leaf, and the rows the rank's whole bank therefore receives.
            received = per_expert_rows * local
            touched = min(local, math.ceil(received))
            assert int(row["weight_read_bytes_per_rank"]) == pytest.approx(
                touched * per_expert * layers, rel=1e-9), (
                f"{model} {parallel} B={batch} {row['aggregation_id']}: "
                f"{received} rows over {local} experts")

    def test_dp_attention_multiplies_the_rows_a_bank_receives(self):
        """Eight DP ranks of one request each dispatch eight requests' rows
        to every bank; the attention beside it still sees one."""
        _r, alone = _decode("moonshotai/Kimi-K3", 1, ep_moe=8)
        _r, with_dp = _decode("moonshotai/Kimi-K3", 1, ep_moe=8, dp_attention=8)
        one = {r["aggregation_id"]: r for r, _, _ in _bank_rows(alone)}
        eight = {r["aggregation_id"]: r for r, _, _ in _bank_rows(with_dp)}
        shared = set(one) & set(eight)
        assert shared
        for sid in shared:
            rows_one = Fraction(str(_operand(one[sid])[1]))
            rows_eight = Fraction(str(_operand(eight[sid])[1]))
            assert rows_eight == rows_one * 8, sid
            # The read follows the rows it is read for, not the step's
            # requests (which are 1 in both runs).
            assert (int(eight[sid]["weight_read_bytes_per_rank"])
                    == 8 * int(one[sid]["weight_read_bytes_per_rank"])), sid

    def test_longcat_rows_are_the_weighted_fraction(self):
        """`B.s.k.E/n_route_targets`, not `B.s.k`: a third of LongCat's route
        targets hold no expert, so a third of the dispatch reaches none."""
        _resolved, result = _decode("meituan-longcat/LongCat-Flash-Omni", 8)
        banks = _bank_rows(result)
        assert banks
        row, local, per_expert_rows = banks[0]
        frame = row["node_dims"]
        k = int(frame["n_experts_active"])
        targets = int(frame["n_route_targets"])
        assert targets > int(frame["n_experts"]), "not the copies != targets case"
        received = per_expert_rows * local
        assert received == Fraction(8 * k * local, targets)
        assert received < 8 * k, "the zero-compute reserve receives nothing"


class TestStoredRead:
    """The law itself: rows x per-token-active, capped at the total."""

    def test_a_fraction_of_a_row_reads_that_fraction_of_the_bank(self):
        read = RowRoofline.stored_read(rows=Fraction(1, 7),
                                       read_per_row=7000, total=7000,
                                       metadata=700)
        assert (read.weight, read.metadata) == (1000, 100)

    def test_a_whole_row_each_reads_the_whole_bank(self):
        read = RowRoofline.stored_read(rows=3,
                                       read_per_row=7000, total=7000,
                                       metadata=700)
        assert (read.weight, read.metadata) == (7000, 700)


class TestRowsComeFromTheDeclaration:
    """Which axes of the operand are ROWS is the op's own declaration, and a
    leaf that declares no contraction over that operand is refused."""

    def test_every_saturating_leaf_of_the_corpus_declares_its_contraction(self):
        """The refusal is reachable (the next test fires it), so this says the
        corpus does not reach it: every weight-bearing leaf whose read
        saturates prices, under the binding that fuses its GEMM pair."""
        from backend.inference.framework_bindings import get_binding
        from tests.snapshots import SNAPSHOT_DIR
        priced = {}
        for path in sorted(SNAPSHOT_DIR.glob("*.json")):
            resolved = resolve_snapshot(path.stem)
            result = analyze(
                resolved, parallel=ParallelConfig(),
                framework_binding=get_binding("vllm"), batch_size=8,
                phase="decode",
                avg_input_tokens=2048, avg_output_tokens=0, breakdown=False)
            for row in result.per_op_timing:
                ratio = row.get("active_ratio_per_token")
                if (ratio is None or float(ratio) >= 1.0
                        or not row.get("weight_bytes_per_rank")):
                    continue
                priced[row["op"]] = priced.get(row["op"], 0) + 1
        # A fused pair, a primitive, a parameter-reading linear and a YAML op:
        # the four declaration kinds the reader has to answer for.
        assert {"linear", "merged_column_parallel_linear", "linear_param",
                "mxfp4_fused_linear"} <= set(priced), priced

    def test_an_undeclared_contraction_is_refused_not_guessed(self):
        """A leaf whose declared axes do not line up with the walked operand
        is named, with both counts — never priced off a positional guess."""
        from backend.arch.eval.values import Shape
        from backend.inference.row_roofline import kernel_rows

        class _Decl:
            shape_in = ("B", "s_ctx", "in")
            couples_on = ("in",)

        class _Node:
            op = "linear"
            display_path = "mlp.experts.down_proj"
            declaration = _Decl()

        node, shape = _Node(), Shape(ins=((8, 1, 768),))
        assert kernel_rows(None, node, shape) == 8
        with pytest.raises(ValueError, match="walked axes 2"):
            kernel_rows(None, node, Shape(ins=((8, 768),)))

    def test_a_contraction_elsewhere_than_last_is_read_where_it_is(self):
        """The rows are every axis the contraction does not name — not
        everything but the last."""
        from backend.arch.eval.values import Shape
        from backend.inference.row_roofline import kernel_rows

        class _Decl:
            shape_in = ("n_vocab", "B", "s_ctx")
            couples_on = ("n_vocab",)

        class _Node:
            op = "odd_contraction"
            display_path = "somewhere.odd"
            declaration = _Decl()

        assert kernel_rows(None, _Node(), Shape(ins=((99, 4, 16),))) == 64
