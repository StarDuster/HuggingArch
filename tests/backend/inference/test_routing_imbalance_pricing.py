"""What routing imbalance costs, and what it must not change.

A router does not spread tokens evenly over experts. `ep_load_factor` is the
measured skew — the hottest EXPERT's rows over the mean's — and under
synchronous expert parallelism a layer's wall clock is the BUSIEST RANK's, so
the skew lengthens the step. It does not create work: the rows are conserved,
so the walk's global FLOPs and bytes are the same whatever the distribution.

The skew is a workload input (`Workload.ep_load_factor`, the symbol `rho`),
and the busiest rank is priced off its own shapes: a routed tensor's
per-rank rows are the mean times `busiest_rank_load_factor`
(`layout.physical`), so a routed row's per-rank FLOPs and bytes, the rows its
weight read saturates on and the route exchange's payload are that rank's.
Token-side tensors — `permute`'s input, `combine`'s output and routing
weights — are each rank's own tokens and do not move.

That makes two readings of one quantity, and the latency model is only honest
while both are reachable:

* **the mean** — what a rank receives on average. The ranks' means sum to the
  global routed total, which is what the walk's global shapes carry
  (`ExpectedRoutedLoad.is_measure_preserving`);
* **the busiest** — `routed_row_capacity_per_rank`, the mean times
  `busiest_rank_load_factor`. That factor is NOT the user's per-expert `rho`:
  the rank holding the hot expert also holds `E_local - 1` ordinary ones, so
  it is `rho` diluted by them. It is what a rank's per-rank shape carries.

Everything is measured on one model and one deployment, twice, and compared
against itself — so the numbers are differences rather than magnitudes and
nothing here has to be re-pinned when an unrelated cost changes.
"""
from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import resolve_snapshot, snapshot_config

#: One routed model on a deployment that holds its experts apart (`ep_moe`)
#: and gives each rank its own requests (`dp_attention`), so the exchange is
#: materialised and the skew has somewhere to land. H200 x 8 so the weights
#: fit and both phases publish a time.
MODEL = "Qwen/Qwen3-30B-A3B"
STEM = "Qwen__Qwen3-30B-A3B"
GPU, NUM_GPUS = "H200", 8
EP, DP = 8, 8
TOKENS, OUTPUT = 2048, 128

#: The hottest expert carries twice the mean. Any value > 1 states the same
#: law; 2 makes the dilution arithmetic easy to check by hand.
RHO = 2.0


def _estimate(rho: float) -> Dict[str, Any]:
    from backend.inference.capacity import estimate_capacity
    from backend.inference.parallel_config import ParallelConfig

    config = snapshot_config(MODEL)
    return estimate_capacity(
        precision={"preset": "as_shipped"},
        config=config,
        resolved=resolve_snapshot(STEM),
        gpu_sku=GPU, num_gpus=NUM_GPUS,
        avg_input_tokens=TOKENS, avg_output_tokens=OUTPUT,
        parallel=ParallelConfig(ep_moe=EP, dp_attention=DP),
        ep_load_factor=rho,
        selection_mode="theoretical_only",
    )


@pytest.fixture(scope="module")
def priced() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """`(balanced, skewed)` — the same deployment at rho 1 and at RHO."""
    return _estimate(1.0), _estimate(RHO)


def _rows(result: Dict[str, Any], phase: str) -> Dict[str, Dict[str, Any]]:
    key = "per_op_timing" if phase == "prefill" else "per_op_timing_decode"
    return {row["aggregation_id"]: row for row in result[key] or ()}


#: The per-rank columns a row publishes its work and bytes in.
COUNTED = ("flops_per_rank", "access_bytes_per_rank",
           "act_in_bytes_per_rank", "act_out_bytes_per_rank")

PHASES = ("prefill", "decode")


def _busiest(experts: int, local_experts: int, rho: float) -> Fraction:
    """`busiest_rank_load_factor`'s closed form, restated here on purpose.

    One expert carries `rho x M`; the rows are conserved, so the other
    `E - 1` share `(E - rho) x M`, and the rank holding the hot one holds
    `k - 1` of those. Written out rather than imported, because a factor
    compared against its own implementation cannot catch the collapse this
    module exists to catch.
    """
    e, k = experts, local_experts
    skew = Fraction(str(rho))
    return (skew + (k - 1) * Fraction(e - skew, e - 1)) / k


def _moved(balanced: Dict[str, Any], skewed: Dict[str, Any], path: str) -> bool:
    return any(balanced[path].get(c) != skewed[path].get(c) for c in COUNTED)


class TestImbalanceCostsTimeAndNotWork:
    @pytest.mark.parametrize("phase", ("prefill", "decode"))
    def test_the_walks_global_work_is_conserved(self, phase):
        """The global walk is every rank's together: the skew moves rows
        between ranks and creates none."""
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.step_eval import evaluate_step

        totals = []
        for rho in (1.0, RHO):
            analysis = evaluate_step(
                resolve_snapshot(STEM),
                ParallelConfig(ep_moe=EP, dp_attention=DP), phase=phase,
                batch=4, input_tokens=TOKENS, output_tokens=OUTPUT,
                ep_load_factor=rho).analysis
            total = analysis.total()
            totals.append((int(total.flops), int(total.params)))
        assert totals[0] == totals[1]

    @pytest.mark.parametrize("phase", PHASES)
    def test_the_step_takes_longer(self, priced, phase):
        balanced, skewed = priced
        assert (skewed["phases"][phase]["t_sol_s"]
                > balanced["phases"][phase]["t_sol_s"])


class TestTheFactorLandsOnRoutedTensorsAndNowhereElse:
    """Which rows the busiest rank's shapes move, and which they do not.

    The expert bank's leaves run on the rows their rank received; `permute`
    writes them and `combine` reads them; the route exchange moves them.
    Everything else — attention, the dense FFN, the router — runs the same
    work on every rank whatever the routing did, so its row may not move.
    """

    @pytest.mark.parametrize("phase", PHASES)
    def test_only_rows_of_the_moe_block_move(self, priced, phase):
        balanced, skewed = (_rows(r, phase) for r in priced)
        assert set(balanced) == set(skewed)
        moved = {path for path in balanced if _moved(balanced, skewed, path)}
        for path in sorted(moved):
            row = skewed[path]
            assert row.get("moe_block_member") or row["op"] == "all_to_all", (
                f"{phase} {path} moved under routing imbalance outside the "
                "MoE block")
        ops = {skewed[path]["op"] for path in moved}
        assert {"permute", "combine", "all_to_all"} <= ops, ops
        assert ops - {"permute", "combine", "all_to_all"}, (
            f"{phase}: no expert bank leaf moved")
        for path in sorted(set(balanced) - moved):
            assert skewed[path]["t_sol_s"] == balanced[path]["t_sol_s"], path

    @pytest.mark.parametrize("phase", PHASES)
    def test_a_weightless_bank_leaf_scales_exactly_by_it(self, priced, phase):
        """A bank leaf with no resident weights (the gated activation) runs
        on routed rows alone: its FLOPs and bytes are the busiest rank's,
        the mean's times the factor."""
        balanced, skewed = (_rows(r, phase) for r in priced)
        experts, local = _expert_counts()
        expected = float(_busiest(experts, local, RHO))
        weightless = [
            path for path in balanced
            if _moved(balanced, skewed, path)
            and skewed[path]["op"] not in ("permute", "combine", "all_to_all")
            and not int(skewed[path].get("weight_read_bytes_per_rank") or 0)]
        assert weightless, f"{phase}: the bank has no weightless leaf"
        for path in sorted(weightless):
            assert (skewed[path]["flops_per_rank"]
                    == pytest.approx(balanced[path]["flops_per_rank"]
                                     * expected, rel=1e-9)), path

    @pytest.mark.parametrize("phase", PHASES)
    def test_the_resident_weight_read_does_not_move(self, priced, phase):
        """Every rank streams every expert it owns however the tokens fell:
        the saturating `min(rows x read_per_row, total)` is at its cap in
        both runs. Asserted as the cap rather than merely as an equality —
        an unsaturated read WOULD move, and a gate that did not say which
        case it is in would pass by luck."""
        balanced, skewed = (_rows(r, phase) for r in priced)
        banked = [path for path in balanced
                  if _moved(balanced, skewed, path)
                  and int(skewed[path].get("weight_read_bytes_per_rank") or 0)]
        assert banked, f"{phase}: the bank has no weight-bearing leaf"
        for path in sorted(banked):
            read = int(skewed[path]["weight_read_bytes_per_rank"])
            assert read == int(balanced[path]["weight_read_bytes_per_rank"])
            assert read == int(skewed[path]["weight_bytes_per_rank"]), (
                f"{phase} {path}: the read is below the bank, so this cell "
                "is pinning an unsaturated min rather than its cap")

    @pytest.mark.parametrize("phase", PHASES)
    def test_the_route_exchange_moves_the_busiest_ranks_rows(self, priced,
                                                             phase):
        """The all-to-all's buffer is the busiest rank's rows, whole: the
        mean's times the factor, to the next whole row."""
        balanced, skewed = (_rows(r, phase) for r in priced)
        experts, local = _expert_counts()
        expected = _busiest(experts, local, RHO)
        exchanges = [path for path in skewed if "all_to_all" in path]
        assert len(exchanges) == 2, exchanges
        for path in exchanges:
            rows_mean = int(balanced[path]["comm_collective_in_shape"][0])
            rows_busy = int(skewed[path]["comm_collective_in_shape"][0])
            assert rows_busy == math.ceil(rows_mean * expected), path
            row_bytes = Fraction(str(balanced[path]["comm_tensor_bytes"])) / rows_mean
            assert (Fraction(str(skewed[path]["comm_tensor_bytes"]))
                    == rows_busy * row_bytes), path


def _expert_counts() -> Tuple[int, int]:
    """`(experts in the pool, experts on one rank)`, asked of the walk's
    typed MoE facts rather than of the model's config."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    program = Program.of(resolve_snapshot(STEM))
    analysis = Deployment(program).at(batch=1, input_tokens=TOKENS).analyze()
    experts = int(analysis.moe.n_expert_copies)
    assert experts % EP == 0, experts
    return experts, experts // EP


class TestBothReadingsStayReachable:
    """The evidence on the edge still answers both questions.

    The structural half of the gate: the pricing above can only be right
    while `ExpectedRoutedLoad` carries the mean AND the busiest, and while
    the global walk takes the mean and a rank's shape the busiest.
    """

    @staticmethod
    def _dispatch(rho: float) -> Any:
        """`(analysis, permute uid, its routed-load evidence)`."""
        from backend.arch.ops import is_dispatch_boundary
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.step_eval import evaluate_step

        analysis = evaluate_step(
            resolve_snapshot(STEM),
            ParallelConfig(ep_moe=EP, dp_attention=DP),
            phase="prefill", batch=1, input_tokens=TOKENS, output_tokens=0,
            ep_load_factor=rho,
        ).analysis
        for uid, state in analysis.states.values.items():
            node = analysis.program.by_uid(uid)
            if state.workload is not None and is_dispatch_boundary(
                    node.declaration):
                return analysis, uid, state.workload
        raise AssertionError("no dispatch carries routed-load evidence")

    @classmethod
    def _dispatch_evidence(cls, rho: float) -> Any:
        return cls._dispatch(rho)[2]

    def test_the_mean_conserves_and_the_busiest_is_separately_askable(self):
        load = self._dispatch_evidence(RHO)
        # Conservation: the ranks' means sum to the cell's rows — the
        # reading the walk's global shapes carry.
        assert load.is_measure_preserving()
        assert (load.expected_rows_per_expert * load.num_experts
                == load.routed_rows_in_cell)
        # The busiest, from the same object, by the closed form.
        expected = _busiest(load.num_experts, load.local_experts, RHO)
        assert load.busiest_rank_load_factor == expected
        assert 1 < expected < Fraction(str(RHO)), (
            "the hot expert's factor must be diluted by the ordinary "
            "experts sharing its rank, and still cost something")
        assert (load.routed_row_capacity_per_rank
                == load.expected_rows_on_rank(load.local_experts) * expected)

    def test_a_balanced_trace_leaves_both_readings_equal(self):
        load = self._dispatch_evidence(1.0)
        assert load.busiest_rank_load_factor == 1
        assert (load.routed_row_capacity_per_rank
                == load.expected_rows_on_rank(load.local_experts))

    def test_the_per_rank_shape_takes_the_busiest_rank(self):
        """The buffer `permute` writes is `[E_local, M, d_model]` on a rank,
        and its rows are the busiest rank's: the mean's times the factor.
        The global shape keeps the mean, so the ranks' means still sum to
        the dispatch's rows."""
        from backend.arch.eval.dim import value_of
        from backend.arch.layout.physical import physical_shapes

        analysis, uid, load = self._dispatch(RHO)
        experts, rows, _feature = physical_shapes(analysis).values[uid].out
        mean = value_of(load.expected_rows_per_expert)
        assert value_of(experts) == load.local_experts
        assert value_of(rows) == mean * load.busiest_rank_load_factor
        local = value_of(experts) * value_of(rows)
        assert local == value_of(load.routed_row_capacity_per_rank)
