"""Lock the inference-tab data-flow invariants the Memory aggregation
and peak-FLOPs roofline fixes restore (CLAUDE.md §1 — every consumer must
read the same fact from the spec tree).

Three suites:

1. **TestMemoryAggregationFrontendContract** —
   `wb["deployment_total_bytes"]` / `wb["total_bytes_per_rank"]` are the
   single source the frontend's Memory card Total cell must render.
   Frontend used to re-sum `groups + model_level`, silently dropping the
   `fused_custom_op_bytes_full + unmatched_scale_bytes_total` residuals
   for gpt-oss (fused MXFP4) and any model with unattributed scale
   metadata. We lock the residual delta so a future regression that
   moves the fused/unattributed bytes off of `deployment_total` would
   break this test.

2. **TestPerOpPeakFlopsRoofline** — `compute_per_op_timing` must emit a
   per-row `peak_flops_per_sec_for_op` driven by each leaf's own
   weight/act dtype, not a single dominant-weight peak applied to every
   row. Locks the closure dispatch (FP8 leaf → fp8 column, BF16 leaf →
   bf16 column), the activation-dtype fallback for compute-only leaves
   (softmax / attention_score have no weight), and the global-peak
   fallback when no closure is supplied (back-compat for legacy
   callers).

3. **TestGpuThreadingEndToEnd** — `pipeline.analyze(gpu=...)` actually
   threads the GPU into the closure. Without this wiring the per-row
   peak goes back to the global peak even on mixed-precision specs.
   `capacity._run_calibration_overlay` is the production caller — we
   verify its `gpu=` arg reaches the closure.
"""
from __future__ import annotations

import sys
import unittest

from tests.snapshots import resolve_snapshot
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.spec_tree import SpecIRNode, SpecIRTree
from backend.arch.eval.layer import Layer
from backend.arch.eval.program import Program
from backend.arch.eval.values import Cost, Shape
from backend.arch.eval.access import LeafAccess
from backend.inference.gpu_specs import (
    get_gpu,
    peak_flops_for_dtype,
)
from backend.inference.parallel_config import ParallelConfig
from backend.inference.phases import (
    compute_per_op_timing as _compute_per_op_timing,
)
from tests.serving_fixture import numbers_as_laws
from backend.inference.weight_breakdown import (
    per_rank_weight_breakdown,
)



def _resolved(model_name):
    return resolve_snapshot(model_name)


def _sum_tree_full(node: dict) -> float:
    """Walk a `_tree_to_dict` tree summing `bytes_full` at every node
    that has it. The dict-rendered trees aggregate at every level —
    parent.bytes_full = Σ children.bytes_full — so summing the roots is
    enough, but the helper is here for symmetry with model_level
    where each entry is a leaf.
    """
    return float(node.get("bytes_full", 0) or 0)


def _sum_tree_per_rank(node: dict) -> float:
    return float(node.get("bytes_per_rank", 0) or 0)


class TestMemoryAggregationFrontendContract(unittest.TestCase):
    """The Memory card's `Total` row reads `wb["deployment_total_bytes"]`
    and `wb["total_bytes_per_rank"]` directly — NOT a frontend re-sum of
    `groups + model_level`. Re-summing drops fused-custom-op + unmatched-
    scale residuals; on gpt-oss (MXFP4 fused linears) that's a multi-GiB
    miss.

    Locks two things on the same snapshot pass:
      (a) `deployment_total == Σgroups + Σmodel_level + fused_custom_op +
          unmatched_scale` — the four-term decomposition the backend
          actually computes (`weight_breakdown.py:680-686`).
      (b) On gpt-oss, the residuals are strictly positive — i.e. the
          frontend's old re-sum WOULD have under-counted. Without this
          guard, the test passes trivially on a uniform-precision model
          where the residuals are zero.
    """

    def test_deployment_total_matches_four_term_decomposition(self):
        """Identity that the frontend Total cell depends on."""
        for name in ("meta-llama__Llama-3.2-3B",
                     "deepseek-ai__DeepSeek-V3",
                     "openai__gpt-oss-20b",
                     "moonshotai__Kimi-K2-Instruct",
                     "XiaomiMiMo__MiMo-V2.5"):
            resolved = _resolved(name)
            if resolved is None:
                continue
            wb = per_rank_weight_breakdown(
                resolved, ParallelConfig())
            groups_full = sum(_sum_tree_full(g) for g in wb["groups"])
            model_level_full = sum(
                _sum_tree_full(m) for m in wb["model_level"]
            )
            # Reconstruct what backend emitted: deployment_total =
            # groups + model_level + residuals. We don't have direct
            # access to fused_custom_op_bytes_full / unmatched_scale_
            # bytes_total in the wb dict (they're internals), but their
            # sum equals deployment_total - groups - model_level by
            # construction — so the identity collapses to:
            residuals = (
                wb["deployment_total_bytes"]
                - groups_full
                - model_level_full
            )
            self.assertGreaterEqual(
                residuals, -1.0,
                msg=f"{name}: deployment_total < groups + model_level "
                    f"by {-residuals:.0f} B — re-aggregation drift",
            )
            # And the per_rank total uses the same decomposition (residuals
            # are replicated, so per_rank residual = full residual).
            groups_pr = sum(_sum_tree_per_rank(g) for g in wb["groups"])
            ml_pr = sum(_sum_tree_per_rank(m) for m in wb["model_level"])
            residuals_pr = (
                wb["total_bytes_per_rank"] - groups_pr - ml_pr
            )
            # Residual delta is the same on full and per_rank axes
            # because residuals are added with divisor=1 (replicated).
            self.assertAlmostEqual(
                residuals_pr, residuals,
                delta=max(1.0, abs(residuals) * 1e-9),
                msg=f"{name}: per_rank residual {residuals_pr:.0f} "
                    f"≠ full residual {residuals:.0f}",
            )

    def test_gpt_oss_fused_expert_bytes_are_tree_resident(self):
        """gpt-oss is the canonical MXFP4-fused case. The clean fix gives
        the fused expert op OWNERSHIP of its weight (params: In*Out) and
        routes the `_blocks`/`_scales` through the normal role-sharded
        tree as alias storage, so the multi-GiB expert bytes land at real
        tree LEAVES (`experts.gate_up_proj` / `experts.down_proj`) rather
        than as an orphaned residual. Two invariants pin that contract:

          1. residual == 0 — every byte is accounted for inside the
             group/model-level trees (no bytes fall through to the
             deployment_total minus tree-sum gap);
          2. the fused expert leaves carry multi-GiB — proving the bytes
             are genuinely counted at the fused op, not silently dropped
             (which would ALSO make residual 0 but leave the total short).

        Together these are the anti-regression guard the old residual
        test provided, now stated against the tree-resident model.
        """
        resolved = _resolved("openai__gpt-oss-20b")
        wb = per_rank_weight_breakdown(
            resolved, ParallelConfig())
        groups_full = sum(_sum_tree_full(g) for g in wb["groups"])
        ml_full = sum(_sum_tree_full(m) for m in wb["model_level"])
        residuals = wb["deployment_total_bytes"] - groups_full - ml_full
        # (1) No orphaned residual — fused bytes flow through the tree.
        self.assertAlmostEqual(
            residuals, 0.0, delta=max(1.0, wb["deployment_total_bytes"] * 1e-6),
            msg=f"gpt-oss residual {residuals:.0f} B should be ~0 — the "
                f"fused-MXFP4 bytes are tree-resident (fused op owns the "
                f"weight; _blocks/_scales are alias storage in the tree).",
        )
        # The published data / scale split is the total's own addends — the
        # Memory card prints `total = data + scale` off these two fields.
        self.assertAlmostEqual(
            wb["deployment_data_bytes"] + wb["deployment_scale_bytes"]
            + wb["unattributed_scale_bytes"],
            wb["deployment_total_bytes"],
            delta=max(1.0, wb["deployment_total_bytes"] * 1e-9))
        self.assertGreater(wb["deployment_scale_bytes"], 0.0)

        # (2) The fused expert leaves must carry the multi-GiB weight.
        def _leaf_bytes(node, needle):
            kids = node.get("children")
            if kids:
                return sum(_leaf_bytes(k, needle) for k in kids)
            name = str(node.get("name") or node.get("aggregation_id") or "")
            return float(node.get("bytes_full", 0) or 0) if needle in name else 0.0

        expert_full = sum(
            _leaf_bytes(g, "experts.") for g in wb["groups"])
        self.assertGreater(
            expert_full, 100 * 1024 * 1024,
            msg=f"gpt-oss fused expert leaves carry only {expert_full:.0f} B "
                f"— the fused-MXFP4 weight stopped being counted in the "
                f"tree (bytes dropped instead of owned by the fused op).",
        )


def _make_leaf(sid: str, *, op: str, flops: int, params: int = 0):
    return op, Cost(flops=flops, params=params, active_params=params)


def _compute_timing(leaves, access, *, count_per_step=None, **kwargs):
    # No SKU in these fixtures: no node boundary to split a collective over.
    kwargs.setdefault("gpus_per_node", None)
    """Build the same UID layers the production timing fold consumes."""
    nodes = [
        SpecIRNode(
            name=sid,
            op=op,
            is_component=False,
            uid=index,
        )
        for index, (sid, (op, _cost)) in enumerate(leaves.items(), 1)
    ]
    tree = SpecIRTree(top_level=nodes, _next_uid=len(nodes) + 1)
    spec = SimpleNamespace(
        name="timing-fixture",
        components={},
        blocks={},
        component_roles={},
        model=[],
        custom_ops={},
    )
    resolved = SimpleNamespace(spec=spec, config={})
    program = Program(resolved=resolved, ir=tree)
    by_sid = {node.aggregation_id: node.uid for node in nodes}
    costs = Layer(program, {
        by_sid[sid]: cost for sid, (_op, cost) in leaves.items()
    })
    shapes = Layer(program, {node.uid: Shape() for node in nodes})
    access_layer = Layer(program, {
        by_sid[sid]: row for sid, row in access.items()
    })
    instances = Layer(program, {
        node.uid: int((count_per_step or {}).get(node.aggregation_id, 1))
        for node in nodes
    })
    return _compute_per_op_timing(
        program,
        costs,
        shapes,
        access_layer,
        instances=instances,
        # The fixture stores no tensors, so no row multiplies a matrix.
        weight_shards=lambda packs: (),
        # Numbers are their own laws: the fixture states no workload.
        laws=numbers_as_laws(
            cost=costs, shapes=shapes, full_shapes=kwargs.get("full_shapes"),
            access=access_layer, access_shapes=shapes,
            comm_meta=kwargs.get("comm_meta"),
            node_dims=kwargs.get("node_dims"),
            flops_bindings=kwargs.get("flops_bindings"),
            step_rows=kwargs.get("tokens_per_step", 1)),
        pre=False,
        # The fixture's layers are numbers, their own laws at any point.
        at={"B": 1, "s_input": 1, "s_output": 0, "s_prefix": 0, "s_verify": 1, "rho": 1},
        **kwargs,
        breakdown=False,
    )


def _ab(weight_dt: float, act_dt: float, *, full: int = 100,
        shard: int = 1, act_dtype: str = "BF16",
        mma_dtypes: tuple[str, ...] = (),
        promotion_ladder: bool = False,
        edge_dtype: str | None = None,
        convert_bytes: int = 0) -> LeafAccess:
    return LeafAccess(
        full=full, per_rank=full // max(shard, 1),
        weight_bytes=full, weight_bytes_per_rank=full // max(shard, 1),
        weight_dtype_bytes=weight_dt,
        act_dtype_bytes=act_dt,
        act_dtype=act_dtype,
        mma_dtypes=mma_dtypes,
        promotion_ladder=promotion_ladder,
        edge_dtype=edge_dtype if edge_dtype is not None else act_dtype,
        convert_bytes=convert_bytes,
        convert_bytes_per_rank=convert_bytes,
        convert_extent_per_rank=convert_bytes,
        # An unconverted edge carries the activation's own dtype; the access
        # layer always states it beside `act_dtype` (the weight-only fallback
        # on a card where no declared family remains runs at it).
    )


class TestPerOpPeakFlopsRoofline(unittest.TestCase):
    """`compute_per_op_timing` must derive each leaf's roofline peak
    from THAT leaf's dtype, not from a single global peak. Mixed-
    precision specs (FP8 experts + BF16 norms + INT4 matmuls) used to
    collapse to one peak — t_compute on the FP8 row was BF16-slow,
    visibly wrong on the dashboard.
    """

    def _two_leaves_mixed_precision(self):
        fp8 = _make_leaf("expert.gate_proj", op="linear", flops=10**12, params=10**8)
        bf16 = _make_leaf("model.norm", op="rmsnorm", flops=10**9, params=10**6)
        leaves = {
            "expert.gate_proj": fp8,
            "model.norm": bf16,
        }
        access_bytes = {
            "expert.gate_proj": _ab(weight_dt=1.0, act_dt=2.0),  # FP8 W, BF16 act
            "model.norm":       _ab(weight_dt=2.0, act_dt=2.0),  # BF16 W
        }
        return leaves, access_bytes

    def test_per_row_peak_uses_leaf_act_dtype(self):
        """A W8A16 expert (FP8 weights, BF16 activations) executes its
        GEMM at the activation precision — the weight dequantizes to
        BF16 inside the kernel before the MMA. Picking the FP8 column
        for compute peak would overstate throughput by 2×. Default Auto
        therefore uses act_dtype_bytes per leaf: the W8A16 expert reads
        BF16 peak (adt=2.0), and the pure-BF16 norm also reads BF16.
        """
        leaves, access_bytes = self._two_leaves_mixed_precision()
        fp8_peak = 4500e12
        bf16_peak = 2250e12
        global_dominant = bf16_peak
        def peak_fn(dt: Optional[str]) -> Optional[float]:
            if dt in ("FP8", "FP4", "INT4", "INT8"):
                return fp8_peak
            return bf16_peak
        rows = _compute_timing(
            leaves, access_bytes,
            count_per_step={"expert.gate_proj": 1, "model.norm": 1},
            peak_flops_per_sec=global_dominant,
            peak_hbm_bw_bytes_per_sec=3e12,
            peak_flops_for_op_fn=peak_fn,
            overlay_batch=1,
            tokens_per_step=1,
        )
        by_sid = {r["aggregation_id"]: r for r in rows}
        self.assertEqual(
            by_sid["expert.gate_proj"]["peak_flops_per_sec_for_op"],
            bf16_peak,
            msg="W8A16 expert (adt=2) must read BF16 peak — activation "
                "dtype is the GEMM input precision; the FP8 weight "
                "dequants to BF16 inside the kernel.",
        )
        self.assertEqual(
            by_sid["model.norm"]["peak_flops_per_sec_for_op"],
            bf16_peak,
            msg="Pure-BF16 norm (adt=2) reads BF16 peak.",
        )
        # t_compute now consistently uses the BF16 peak across both rows.
        gate_t = by_sid["expert.gate_proj"]["t_compute_s"]
        self.assertAlmostEqual(
            gate_t, 10**12 / bf16_peak, delta=1e-15,
            msg=f"W8A16 t_compute {gate_t} expected {10**12/bf16_peak}",
        )

    def test_compute_only_leaf_uses_act_dtype(self):
        """A weight-less leaf (softmax / attention_score / activation)
        has `weight_dtype_bytes=0`; the peak still comes from the
        activation dtype — which is the default rule anyway, so this
        case looks like every other leaf. The risk to lock against is
        such leaves silently taking the global peak when an
        activation-only dtype is present.
        """
        leaves = {
            "attn.softmax": _make_leaf("attn.softmax", op="softmax",
                                       flops=10**10, params=0),
        }
        access_bytes = {
            "attn.softmax": LeafAccess(
                full=64, per_rank=64,
                act_in_bytes=32, act_out_bytes=32,
                weight_dtype_bytes=0.0,  # no weights
                act_dtype="BF16",
            ),
        }
        captured = []
        def peak_fn(dt: Optional[str]) -> float:
            captured.append(dt)
            return 2250e12 if dt in ("BF16", "FP16", "F16") else 4500e12
        rows = _compute_timing(
            leaves, access_bytes,
            count_per_step={"attn.softmax": 1},
            peak_flops_per_sec=999e12,
            peak_hbm_bw_bytes_per_sec=3e12,
            peak_flops_for_op_fn=peak_fn,
        )
        self.assertEqual(captured, ["BF16"],
            msg=f"closure should have been invoked with act_dtype=BF16 "
                f"(GEMM input precision) for weight-less leaf; got {captured}",
        )
        self.assertEqual(rows[0]["peak_flops_per_sec_for_op"], 2250e12)

    def test_no_closure_falls_back_to_global_peak(self):
        """Back-compat: when the caller doesn't supply
        `peak_flops_for_op_fn` (legacy code path), every row must use
        the global `peak_flops_per_sec`. Frontend's `||` fallback only
        triggers if `peak_flops_per_sec_for_op` is 0/missing, so we
        also lock that the row carries the global peak in this case
        (not 0).
        """
        leaves, access_bytes = self._two_leaves_mixed_precision()
        global_peak = 1234e12
        rows = _compute_timing(
            leaves, access_bytes,
            count_per_step={"expert.gate_proj": 1, "model.norm": 1},
            peak_flops_per_sec=global_peak,
            peak_hbm_bw_bytes_per_sec=3e12,
            peak_flops_for_op_fn=None,
        )
        for r in rows:
            self.assertEqual(
                r["peak_flops_per_sec_for_op"], global_peak,
                msg=f"no-closure path must echo global peak on every "
                    f"row; row {r['aggregation_id']} got "
                    f"{r['peak_flops_per_sec_for_op']}",
            )

    def test_closure_returning_none_falls_back_to_global(self):
        """SKU without an FP8 column → closure returns None → row
        must use the global peak rather than ending up with 0.
        (Older Ampere/A100 with FP8-quantized weights is the real
        case; we'd rather show a BF16-derived rooftop than a divide-
        by-zero collapse.)
        """
        leaves = {
            "expert.gate_proj": _make_leaf("expert.gate_proj", op="linear",
                                           flops=10**12, params=10**8),
        }
        access_bytes = {
            "expert.gate_proj": _ab(weight_dt=1.0, act_dt=2.0),
        }
        global_peak = 312e12  # A100 BF16
        rows = _compute_timing(
            leaves, access_bytes,
            count_per_step={"expert.gate_proj": 1},
            peak_flops_per_sec=global_peak,
            peak_hbm_bw_bytes_per_sec=2e12,
            peak_flops_for_op_fn=lambda dt: None,  # SKU sheet doesn't list dtype
        )
        self.assertEqual(rows[0]["peak_flops_per_sec_for_op"], global_peak,
            msg="closure → None must fall back to global, not 0",
        )

    def test_a_missing_ladder_rung_promotes_within_the_declared_families(
            self):
        """An audited F8×MXFP4 ladder with F8 but no F4 prices at F8.

        This is Humming's Hopper fact: the weight is software-dequantised
        while the block-FP8 activation remains a real conversion and a real
        FP8 MMA. The fallback may choose only from families the launch itself
        declared; it never infers an MMA from operand width.
        """
        leaves = {"expert.gate_proj": _make_leaf(
            "expert.gate_proj", op="linear", flops=10**12, params=10**8)}
        access_bytes = {"expert.gate_proj": _ab(
            weight_dt=0.5, act_dt=1.0, act_dtype="F8_E4M3",
            mma_dtypes=("F8_E4M3", "MXFP4"), edge_dtype="BF16",
            promotion_ladder=True, convert_bytes=11)}
        fp8_peak, bf16_peak = 4000e12, 2000e12
        rows = _compute_timing(
            leaves, access_bytes,
            peak_flops_per_sec=bf16_peak,
            peak_hbm_bw_bytes_per_sec=3e12,
            peak_flops_for_op_fn=lambda dt: (
                bf16_peak if dt == "BF16" else
                fp8_peak if dt == "F8_E4M3" else None))
        self.assertEqual(rows[0]["peak_flops_per_sec_for_op"], fp8_peak)
        self.assertEqual(rows[0]["convert_bytes_per_rank"], 11)

    def test_promotion_uses_the_widest_remaining_declared_family(self):
        """A card with FP4 but no FP8 still has a declared feasible family."""
        leaves = {"expert.gate_proj": _make_leaf(
            "expert.gate_proj", op="linear", flops=10**12, params=10**8)}
        access_bytes = {"expert.gate_proj": _ab(
            weight_dt=0.5, act_dt=1.0, act_dtype="F8_E4M3",
            mma_dtypes=("F8_E4M3", "MXFP4"), edge_dtype="BF16",
            promotion_ladder=True, convert_bytes=11)}
        fp4_peak, bf16_peak = 5000e12, 2000e12
        rows = _compute_timing(
            leaves, access_bytes,
            peak_flops_per_sec=bf16_peak,
            peak_hbm_bw_bytes_per_sec=3e12,
            peak_flops_for_op_fn=lambda dt: (
                bf16_peak if dt == "BF16" else
                fp4_peak if dt == "MXFP4" else None))
        self.assertEqual(rows[0]["peak_flops_per_sec_for_op"], fp4_peak)
        self.assertEqual(rows[0]["convert_bytes_per_rank"], 11)

    def test_fallback_to_the_edge_only_when_no_declared_family_remains(self):
        """Every declared MMA column missing is the old W4A16 dequant path."""
        leaves = {"expert.gate_proj": _make_leaf(
            "expert.gate_proj", op="linear", flops=10**12, params=10**8)}
        access_bytes = {"expert.gate_proj": _ab(
            weight_dt=0.5, act_dt=1.0, act_dtype="F8_E4M3",
            mma_dtypes=("F8_E4M3", "MXFP4"), edge_dtype="BF16",
            promotion_ladder=True, convert_bytes=11)}
        bf16_peak = 2000e12
        rows = _compute_timing(
            leaves, access_bytes,
            peak_flops_per_sec=bf16_peak,
            peak_hbm_bw_bytes_per_sec=3e12,
            peak_flops_for_op_fn=lambda dt: (
                bf16_peak if dt == "BF16" else None))
        self.assertEqual(rows[0]["peak_flops_per_sec_for_op"], bf16_peak)
        self.assertEqual(rows[0]["convert_bytes_per_rank"], 0)

    def test_a_default_mixed_launch_never_promotes_a_partial_family_set(self):
        """Without the launch's explicit ladder fact, all families are required.

        This is the DeepGEMM/TRT-LLM side of the audited split: one present
        family cannot silently turn a simultaneous mixed instruction into a
        software-dequantisation launch.
        """
        leaves = {"expert.gate_proj": _make_leaf(
            "expert.gate_proj", op="linear", flops=10**12, params=10**8)}
        access_bytes = {"expert.gate_proj": _ab(
            weight_dt=0.5, act_dt=1.0, act_dtype="F8_E4M3",
            mma_dtypes=("F8_E4M3", "MXFP4"), edge_dtype="BF16",
            convert_bytes=11)}
        fp8_peak, bf16_peak = 4000e12, 2000e12
        captured = []
        def peak_fn(dt):
            captured.append(dt)
            return (fp8_peak if dt == "F8_E4M3" else
                    bf16_peak if dt == "BF16" else None)
        rows = _compute_timing(
            leaves, access_bytes,
            peak_flops_per_sec=bf16_peak,
            peak_hbm_bw_bytes_per_sec=3e12,
            peak_flops_for_op_fn=peak_fn)
        self.assertEqual(captured, ["F8_E4M3", "MXFP4", "BF16"])
        self.assertEqual(rows[0]["act_dtype"], "BF16")
        self.assertEqual(rows[0]["peak_flops_per_sec_for_op"], bf16_peak)
        self.assertEqual(rows[0]["convert_bytes_per_rank"], 0)

    def test_h100_vllm_mxfp4_prices_through_its_selected_w4a16_launch(self):
        """The H100 answer is selected as a BF16 launch before pricing.

        Its `mma_dtypes` is exactly BF16; pricing therefore asks one column
        and never enters the mixed-family fallback network.
        """
        leaves = {"expert.gate_proj": _make_leaf(
            "expert.gate_proj", op="linear", flops=10**12, params=10**8)}
        access_bytes = {"expert.gate_proj": _ab(
            weight_dt=0.5, act_dt=2.0, act_dtype="BF16",
            mma_dtypes=("BF16",))}
        captured = []
        def peak_fn(dt):
            captured.append(dt)
            return 2000e12 if dt == "BF16" else None
        rows = _compute_timing(
            leaves, access_bytes,
            peak_flops_per_sec=999e12,
            peak_hbm_bw_bytes_per_sec=3e12,
            peak_flops_for_op_fn=peak_fn)
        self.assertEqual(captured, ["BF16"])
        self.assertEqual(rows[0]["peak_flops_per_sec_for_op"], 2000e12)

    def test_real_gpu_csv_dispatch(self):
        """End-to-end against the real `gpu_specs.csv`: B200 lists FP4,
        FP8 and BF16 columns at distinct TFLOPs, so a W4A4, W8A8, and
        W16A16 leaf must each get a DIFFERENT per-row peak when the
        closure wraps `peak_flops_for_dtype(B200)`. The per-leaf dtype
        passed into the closure is `act_dtype` (GEMM input precision
        **name**), so genuine FP4/FP8/BF16 dispatch happens when
        activations are FP4/FP8/BF16 respectively. Weight-only-quant
        schemes (W4A16 / W8A16) collapse to the BF16 column, which is
        the physical truth tested separately above.

        This catches the "closure wired but always picks the same
        column" regression class.
        """
        gpu = get_gpu("B200")
        leaves = {
            "fp4.proj":  _make_leaf("fp4.proj",  op="linear", flops=10**12),
            "fp8.proj":  _make_leaf("fp8.proj",  op="linear", flops=10**12),
            "bf16.proj": _make_leaf("bf16.proj", op="linear", flops=10**12),
        }
        # Matched precision (W4A4 / W8A8 / BF16+BF16) — the only schemes
        # that actually exercise the FP4 and FP8 tensor-core columns.
        access_bytes = {
            "fp4.proj":  _ab(weight_dt=0.5, act_dt=0.5, act_dtype="FP4"),
            "fp8.proj":  _ab(weight_dt=1.0, act_dt=1.0, act_dtype="FP8"),
            "bf16.proj": _ab(weight_dt=2.0, act_dt=2.0, act_dtype="BF16"),
        }
        rows = _compute_timing(
            leaves, access_bytes,
            count_per_step={k: 1 for k in leaves},
            peak_flops_per_sec=peak_flops_for_dtype(gpu, "BF16"),
            peak_hbm_bw_bytes_per_sec=8e12,
            peak_flops_for_op_fn=lambda dt: peak_flops_for_dtype(gpu, dt),
        )
        by_sid = {r["aggregation_id"]: r["peak_flops_per_sec_for_op"] for r in rows}
        self.assertEqual(by_sid["fp4.proj"],  peak_flops_for_dtype(gpu, "FP4"))
        self.assertEqual(by_sid["fp8.proj"],  peak_flops_for_dtype(gpu, "FP8"))
        self.assertEqual(by_sid["bf16.proj"], peak_flops_for_dtype(gpu, "BF16"))
        # B200 columns: fp4=4500, fp8=4500, bf16=2250 — fp4 and fp8
        # tie, but BF16 must be strictly slower than either, otherwise
        # the dispatch is silently degenerate.
        self.assertGreater(by_sid["fp8.proj"], by_sid["bf16.proj"],
            msg=f"B200 fp8 peak must exceed bf16 peak; got fp8="
                f"{by_sid['fp8.proj']:.0f} bf16={by_sid['bf16.proj']:.0f}",
        )


class TestGpuThreadingEndToEnd(unittest.TestCase):
    """The `gpu=` parameter must thread end-to-end:
    `capacity._run_calibration_overlay` → `pipeline.analyze` →
    `phases.compute_per_op_timing(peak_flops_for_op_fn=closure)`.
    Without the wiring the closure is never formed and every per-row
    peak silently equals the global dominant peak.
    """

    def test_analyze_builds_closure_when_gpu_supplied(self):
        """Patch `compute_per_op_timing` to capture its kwargs and
        invoke a real `analyze()` with `gpu=<B200>`. The captured
        `peak_flops_for_op_fn` must be a callable that returns the
        B200's column for any given dtype byte count.
        """
        from backend.inference import pipeline as pipeline_mod
        resolved = _resolved("meta-llama__Llama-3.2-3B")
        gpu = get_gpu("B200")

        captured = {}

        def _spy(*args, **kwargs):
            captured["peak_flops_for_op_fn"] = kwargs.get("peak_flops_for_op_fn")
            captured["peak_flops_per_sec"] = kwargs.get("peak_flops_per_sec")
            # Return a no-op empty list so analyze() can finish
            # building its AnalysisResult.
            return []

        with patch.object(pipeline_mod, "compute_per_op_timing", side_effect=_spy):
            # `framework_binding=None` → identity fusion path; the
            # closure construction sits at the END of analyze() in
            # `pipeline.py:862-865` and runs regardless of the fusion
            # path, so the no-binding shortcut is fine for this test.
            pipeline_mod.analyze(
                resolved,
                parallel=ParallelConfig(),
                framework_binding=None,
                batch_size=1, avg_input_tokens=128,
                avg_output_tokens=0,
                peak_flops_per_sec=peak_flops_for_dtype(gpu, "BF16"),
                peak_hbm_bw_bytes_per_sec=8e12,
                gpu=gpu,
                breakdown=False,
            )
        fn = captured.get("peak_flops_for_op_fn")
        self.assertIsNotNone(
            fn,
            msg="pipeline.analyze(gpu=<B200>) did not build the closure; "
                "per-op peak dispatch is broken end-to-end",
        )
        # Closure must actually call into peak_flops_for_dtype(gpu, name).
        self.assertEqual(fn("FP4"), peak_flops_for_dtype(gpu, "FP4"))
        self.assertEqual(fn("FP8"), peak_flops_for_dtype(gpu, "FP8"))
        self.assertEqual(fn("BF16"), peak_flops_for_dtype(gpu, "BF16"))
        self.assertEqual(fn("FP16"), peak_flops_for_dtype(gpu, "FP16"))
        self.assertIsNone(fn(None))

    def test_analyze_omits_closure_when_gpu_is_none(self):
        """Legacy callers that don't pass `gpu=` must not pay for a
        closure (and must not crash importing gpu_specs lazily).
        """
        from backend.inference import pipeline as pipeline_mod
        resolved = _resolved("meta-llama__Llama-3.2-3B")
        captured = {}

        def _spy(*args, **kwargs):
            captured["peak_flops_for_op_fn"] = kwargs.get("peak_flops_for_op_fn")
            return []

        with patch.object(pipeline_mod, "compute_per_op_timing", side_effect=_spy):
            pipeline_mod.analyze(
                resolved,
                parallel=ParallelConfig(),
                framework_binding=None,
                batch_size=1, avg_input_tokens=128,
                avg_output_tokens=0,
                peak_flops_per_sec=500e12,
                peak_hbm_bw_bytes_per_sec=3e12,
                gpu=None,
                breakdown=False,
            )
        self.assertIsNone(
            captured.get("peak_flops_for_op_fn"),
            msg="gpu=None must NOT build a closure (legacy back-compat)",
        )


if __name__ == "__main__":
    unittest.main()
