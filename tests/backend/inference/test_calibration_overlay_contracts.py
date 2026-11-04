"""Contract tests for the post-refactor calibration architecture.

Calibration is a first-class stage with typed inputs (`WorkloadShape` /
`DeploymentContext`): the overlay states each row's identity and theory
classification and its `PointRow`; what is measured is the measured fold's
(`batch_sweep.step_measure` / `selector`, `auto_sweep.PointMeasure.fold`),
the same for an estimate as for a sweep point. These tests guard the
contracts:

  * CalibrationOverlay produces one row per leaf in the AnalysisResult.
  * Every row's `source` is in the documented 4-value enum.
  * `SlimBenchmarkKey` collapses dtype aliases (UI `fp8` matches CSV
    `fp8_e4m3` without separate lookup paths).
  * `prefer_measured=False` route response has no `per_op_timing` key
    (legacy byte-equivalent shape).
  * Materialize is consulted for the dominant weight-scale dtype
    (CLAUDE.md §2.3 invariant — materialize is the single quant fusion
    entry).

Older tests asserted specific magic numbers (`387.5e-6` for
FlashAttention at cnt=28, `weight_scale_bpe=1.0` threading, etc.) —
those are case-locking, not contract testing, and made every refactor
require re-deriving handful numbers. The contracts here are
shape/structure assertions that survive any sensible refactor.
"""
from __future__ import annotations

from types import SimpleNamespace

from typing import Any

import pytest

from backend.inference.row_roofline import LawBook, build_row_exprs, law_columns

from tests.snapshots import resolve_snapshot, snapshot_config

import tests.corpus_row as _corpus

from backend.arch.eval.analysis import Analysis
from backend.arch.eval.layer import Layer, Rewrite
from backend.arch.eval.program import Program
from backend.arch.eval.access import LeafAccess
from backend.arch.eval.values import Cost, Shape
from backend.arch.spec_tree import SpecIRNode, SpecIRTree
from backend.inference.benchmark_identity import (
    compute_per_row_dtype_env_json,
    hardware_hash,
    op_kernel_hash,
    workload_hash,
)
from backend.inference.calibration_overlay import (
    CalibrationOverlay,
    DeploymentContext,
    WorkloadShape,
    build_deployment_context,
)
from backend.inference.op_calibration import (
    BenchmarkKey,
    CalibrationRegistry,
    merge_validated_rows,
)
from backend.inference.batch_sweep import _measured_row, selector, step_measure
from backend.inference.parallel_config import ParallelConfig
from backend.inference.row_roofline import StepSum
from backend.measure.auto_sweep import PointMeasure
from backend.inference.pipeline import AnalysisResult


# ---------------------------------------------------------------------------
# WorkloadShape / DeploymentContext — typed inputs
# ---------------------------------------------------------------------------



class TestDeploymentContextParallelSignature:
    """parallel_signature is canonical sorted form — matches CSV
    `parallel` column convention so lookup is shape-stable across
    callers that build ParallelConfig differently."""

    def test_single_tp_no_other_knobs(self):
        deployment = build_deployment_context(
            framework="vllm",
            framework_binding=None,
            parallel=ParallelConfig(),
            gpu_sku="H100-SXM",
        )
        assert deployment.parallel_signature == "tp=1"

    def test_multi_knob_sorted_alphabetically(self):
        deployment = build_deployment_context(
            framework="vllm",
            framework_binding=None,
            parallel=ParallelConfig(tp_attention=4, ep_moe=2),
            gpu_sku="H100-SXM",
        )
        # sorted alphabetically: ep_moe before tp_attention
        assert deployment.parallel_signature == "ep_moe=2,tp_attention=4"


# ---------------------------------------------------------------------------
# SlimBenchmarkKey — canonical 4-hash identity
# ---------------------------------------------------------------------------


class TestSlimKeyCanonicalization:
    """SlimBenchmarkKey collapses dtype aliases / case / JSON whitespace
    into the same 4 hashes — so a contributor's `fp8` row matches a
    lookup that constructs `fp8_e4m3` (and vice versa)."""

    def _wide(self, **overrides) -> BenchmarkKey:
        defaults = dict(
            model="meta-llama/Llama-3.2-3B",
            spec_hash="abc123",
            phase="prefill",
            parallel="tp=1",
            framework="vllm",
            framework_version="0.21.0",
            gpu="H100-SXM",
            op="flash_attention",
            node_spec_id="self_attn.attn_apply",
            shape_json='{"B":1,"s_ctx":2048}',
            dtype_env_json=(
                '{"activation":"bf16","kv":"bf16","metadata":"fp32",'
                '"output":"bf16","weight":"bf16"}'
            ),
            kernel_impl="flash_attention_v3",
        )
        defaults.update(overrides)
        return BenchmarkKey(**defaults)

    def test_same_wide_key_produces_same_slim_key(self):
        assert self._wide().to_slim() == self._wide().to_slim()

    def test_slim_key_has_four_string_fields(self):
        slim = self._wide().to_slim()
        assert isinstance(slim.spec_hash, str)
        assert isinstance(slim.workload_hash, str)
        assert isinstance(slim.hardware_hash, str)
        assert isinstance(slim.op_kernel_hash, str)
        # No 12 fields — 4 was the goal.
        assert len(slim.__dataclass_fields__) == 4

    def test_different_spec_hash_changes_slim(self):
        a = self._wide(spec_hash="abc").to_slim()
        b = self._wide(spec_hash="xyz").to_slim()
        assert a != b

    def test_different_workload_changes_slim(self):
        a = self._wide(shape_json='{"B":1,"s_ctx":2048}').to_slim()
        b = self._wide(shape_json='{"B":1,"s_ctx":4096}').to_slim()
        assert a.workload_hash != b.workload_hash

    def test_different_hardware_changes_slim(self):
        a = self._wide(gpu="H100-SXM").to_slim()
        b = self._wide(gpu="A100-SXM").to_slim()
        assert a.hardware_hash != b.hardware_hash

    def test_different_kernel_changes_slim(self):
        a = self._wide(kernel_impl="flash_attention_v3").to_slim()
        b = self._wide(kernel_impl="flash_attention_v2").to_slim()
        assert a.op_kernel_hash != b.op_kernel_hash


class TestSlimKeyHelpersIndependence:
    """The three hash helpers (workload_hash / hardware_hash /
    op_kernel_hash) must be deterministic and isolated — perturbing
    one input changes only that hash."""

    def test_workload_hash_deterministic(self):
        shape = '{"B":1,"d_out":768,"d_reduce":4096}'
        assert workload_hash(
            shape_json=shape, weight_dtype="bf16", activation_dtype="bf16",
            kv_dtype="bf16", output_dtype="bf16", metadata_dtype="fp32",
            weight_granularity=None, activation_granularity=None,
            output_format=None, kv_format=None,
        ) == workload_hash(
            shape_json=shape, weight_dtype="bf16", activation_dtype="bf16",
            kv_dtype="bf16", output_dtype="bf16", metadata_dtype="fp32",
            weight_granularity=None, activation_granularity=None,
            output_format=None, kv_format=None,
        )

    def test_a_weight_without_scales_hashes_its_absent_slot(self):
        """No scales is no `metadata` slot, hashed the same every time and
        apart from any scale dtype a row could state."""
        shape = '{"B":1,"d_out":768,"d_reduce":4096}'
        bare = {"shape_json": shape, "weight_dtype": "bf16",
                "activation_dtype": "bf16", "kv_dtype": "bf16",
                "output_dtype": "bf16", "weight_granularity": None,
                "activation_granularity": None, "output_format": None,
                "kv_format": None}
        assert (workload_hash(**bare, metadata_dtype=None)
                == workload_hash(**bare, metadata_dtype=None))
        assert (workload_hash(**bare, metadata_dtype=None)
                != workload_hash(**bare, metadata_dtype="fp32"))

    def test_the_weight_granularity_is_part_of_the_hash(self):
        """Two block sizes of one FP8 GEMM are two kernels: the slim key must
        not fold them together."""
        shape = '{"B":1,"d_out":768,"d_reduce":4096}'
        fp8 = {"shape_json": shape, "weight_dtype": "fp8_e4m3",
               "activation_dtype": "fp8_e4m3", "kv_dtype": "bf16",
               "output_dtype": "bf16", "metadata_dtype": "fp32",
               "activation_granularity": None, "output_format": None,
                "kv_format": None}
        assert (workload_hash(**fp8, weight_granularity="per-tile:128x128")
                != workload_hash(**fp8, weight_granularity="per-channel"))

    def test_the_activation_granularity_is_part_of_the_hash(self):
        """A per-token cast and a per-128-group one are two kernels between
        the same two dtypes: the slim key must not fold them together."""
        shape = '{"B":1,"d_model":7168,"s_query":4096}'
        cast = {"shape_json": shape, "weight_dtype": None,
                "activation_dtype": "bf16", "kv_dtype": None,
                "output_dtype": "fp8_e4m3", "metadata_dtype": "fp32",
                "weight_granularity": None, "output_format": None,
                "kv_format": None}
        assert (workload_hash(**cast, activation_granularity="per-token")
                != workload_hash(**cast, activation_granularity="per-group:128"))

    def test_the_output_format_is_part_of_the_hash(self):
        """Two cached records over one element dtype — DeepSeek-V4's
        FlashMLA record and an FP8 index key with one F32 scale — are two
        launches: the slim key must not fold them together, and a format
        whose elements are not the `output` token is refused."""
        store = {"shape_json": '{"B":1,"d_head":512,"s_query":64}',
                 "weight_dtype": "bf16", "activation_dtype": "bf16",
                 "kv_dtype": None, "output_dtype": "fp8_e4m3",
                 "metadata_dtype": None, "weight_granularity": None,
                 "activation_granularity": None, "kv_format": None}
        assert (workload_hash(
                    **store, output_format="fp8_ue8m0_group64_bf16_rope_tail")
                != workload_hash(**store, output_format="fp8_group128"))
        with pytest.raises(ValueError, match="stores fp4_e2m1 elements"):
            workload_hash(**store, output_format="mxfp4")
        with pytest.raises(ValueError, match="not a registry cache format"):
            workload_hash(**store, output_format="fp8_flashmla")

    def test_the_kv_format_is_part_of_the_hash(self):
        """The cache a launch reads, or writes beside its output, under one
        FP8 dtype: DeepSeek-V4's FlashMLA record and a per-tensor-scaled
        cache are two launches, on the writer's side (the q/KV insert) and
        the reader's (the sparse decode). Same rule as `output_format`: a
        registry name whose elements are the `kv` token."""
        row = {"shape_json": '{"B":1,"d_head":512,"s_query":64}',
               "weight_dtype": None, "activation_dtype": "bf16",
               "kv_dtype": "fp8_e4m3", "output_dtype": "bf16",
               "metadata_dtype": None, "weight_granularity": None,
               "activation_granularity": None, "output_format": None}
        record = workload_hash(**row,
                               kv_format="fp8_ue8m0_group64_bf16_rope_tail")
        assert record != workload_hash(**row, kv_format="fp8_per_tensor")
        assert record != workload_hash(**row, kv_format=None)
        with pytest.raises(ValueError, match="stores fp4_e2m1 elements, and "
                                             "the `kv` slot"):
            workload_hash(**row, kv_format="mxfp4")
        with pytest.raises(ValueError, match="not a registry cache format"):
            workload_hash(**row, kv_format="fp8_flashmla")

    def test_the_csv_validator_reads_kv_format_by_the_same_rule(self):
        import json
        from backend.inference.benchmarks import (BenchmarkValidationError,
                                                  _check_dtype_env_json)
        ok = {"activation": "bf16", "output": "bf16", "kv": "fp8_e4m3",
              "kv_format": "fp8_ue8m0_group64_bf16_rope_tail"}
        assert _check_dtype_env_json(json.dumps(ok))["kv_format"] == (
            ok["kv_format"])
        for bad, why in (({"kv_format": "mxfp4"}, "stores fp4_e2m1"),
                         ({"kv_format": "nope"}, "not a registry"),
                         ({"kv": "bf16"}, "stores fp8_e4m3")):
            with pytest.raises(BenchmarkValidationError, match=why):
                _check_dtype_env_json(json.dumps({**ok, **bad}))

    def test_hardware_hash_deterministic(self):
        a = hardware_hash(gpu="H100-SXM", framework="vllm",
                          framework_version="0.21.0", phase="prefill",
                          parallel="tp=1")
        b = hardware_hash(gpu="H100-SXM", framework="vllm",
                          framework_version="0.21.0", phase="prefill",
                          parallel="tp=1")
        assert a == b

    def test_op_kernel_hash_deterministic(self):
        a = op_kernel_hash(node_spec_id="self_attn", op="flash_attention",
                           kernel_impl="flash_attention_v3")
        b = op_kernel_hash(node_spec_id="self_attn", op="flash_attention",
                           kernel_impl="flash_attention_v3")
        assert a == b


# ---------------------------------------------------------------------------
# CalibrationOverlay — stage contracts
# ---------------------------------------------------------------------------


class _FixtureAnalysis(Analysis):
    """A fixture graph's own dtypes, stated rather than derived.

    A row's key reads only the row's own dtypes, but the overlay still asks
    the walk whether it stores a weight at all (`WorkloadShape.of_walk`).
    A real analysis reads that off the checkpoint. This fixture has none —
    it is three layers over one hand-built node — so it says what it ships
    at instead of being given a spec deep enough to derive it from.
    """

    @property
    def resolved_precision(self) -> Any:
        return SimpleNamespace(model_dtype="BF16")

    def dominant_weight_dtype(self) -> str:
        return "BF16"

    def dominant_cache_dtype(self) -> str:
        return "BF16"

    def dominant_weight_scale_dtype(self) -> None:
        return None


class _StubAnalysis(AnalysisResult):
    """Small real analysis graph for overlay contract tests."""

    def __init__(self):
        node = SpecIRNode(
            name="attn_apply",
            op="flash_attention",
            role="attention",
            is_component=False,
            uid=1,
        )
        tree = SpecIRTree(top_level=[node], _next_uid=2)
        spec = SimpleNamespace(
            name="calibration-fixture",
            components={},
            blocks={},
            component_roles={},
            model=[],
            custom_ops={},
            encoder_groups=lambda: (),
        )
        program = Program(
            # A fixture analysis has to say what it ships at — the same
            # question every real spec answers.
            resolved=SimpleNamespace(spec=spec, config={},
                                     shipped_dtype=lambda: ("BF16", "config")),
            ir=tree,
        )
        cost = Layer(program, {node.uid: Cost(flops=1_000_000)})
        shapes = Layer(program, {
            node.uid: Shape(
                ins=((1, 24, 2048, 128), (1, 8, 2048, 128)),
                out=(1, 24, 2048, 128),
            ),
        })
        access = Layer(program, {
            node.uid: LeafAccess(full=1_000_000, per_rank=1_000_000),
        })
        instances = Layer(program, {node.uid: 1})
        structural = _FixtureAnalysis(program=program, cost=cost, shapes=shapes)

        super().__init__(
            analysis=structural,
            pre_program=program,
            program=program,
            pre_tree=tree,
            comm_tree=tree,
            post_tree=tree,
            pre_cost=cost,
            cost=cost,
            replica_cost=cost,
            pre_shapes=shapes,
            shapes=shapes,
            pre_full_shapes=shapes,
            full_shapes=shapes,
            pre_access=access,
            access=access,
            pre_kv_access=Layer(program, {}),
            kv_access=Layer(program, {}),
            pre_instances=instances,
            instances=instances,
            rewrite=Rewrite(),
            per_op_timing=[
            {
                "spec_id": "self_attn.attn_apply",
                "op": "flash_attention",
                "role": "attention",
                "count_per_step": 28,
                "flops_per_rank": 1_000_000,
                "access_bytes_per_rank": 1_000_000,
                "t_compute_s": 0.001,
                "t_memory_s": 0.001,
                "t_sol_s": 0.001,
                "t_op_s": 0.001,
                "source": "theoretical_naive",
                "source_badge": {},
                "shape_json": "",
                "dtype_env_json": "",
                # The engine's own per-op shapes — the overlay derives its
                # lookup identity from these, the same derivation the write
                # side uses. A row without them has no measurable identity
                # and the overlay refuses the join instead of fabricating a
                # model-wide key.
                "shape_in": [[1, 24, 2048, 128], [1, 8, 2048, 128]],
                "shape_out": [1, 24, 2048, 128],
                # The node's own dims, which the key reads beside its shapes.
                "node_dims": {"n_heads": 24, "n_kv_heads": 8, "d_head": 128},
                "act_dtype": "BF16",
                # The row's law, as every per-op row carries it: whether it
                # costs anything is read off it (`calibration_overlay.zero_cost`).
                **law_columns(build_row_exprs(
                    count=28, flops=1_000_000 / 28, act_in=1_000_000 / 28,
                    act_out=0, demand=0, weight_total=0,
                    metadata_total=0, comm_factor=0.0, comm_priced=False),
                    LawBook()),
            },
            ],
        )


class TestOverlayContract:
    """Stage-level contracts that must survive any internal refactor."""

    def _measured(self, result, registry, *, strategy="latest", kernel_mix=None):
        """`result`'s rows with the measured fold over `registry` applied, as
        an estimate applies it (`batch_sweep.apply_measured`)."""
        rows = result.per_op_timing
        m = step_measure(StepSum.of(rows), dict(zip(map(id, rows), result.points, strict=True)),
                         dict.fromkeys(map(id, rows), "per_op_timing"),
                         gpu_sku="H100-SXM", framework="vllm", binding=None, written={})
        folded = PointMeasure.from_json(m["measure"]).fold(
            selector(registry, strategy=strategy, kernel_mix=kernel_mix))
        return [_measured_row(r, folded.rows[k]) for k, r in enumerate(rows)]

    def _deployment(self) -> DeploymentContext:
        return build_deployment_context(
            framework="vllm",
            framework_binding=None,
            parallel=ParallelConfig(),
            gpu_sku="H100-SXM",
            # The page the attention cell is keyed at (an axis of every
            # cell that reads a paged cache); no binding states one here.
            kv_block_size=16,
        )

    def _workload(self) -> WorkloadShape:
        return WorkloadShape.from_request(
            batch_size=1, seq_len=2048,
        )

    def test_overlay_returns_one_row_per_input_row(self):
        analysis = _StubAnalysis()
        overlay = CalibrationOverlay()
        result = overlay.apply(
            analysis, self._workload(), self._deployment(),
            model_id="meta-llama/Llama-3.2-3B", spec_hash="abc",
        )
        assert len(result.per_op_timing) == len(analysis.per_op_timing), (
            "CalibrationOverlay.apply must produce exactly one enriched "
            "row per input AnalysisResult row — no rows dropped, no rows "
            "duplicated"
        )

    def test_overlay_row_source_in_documented_enum(self):
        analysis = _StubAnalysis()
        overlay = CalibrationOverlay()
        result = overlay.apply(
            analysis, self._workload(), self._deployment(),
            model_id="x", spec_hash="x",
        )
        ALLOWED = {"measured", "measured_anomalous",
                   "theoretical_fused", "theoretical_naive"}
        for row in result.per_op_timing:
            assert row["source"] in ALLOWED, (
                f"row source {row['source']!r} not in documented "
                f"4-value enum {sorted(ALLOWED)!r}"
            )

    def test_overlay_refreshes_shape_and_dtype_identity(self):
        """Rooftop rows from pipeline.analyze carry empty
        shape_json / dtype_env_json placeholders. The overlay owns the
        canonical identity bytes: shape identity derives from the row's
        own engine shapes (the write side's derivation), dtype identity
        from WorkloadShape. A row whose op has no shape projection gets
        NO shape identity and no measured join — the whole-step workload
        shape used to stand in, which fabricates a key the corpus
        cannot legitimately contain."""
        analysis = _StubAnalysis()
        overlay = CalibrationOverlay()
        result = overlay.apply(
            analysis, self._workload(), self._deployment(),
            model_id="x", spec_hash="x",
        )
        for row in result.per_op_timing:
            assert row["shape_json"], (
                "overlay must derive shape_json from the row's engine shapes"
            )
            assert row["dtype_env_json"], (
                "overlay must populate dtype_env_json from WorkloadShape"
            )

    def test_row_without_shape_projection_refuses_the_join(self):
        analysis = _StubAnalysis()
        for row in analysis.per_op_timing:
            row.pop("shape_in"); row.pop("shape_out")
        overlay = CalibrationOverlay()
        result = overlay.apply(
            analysis, self._workload(), self._deployment(),
            model_id="x", spec_hash="x",
        )
        for row in result.per_op_timing:
            assert row["shape_json"] is None, (
                "no projection → no identity; the model-wide fallback is retired"
            )
            assert row.get("t_measured_s") is None

    def test_fusion_delta_is_signed_and_reads_both_sides_rows(self):
        from types import SimpleNamespace
        from backend.inference.pipeline import AnalysisResult
        slower = SimpleNamespace(
            pre_per_op_timing=[{"t_sol_s": 1.0}, {"t_sol_s": 2.0}],
            per_op_timing=[{"t_sol_s": 3.5}])
        delta = AnalysisResult.fusion_delta(slower)
        assert delta == {"pre_total_s": 3.0, "post_total_s": 3.5,
                         "delta_s": -0.5}, (
            "a fused graph that is slower reads as a negative delta, not 0")

    def test_invalid_strategy_rejected(self):
        result = CalibrationOverlay().apply(
            _StubAnalysis(), self._attn_workload(), self._deployment(),
            model_id="meta-llama/Llama-3.2-3B", spec_hash="abc")
        with pytest.raises(ValueError, match="strategy must be"):
            self._measured(result, CalibrationRegistry(), strategy="invalid")

    def _attn_row(self, *, framework, kernel_impl, latency, version="0.21.0",
                  model="meta-llama/Llama-3.2-3B"):
        return _corpus.row(
            model=model, spec_hash="abc", parallel="tp=1",
            framework=framework, framework_version=version,
            node_spec_id="self_attn.attn_apply",
            # Canonical per-op identity vocabulary — what kernel_bench's
            # plan emits. (A legacy `s_ctx` row deliberately does NOT match
            # an `s_query`/`s_kv` key: in decode the two mean different
            # numbers, and `OpSignature.project` keeps the spelling it
            # found for exactly that reason.)
            shape_json='{"B":1,"d_head":128,"kv_block_size":16,"n_heads":24,'
                       '"n_kv_heads":8,"s_kv":2048,"s_query":2048}',
            # The stub's attention node is in no cache's access set (its
            # analysis keeps none), so its key has no `kv` slot, and it
            # streams no weight.
            dtype_env_json=compute_per_row_dtype_env_json(
                weight_dtype=None, activation_dtype="bf16", kv_dtype=None,
                metadata_dtype=None),
            kernel_impl=kernel_impl,
            latency_us=str(latency), p50_us=str(latency),
            p95_us=str(latency + 10), samples="100",
            contributor="@t", date="2026-07-27")

    def test_envelope_mode_surfaces_fastest_library(self):
        # Two libraries measured the same physical attention op; envelope
        # mode must overlay the FASTER one onto the estimate, ignoring
        # framework/kernel_impl in the match.
        reg = CalibrationRegistry()
        merge_validated_rows(reg, [
            self._attn_row(framework="vllm", kernel_impl="fa_v2", latency=500.0),
            self._attn_row(framework="flashinfer", kernel_impl="fi_ragged", latency=300.0),
        ], source="<t>")
        row = self._measured_attn(reg, strategy="envelope")
        assert row["source"] == "measured"
        # measured sits BESIDE theory: t_op_s stays the theoretical rooftop,
        # the measured value lands in t_measured_s. Envelope = fastest single-
        # kernel latency (300µs, flashinfer over vllm's 500) × count_per_step (28).
        assert row["t_op_s"] == row["t_sol_s"]
        assert row["t_measured_s"] == pytest.approx(300.0e-6 * 28, rel=1e-9)
        assert row["source_badge"]["selection_mode"] == "envelope"
        assert "flashinfer" in row["source_badge"]["envelope_winner"]

    def _attn_workload(self) -> WorkloadShape:
        return WorkloadShape.from_request(
            batch_size=1, seq_len=2048,
        )

    def _measured_attn(self, reg, **select):
        result = CalibrationOverlay().apply(
            _StubAnalysis(), self._attn_workload(), self._deployment(),
            model_id="meta-llama/Llama-3.2-3B", spec_hash="abc",
        )
        rows = self._measured(result, reg, **select)
        return next(r for r in rows if r["op"] == "flash_attention")

    def test_latest_mode_takes_the_frameworks_newest_release_and_says_so(self):
        """A pinned exact implementation is read at its newest measured
        release: the badge names that release, and another library's faster
        kernel does not replace the requested identity."""
        reg = CalibrationRegistry()
        merge_validated_rows(reg, [
            self._attn_row(framework="vllm", kernel_impl="fa_v3", latency=500.0,
                           version="0.26.0"),
            self._attn_row(framework="vllm", kernel_impl="fa_v3", latency=450.0,
                           version="0.30.0", model="Qwen/Qwen3-8B"),
            self._attn_row(framework="flashinfer", kernel_impl="fi_ragged",
                           latency=300.0, version="0.6.16"),
        ], source="<t>")
        row = self._measured_attn(reg, kernel_mix={"flash_attention": "fa_v3"})
        assert row["source"] == "measured"
        assert row["t_measured_s"] == pytest.approx(450.0e-6 * 28, rel=1e-9)
        badge = row["source_badge"]
        assert (badge["framework"], badge["framework_version"],
                badge["kernel_impl"]) == ("vllm", "0.30.0", "fa_v3")
        assert badge["selection_mode"] == "latest"
        assert badge["framework_versions_measured"] == "0.30.0,0.26.0"

    def test_an_older_release_fills_a_gap_labelled_as_itself(self):
        reg = CalibrationRegistry()
        merge_validated_rows(reg, [
            self._attn_row(framework="vllm", kernel_impl="fa_v3", latency=500.0,
                           version="0.26.0"),
        ], source="<t>")
        row = self._measured_attn(reg, kernel_mix={"flash_attention": "fa_v3"})
        assert row["source"] == "measured"
        assert row["source_badge"]["framework_version"] == "0.26.0"

    def test_two_kernels_of_the_framework_name_no_single_identity(self):
        """When the framework's rows for one physical op disagree about the
        kernel, nothing picks one of them out: the op stays on theory."""
        reg = CalibrationRegistry()
        merge_validated_rows(reg, [
            self._attn_row(framework="vllm", kernel_impl="fa_v3", latency=500.0),
            self._attn_row(framework="vllm", kernel_impl="fa_v2", latency=400.0,
                           model="Qwen/Qwen3-8B"),
        ], source="<t>")
        row = self._measured_attn(reg)
        assert row["source"] != "measured"
        pinned = self._measured_attn(
            reg, kernel_mix={"flash_attention": "fa_v2"})
        assert pinned["source_badge"]["kernel_impl"] == "fa_v2"

    def test_theoretical_only_still_produces_structural_rows(self):
        """`enable_measured_lookup=False` (theoretical-only mode) must
        still produce one row per leaf with rooftop t_op_s — the merged
        Per-Module Breakdown's `Pre t_op` column depends on this. Source
        falls back to theoretical_{naive,fused}; no measured rows."""
        analysis = _StubAnalysis()
        overlay = CalibrationOverlay()
        result = overlay.apply(
            analysis, self._workload(), self._deployment(),
            model_id="x", spec_hash="x",
        )
        assert len(result.per_op_timing) == len(analysis.per_op_timing)
        for row in result.per_op_timing:
            assert row["source"] in ("theoretical_naive", "theoretical_fused"), (
                f"enable_measured_lookup=False must never emit a measured "
                f"source; got {row['source']!r}"
            )
            assert row["t_op_s"] == row["t_sol_s"], (
                "theoretical-only rows: t_op_s must equal rooftop t_sol_s"
            )
            assert row["t_measured_s"] is None, (
                "theoretical-only rows carry no measured value — the measured "
                "track stays None until a measurement is found"
            )

    def test_theoretical_only_skips_miss_reason_badge(self):
        """In theoretical-only mode there's no measurement expectation,
        so the 'no exact-match measurement for this key' reason badge
        is noise — assert it's NOT emitted (empty badge dict)."""
        analysis = _StubAnalysis()
        overlay = CalibrationOverlay()
        result = overlay.apply(
            analysis, self._workload(), self._deployment(),
            model_id="x", spec_hash="x",
        )
        for row in result.per_op_timing:
            assert "reason" not in (row.get("source_badge") or {}), (
                "theoretical-only mode must not emit 'no exact-match' "
                "miss reason — that's a lookup-was-attempted signal"
            )


# ---------------------------------------------------------------------------
# Top-table aggregates derived from breakdown
# ---------------------------------------------------------------------------


class TestPhaseAggregatesDeriveFromBreakdown:
    """Top-table prefill AND decode `total_bytes` must equal Σ of per-op
    `access_bytes_per_rank` from per_op_timing (the per-op breakdown). The
    pre-refactor closed-form (weights + cached KV only) undercount missed
    activation R/W which dominates at long S × B. Contract: after
    capacity.estimate_capacity completes,
    `phases.{prefill,decode}.total_bytes == Σ per_op_timing*`.

    Against `total_bytes`, not `mem_read_bytes`: a leaf's
    `access_bytes_per_rank` is `weight + metadata + act_in + act_out` — its
    whole traffic, both directions — and the phase splits that one sum into a
    read half and a write half. The two aggregates were interchangeable only
    while `mem_write_bytes` was the literal zero."""

    def test_the_phase_folds_the_rows_for_its_bytes_and_its_time(self):
        from backend.inference.phases import PhaseInputs, estimate_phase

        # Three-leaf breakdown:
        #   - compute-bound:    t_op = max(2ms, 1ms, 0) = 2ms
        #   - memory-bound:     t_op = max(1ms, 3ms, 0) = 3ms
        #   - comm-only:        t_op = max(0,   0,   5ms) = 5ms
        # Σ t_op = 10ms (the breakdown Total the summary mirrors), and
        # Σ access = 50MB against a closed form that would have said 1MB.
        per_op = [
            {"access_bytes_per_rank": 20_000_000,
             "t_compute_s": 0.002, "t_memory_s": 0.001, "t_comm_s": 0.0,   "t_launch_s": 0.0, "t_op_s": 0.002},
            {"access_bytes_per_rank": 20_000_000,
             "t_compute_s": 0.001, "t_memory_s": 0.003, "t_comm_s": 0.0,   "t_launch_s": 0.0, "t_op_s": 0.003},
            {"access_bytes_per_rank": 10_000_000,
             "t_compute_s": 0.0,   "t_memory_s": 0.0,   "t_comm_s": 0.005, "t_launch_s": 0.0, "t_op_s": 0.005},
        ]
        from tests.snapshots import shared_loader
        cfg = snapshot_config("meta-llama/Llama-3.2-3B")
        spec = shared_loader().match(cfg, model_id="meta-llama/Llama-3.2-3B")
        phase = estimate_phase(
            PhaseInputs(
                avg_input_tokens=128, avg_output_tokens=8,
                weight_bytes_per_rank=1_000_000,
                per_token_kv_bytes_per_rank=0, static_kv_bytes_per_rank=0,
                resolved_spec=spec,
            ),
            phase="prefill", batch=1, graph_rows=per_op,
        )
        assert phase.mem_read_bytes == 50_000_000
        assert phase.total_bytes == 50_000_000
        assert phase.arithmetic_intensity == phase.flops_per_step / 50_000_000
        # Per-leaf roofline sums — the summary reconciles with the
        # Per-Module Breakdown Total (Σ leaf max(t_compute, t_memory,
        # t_comm, t_launch)) instead of using the cluster roofline max(Σc, Σm).
        assert phase.t_compute_s == pytest.approx(0.003)
        assert phase.t_memory_s == pytest.approx(0.004)
        assert phase.t_comm_s == pytest.approx(0.005)
        assert phase.t_sol_s == pytest.approx(0.010)
        # Formula tooltips:
        # The read half names the three terms it is made of, and the write half
        # names the one it is: `access_bytes_per_rank` is the sum of all four,
        # so no single formula can be spelled with it.
        assert "act_in" in phase.formulas["mem_read"]["symbolic"]
        assert "act_out" in phase.formulas["mem_write"]["symbolic"]
        assert "Σ_leaf max" in phase.formulas["t_sol"]["symbolic"]
        # Substituted line shows the actual Σ max (= t_sol) value so the
        # symbolic and the rendered value reconcile. The per-component
        # sums (which DON'T add up to t_sol because each leaf only
        # contributes its winning bound) live in a separate note line —
        # NEVER inlined into substituted as additive terms (would read
        # like "0.003 + 0.004 + 0.005 → 0.010" and mislead).
        tmin_form = phase.formulas["t_sol"]
        sub = tmin_form["substituted"]
        assert "Σ rooflines" not in sub
        assert "→" not in sub
        assert "+" not in sub
        assert "0.01" in sub
        note = tmin_form.get("note", "")
        assert "per-bound sums, not addends" in note
        assert "Σt_compute" in note and "Σt_memory" in note and "Σt_comm" in note

    def test_estimate_capacity_emits_decode_per_op_timing(self):
        """End-to-end smoke: capacity.estimate_capacity now runs the
        overlay for BOTH phases. Decode rows live under the new
        `per_op_timing_decode` key so the frontend (today: prefill
        only) can render a Decode t_op column without changing the
        legacy `per_op_timing` (prefill) key."""
        from backend.inference.capacity import estimate_capacity
        from backend.inference.parallel_config import ParallelConfig

        cfg = snapshot_config("meta-llama/Llama-3.2-3B")
        resolve_snapshot("meta-llama/Llama-3.2-3B")
        result = estimate_capacity(
             config=cfg,
            gpu_sku="H100-SXM",
            avg_input_tokens=2048,
            avg_output_tokens=128,
            parallel=ParallelConfig(),
            model_id="meta-llama/Llama-3.2-3B",
            selection_mode="theoretical_only",
        )
        assert "per_op_timing" in result, "prefill per_op_timing missing"
        assert "per_op_timing_decode" in result, "decode per_op_timing missing"
        # Σ access_bytes from decode breakdown must equal top-table
        # phases.decode.total_bytes (contract — top-table = Σ breakdown), and
        # the read/write pair must be a partition of that same sum rather than
        # a second derivation of it.
        decode = result["phases"]["decode"]
        decode_sum = sum(
            int(r.get("access_bytes_per_rank") or 0)
            for r in result["per_op_timing_decode"]
        )
        if decode_sum > 0:
            assert decode["total_bytes"] == decode_sum, (
                f"decode top-table total_bytes ({decode['total_bytes']}) "
                f"must equal Σ breakdown access_bytes ({decode_sum})"
            )
            assert (decode["mem_read_bytes"] + decode["mem_write_bytes"]
                    == decode_sum), (
                f"decode read ({decode['mem_read_bytes']}) + write "
                f"({decode['mem_write_bytes']}) must partition the same "
                f"{decode_sum}"
            )

    def test_comm_rows_anchor_resolves_in_flops_by_module(self):
        """Cross-module contract for the frontend's comm-row inline
        graft. Three components have to stay in lockstep:

          1. parallelism_comm._build_comm_node sets __anchor_spec_id__
             (producer for AR/RS, consumer for AG/A2A).
          2. communication materialization records that anchor in the
             UID-keyed metadata layer.
          3. phases.compute_per_op_timing copies both fields onto each
             comm row in per_op_timing.

        The frontend's `inference.js` (comm-graft loop) then looks
        up the flops_by_module row whose `aggregation_id ==
        parent_aggregation_id` and inserts the comm row immediately
        after. If either field is missing, or if the anchor id is
        absent from flops_by_module, the graft silently falls back
        to `rows.push(commRow)` — and the user sees every collective
        piled at the end of the breakdown.

        This test exercises a TP=8 dense Llama (guaranteed to emit AR
        collectives) and locks the contract end-to-end."""
        from backend.inference.capacity import estimate_capacity

        cfg = snapshot_config("meta-llama/Llama-3.2-3B")
        resolve_snapshot("meta-llama/Llama-3.2-3B")
        result = estimate_capacity(
             config=cfg,
            gpu_sku="H100-SXM",
            avg_input_tokens=2048,
            avg_output_tokens=128,
            parallel=ParallelConfig(tp_attention=8, tp_dense_ffn=8),
            model_id="meta-llama/Llama-3.2-3B",
            selection_mode="theoretical_only",
        )

        comm_ops = {"all_reduce", "reduce_scatter", "all_gather", "all_to_all"}
        pot = result.get("per_op_timing") or []
        comm_rows = [r for r in pot if (r.get("op") or "") in comm_ops]
        assert comm_rows, (
            "TP=8 dense Llama must emit at least one parallelism collective "
            "in per_op_timing (prefill)"
        )

        fbm = (
            result.get("phases", {}).get("prefill", {}).get("flops_by_module")
            or []
        )
        agg_ids = {
            (r.get("aggregation_id") or "")
            for r in fbm
            if r.get("aggregation_id")
        }

        for r in comm_rows:
            parent = r.get("parent_aggregation_id") or ""
            agg = r.get("aggregation_id") or ""
            # The vocab-parallel embedding all_reduce anchors to `embed_tokens`,
            # a zero-FLOPs leaf that has no flops_by_module row — so this minor
            # collective legitimately cannot inline-graft and falls back to the
            # end of the breakdown. (The dominant vocab collective, the lm_head
            # all_gather, anchors to the lm_head FLOPs row and DOES graft.)
            if agg.startswith("embed_tokens.vocab_"):
                continue
            assert parent, (
                f"comm row {agg!r} has empty parent_aggregation_id — "
                f"frontend graft will fall back to push (piled at end)"
            )
            assert parent in agg_ids, (
                f"comm row {agg!r} anchor (parent_aggregation_id={parent!r}) "
                f"is missing from flops_by_module — frontend graft "
                f"cannot inline-attach it to its producer"
            )


class TestFusedRowsAreNativeInTheModuleTable:
    """The module table walks the grafted + fused tree, so fused kernels
    are rows of their own (badge names the fused op and the absorbed
    siblings) and the absorbed pre-fusion rows simply do not exist. The
    old `fused_away_aggregation_ids` filter — which hid pre-fusion rows on
    a table that no longer contains them — is gone from the response."""

    def test_estimate_ships_native_fused_rows_and_no_fused_away(self):
        from backend.inference.capacity import estimate_capacity
        from backend.inference.parallel_config import ParallelConfig
        cfg = snapshot_config("deepseek-ai/DeepSeek-V3")
        resolve_snapshot("deepseek-ai/DeepSeek-V3")
        result = estimate_capacity(
             config=cfg, gpu_sku="H200", num_gpus=8,
            avg_input_tokens=2048, avg_output_tokens=128,
            # A deployment that holds the weights: one that does not runs
            # no step, and publishes no table (`capacity.without_step`).
            parallel=ParallelConfig(dp_attention=8, ep_moe=8),
            model_id="deepseek-ai/DeepSeek-V3",
            selection_mode="theoretical_only",
        )
        assert "fused_away_aggregation_ids" not in result
        rows = result["phases"]["decode"]["flops_by_module"]
        fused = {
            (r.get("source_badge") or {}).get("fused_op"): r
            for r in rows
            if (r.get("source") or "") == "theoretical_fused"
        }
        # The vLLM-aligned fusions appear as native rows…
        for op in ("flash_attention", "silu_and_mul",
                   "merged_column_parallel_linear"):
            assert op in fused, f"{op} missing from the module table"
        # …and their absorbed members are not rows at all.
        names = {r.get("name") for r in rows}
        for absorbed in ("qk_score", "softmax", "silu", "up_proj"):
            assert absorbed not in names, (
                f"{absorbed} should be absorbed into its fused row, "
                f"not a row of its own"
            )

    def test_module_row_aggregation_id_matches_the_timing_row(self):
        """One fused kernel has ONE identity: the table row's
        aggregation_id equals its per-op timing row's (the parallel-fusion
        anchor, not the sink)."""
        from backend.inference.capacity import estimate_capacity
        from backend.inference.parallel_config import ParallelConfig
        cfg = snapshot_config("deepseek-ai/DeepSeek-V3")
        resolve_snapshot("deepseek-ai/DeepSeek-V3")
        result = estimate_capacity(
             config=cfg, gpu_sku="H200", num_gpus=8,
            avg_input_tokens=2048, avg_output_tokens=128,
            # A deployment that holds the weights: one that does not runs
            # no step, and publishes no table (`capacity.without_step`).
            parallel=ParallelConfig(dp_attention=8, ep_moe=8),
            model_id="deepseek-ai/DeepSeek-V3",
            selection_mode="theoretical_only",
        )
        rows = result["phases"]["decode"]["flops_by_module"]
        timing = {
            t.get("uid"): t for t in (result.get("per_op_timing_decode") or [])
        }
        for r in rows:
            uid = r.get("uid")
            if uid is None or uid not in timing:
                continue
            assert r["aggregation_id"] == timing[uid]["aggregation_id"], (
                f"uid {uid}: table row id {r['aggregation_id']!r} != "
                f"timing row id {timing[uid]['aggregation_id']!r}"
            )

    def test_per_op_timing_contains_vllm_aligned_fusion_ops(self):
        """End-to-end: post-fusion per_op rows surface the 4 vLLM-
        aligned fused op names (flash_attention / silu_and_mul /
        merged_column_parallel_linear / fused_add_rmsnorm)."""
        from backend.inference.capacity import estimate_capacity
        from backend.inference.parallel_config import ParallelConfig
        cfg = snapshot_config("deepseek-ai/DeepSeek-V3")
        resolve_snapshot("deepseek-ai/DeepSeek-V3")
        result = estimate_capacity(
             config=cfg, gpu_sku="H200", num_gpus=8,
            avg_input_tokens=2048, avg_output_tokens=128,
            # A deployment that holds the weights: one that does not runs
            # no step, and publishes no table (`capacity.without_step`).
            parallel=ParallelConfig(dp_attention=8, ep_moe=8),
            model_id="deepseek-ai/DeepSeek-V3",
            selection_mode="theoretical_only",
        )
        ops = {r["op"] for r in result["per_op_timing"]}
        expected = {
            "flash_attention",
            "silu_and_mul",
            "merged_column_parallel_linear",
            "fused_add_rmsnorm",
        }
        missing = expected - ops
        assert not missing, (
            f"vLLM-aligned fused ops missing from per_op_timing: "
            f"{sorted(missing)}. saw ops: {sorted(ops)[:20]}"
        )


class TestMoEActiveExpertSaturation:
    """MoE expert leaves carry `active_ratio_per_token = K/E` (top-K
    active out of E stored). Per-step HBM weight reads saturate at
    `min(B × S × per_token_active, weight_total)` — small batch reads
    less than full bank (only some experts touched), large batch hits
    the ceiling. Non-MoE leaves carry `active_ratio_per_token = 1.0`
    so the saturation collapses to `weight_total`.

    Skipped unless a real MoE snapshot is on disk (DeepSeek-V3)."""

    def _resolve_dsv3(self):
        cfg = snapshot_config("deepseek-ai/DeepSeek-V3")
        resolved = resolve_snapshot("deepseek-ai/DeepSeek-V3")
        return cfg, resolved

    def test_moe_expert_leaf_active_ratio_equals_K_over_E(self):
        from backend.inference.pipeline import analyze
        from backend.inference.parallel_config import ParallelConfig
        _cfg, resolved = self._resolve_dsv3()
        ar = analyze(
            resolved, parallel=ParallelConfig(),
            batch_size=1, avg_input_tokens=1, avg_output_tokens=0,
            peak_flops_per_sec=989e12, peak_hbm_bw_bytes_per_sec=3350e9,
            breakdown=True,
        )
        # DSv3 has E=256 routed experts, K=8 top-active per token.
        # Walk must propagate K/E = 1/32 ≈ 0.03125 to expert-leaf rows.
        from backend.arch.spec_id_compose import split_aggregation_id
        def _is_expert_proj(r):
            _, sid = split_aggregation_id(r["aggregation_id"])
            return sid.startswith("mlp.experts.") and sid.endswith("_proj")
        expert_rows = [r for r in ar.per_op_timing if _is_expert_proj(r)]
        assert len(expert_rows) > 0, "expected MoE expert leaves in per_op"
        for r in expert_rows:
            ratio = r["active_ratio_per_token"]
            assert 0.025 < ratio < 0.05, (
                f"{r['aggregation_id']} active_ratio={ratio:.4f}; expected ≈ K/E "
                f"(8/256 = 0.03125). Off by ~32× usually means the walk "
                f"compounded ratios across nested MoE wrappers — should "
                f"use marginal own_ratio/child_aggregated_ratio."
            )

    def test_non_moe_leaf_active_ratio_is_one(self):
        from backend.inference.pipeline import analyze
        from backend.inference.parallel_config import ParallelConfig
        _cfg, resolved = self._resolve_dsv3()
        ar = analyze(
            resolved, parallel=ParallelConfig(),
            batch_size=1, avg_input_tokens=1, avg_output_tokens=0,
            peak_flops_per_sec=989e12, peak_hbm_bw_bytes_per_sec=3350e9,
            breakdown=True,
        )
        # Non-MoE leaves (attention linears, etc.) must carry ratio=1.0
        # so the saturation min collapses to weight_total (full read).
        from backend.arch.spec_id_compose import split_aggregation_id
        non_moe_specs = ["self_attn.q_b_proj", "self_attn.o_proj"]
        for sid in non_moe_specs:
            rows = [
                r for r in ar.per_op_timing
                if split_aggregation_id(r["aggregation_id"])[1] == sid
            ]
            if not rows:
                continue
            assert rows[0]["active_ratio_per_token"] == 1.0, (
                f"non-MoE leaf {sid} has active_ratio="
                f"{rows[0]['active_ratio_per_token']!r}; must be 1.0"
            )

    def test_moe_weight_saturates_at_large_batch(self):
        """Σ across MoE expert leaves at small B (B×S×K << E) gives
        ~K/E × full_bank; at large B (B×S×K >> E) gives ~full_bank.
        The ratio (large / small) on the MoE part should approach
        E/K = 32× (with noise from activation R/W which scales
        linearly with tokens — dominates at large B/S)."""
        from backend.inference.pipeline import analyze
        from backend.inference.parallel_config import ParallelConfig
        _cfg, resolved = self._resolve_dsv3()
        small = analyze(
            resolved, parallel=ParallelConfig(),
            batch_size=1, avg_input_tokens=1, avg_output_tokens=0,
            peak_flops_per_sec=989e12, peak_hbm_bw_bytes_per_sec=3350e9,
            breakdown=True,
        )
        large = analyze(
            resolved, parallel=ParallelConfig(),
            batch_size=64, avg_input_tokens=2048, avg_output_tokens=0,
            peak_flops_per_sec=989e12, peak_hbm_bw_bytes_per_sec=3350e9,
            breakdown=True,
        )

        from backend.arch.spec_id_compose import split_aggregation_id
        def expert_access_sum(ar):
            total = 0
            for r in ar.per_op_timing:
                _, sid = split_aggregation_id(r["aggregation_id"])
                if sid.startswith("mlp.experts.") and sid.endswith("_proj"):
                    total += r["access_bytes_per_rank"]
            return total

        small_sum = expert_access_sum(small)
        large_sum = expert_access_sum(large)
        assert large_sum > 10 * small_sum, (
            f"large-batch MoE access ({large_sum:,}) must be much "
            f"larger than small-batch ({small_sum:,}) — both saturation "
            f"of weights and activation scaling contribute. If ratio < 10× "
            f"the MoE saturation logic isn't kicking in."
        )


# ---------------------------------------------------------------------------
# Request boundary: prefer_measured toggle
# ---------------------------------------------------------------------------


class TestRequestSelectionModeResolution:
    """`InferenceEstimateRequest.resolved_selection_mode()` is the
    single translation point from user-facing `prefer_measured` toggle
    to internal mode string. The 4 user-visible cases are pinned here."""

    def _req(self, **kwargs):
        from backend.api._models import InferenceEstimateRequest
        return InferenceEstimateRequest(
            model_id="x", gpu_sku="H100-SXM", **kwargs,
        )

    def test_prefer_measured_true_resolves_to_envelope(self):
        assert self._req(prefer_measured=True).resolved_selection_mode() \
            == "envelope"

    def test_prefer_measured_false_resolves_to_theoretical_only(self):
        assert self._req(prefer_measured=False).resolved_selection_mode() \
            == "theoretical_only"

    def test_explicit_selection_mode_wins_for_backcompat(self):
        """Legacy callers still pass `selection_mode` directly; resolver
        must honor it over `prefer_measured`."""
        req = self._req(prefer_measured=True, selection_mode="latest")
        assert req.resolved_selection_mode() == "latest"

    def test_kernel_mix_populated_resolves_to_kernel_mix(self):
        """A populated `kernel_mix` is unambiguous user intent → use
        the kernel_mix mode without requiring explicit
        `selection_mode='kernel_mix'`."""
        req = self._req(kernel_mix={"flash_attention": "v3"})
        assert req.resolved_selection_mode() == "kernel_mix"


# ---------------------------------------------------------------------------
# Materialize-as-source-of-truth — CLAUDE.md §2.3 invariant
# ---------------------------------------------------------------------------


class TestAnalysisOwnsScaleDtype:
    """capacity.estimate_capacity reads scale dtype from Analysis."""

    def test_helper_lives_on_analysis(self):
        from backend.arch.eval.analysis import Analysis
        assert callable(Analysis.dominant_weight_scale_dtype)
