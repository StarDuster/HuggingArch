"""Tests for the op-level calibration overlay.

Covers exact-hit, exact-miss, stale spec_hash, different framework_version,
mode-aware peak-drift compensation (compute-bound + memory-bound),
current-peak equality (`t_op == measured` exactly), anomaly warning fires
below threshold, no silent clamping (transparency), and that the
registry mirrors the validator (rows the CLI rejects cannot become
calibration hits).
"""
from __future__ import annotations

import csv
import io
import json

import pytest

import tests.corpus_row as _corpus

from backend.inference import benchmarks as _benchmarks
from backend.inference.op_calibration import (
    ANOMALY_THRESHOLD,
    BenchmarkKey,
    CalibrationRegistry,
    MeasuredTiming,
    PhysicalKey,
    measured_time_for,
    merge_validated_rows,
    physical_time_for,
    select_time,
)


def _base_csv_row(**over) -> dict:
    """One shared row — see `tests/corpus_row.py`."""
    return _corpus.row(
        model="meta-llama/Llama-3.2-3B", spec_hash="llama32_3b_v1",
        parallel="tp=1", framework_version="0.6.3", node_spec_id="self_attn",
        **over)


def _registry_with_row(row: dict) -> CalibrationRegistry:
    """Build a one-row registry through the production indexer.

    This used to hand-build the `_Row` field by field after validating the
    text, which made it a SECOND construction path: a column the real
    indexer started carrying was simply absent here, so a test asserting on
    it could not fail. `merge_validated_rows` is the path posted rows take,
    and it indexes through the same `_index_row` the shared corpus does.
    """
    return merge_validated_rows(CalibrationRegistry(), [dict(row)],
                                source="<test>", origin="shared")


def _key_from(row: dict) -> BenchmarkKey:
    return BenchmarkKey(**{c: row[c] for c in _benchmarks.IDENTITY_COLUMNS})


class TestModuleConstants:
    def test_anomaly_threshold_is_0_95(self):
        assert ANOMALY_THRESHOLD == 0.95


class TestExactMatchHit:
    def test_hit_returns_measured_timing_with_correct_t_op_s(self):
        row = _base_csv_row()
        reg = _registry_with_row(row)
        result = measured_time_for(
            reg, _key_from(row),
            current_peak_tflops_at=lambda _dtype: 989.0,
            current_peak_bw_gbps=3350.0,
        )
        assert isinstance(result, MeasuredTiming)
        # current peaks equal test peaks → t_op_s == latency exactly
        assert result.t_op_s == pytest.approx(387.5e-6, abs=0.0)
        assert result.source == "measured"
        assert result.anomaly_reason is None

    def test_badge_carries_framework_metadata(self):
        row = _base_csv_row()
        reg = _registry_with_row(row)
        result = measured_time_for(
            reg, _key_from(row),
            current_peak_tflops_at=lambda _dtype: 989.0,
            current_peak_bw_gbps=3350.0,
        )
        b = result.source_badge
        assert b["framework"] == "vllm"
        assert b["framework_version"] == "0.6.3"
        assert b["kernel_impl"] == "flash_attention_v3"
        assert b["contributor"] == "@huggingarch-seed"
        assert b["date"] == "2026-05-16"

    def test_source_anchor_threaded_when_provided(self):
        row = _base_csv_row()
        reg = _registry_with_row(row)
        result = measured_time_for(
            reg, _key_from(row),
            current_peak_tflops_at=lambda _dtype: 989.0,
            current_peak_bw_gbps=3350.0,
            source_anchor={"text": "FlashAttentionImpl", "segments": []},
        )
        assert result.source_badge["source_anchor"] == (
            {"text": "FlashAttentionImpl", "segments": []}
        )


class TestBadgeSaysWhichDriverRanIt:
    """The driver pair is PROVENANCE, so selection does not split on it — the
    row still supersedes. The badge is therefore the only thing that can keep
    the reader from mistaking a number measured through a substituted
    user-mode driver for one measured natively, and on this repo's H100 box
    that substitution is outside NVIDIA's supported matrix.
    """

    def _badge(self, **over) -> dict:
        row = _base_csv_row(**over)
        return measured_time_for(
            _registry_with_row(row), _key_from(row),
            current_peak_tflops_at=lambda _dtype: 989.0,
            current_peak_bw_gbps=3350.0,
        ).source_badge

    def test_the_badge_names_the_driver_that_executed_the_kernel(self):
        """The USER-mode driver, not the box's. The two values differ here so
        a badge wired to the wrong column is red: `575.57.08` is the kernel
        driver and naming it would describe something that did not run the
        kernel."""
        b = self._badge(kmd_version="575.57.08", umd_version="580.95.05")
        assert b["driver"] == "580.95.05"

    def test_a_substituted_driver_is_flagged_with_both_halves(self):
        """Excludes a badge that records only the driver: `580.95.05` alone
        does not tell a reader it was not this box's own."""
        b = self._badge(kmd_version="575.57.08", umd_version="580.95.05")
        assert b["driver_substituted_for"] == "575.57.08"

    def test_a_native_row_is_not_flagged(self):
        """Excludes the "always flag" implementation. The overwhelming
        majority of rows are native, and a warning on all of them is a
        warning on none."""
        b = self._badge(kmd_version="595.84", umd_version="595.84")
        assert b["driver"] == "595.84"
        assert "driver_substituted_for" not in b

    def test_a_pre_v14_row_claims_no_driver_at_all(self):
        """Excludes both "absent reads as native" and any retrofit. A row
        measured before the corpus recorded a driver must put NOTHING on the
        badge — neither a driver nor a reassuring absence of the flag that
        a reader could mistake for a verified native configuration."""
        b = self._badge(kmd_version="", umd_version="")
        assert "driver" not in b
        assert "driver_substituted_for" not in b

    def test_a_half_observed_row_is_not_flagged_as_substituted(self):
        """One side unknown cannot support the claim. Excludes the plain
        string comparison, under which `"575.57.08" != ""` flags a
        substitution nobody observed."""
        b = self._badge(kmd_version="575.57.08", umd_version="")
        assert "driver_substituted_for" not in b
        assert "driver" not in b

    def test_the_driver_survives_the_lookup_s_column_cut(self):
        """In production the rows arrive from the bench service cut to
        `ROW_COLUMNS`, not as whole corpus rows.

        So the badge can be correct everywhere above and still be empty for
        every real request, if the driver is not on that wire list — which is
        exactly what the other assertions in this class could not see: they
        build the registry from a full row. Verified by mutation: dropping
        the two columns from `ROW_COLUMNS` left all six of them passing.
        """
        from backend.inference.op_calibration import (
            ROW_COLUMNS, registry_from_lookup,
        )
        row = _base_csv_row(kmd_version="575.57.08", umd_version="580.95.05")
        key = _key_from(row)
        cut = {c: row[c] for c in ROW_COLUMNS}
        cut["origin"] = "shared"
        reg = registry_from_lookup([key.to_physical()], [[cut]])
        result = measured_time_for(
            reg, key,
            current_peak_tflops_at=lambda _dtype: 989.0,
            current_peak_bw_gbps=3350.0,
        )
        assert result is not None, "the cut row must still index"
        assert result.source_badge["driver"] == "580.95.05"
        assert result.source_badge["driver_substituted_for"] == "575.57.08"

    def test_the_driver_does_not_enter_the_selection_key(self):
        """Two rows differing ONLY in driver are one cell, and the newer wins
        as it always did — the pair is provenance. Excludes an
        implementation that put the driver in `BenchmarkKey`, which would
        make these two separate cells that never supersede each other."""
        native = _base_csv_row(kmd_version="595.84", umd_version="595.84")
        compat = _base_csv_row(kmd_version="575.57.08", umd_version="580.95.05")
        assert _key_from(native) == _key_from(compat)


class TestExactMatchMiss:
    """No exact match → return None, no silent fallback."""

    def test_missing_key_returns_none(self):
        row = _base_csv_row()
        reg = _registry_with_row(row)
        miss_key = _key_from({**row, "node_spec_id": "different_node"})
        assert measured_time_for(
            reg, miss_key,
            current_peak_tflops_at=lambda _dtype: 989.0,
            current_peak_bw_gbps=3350.0,
        ) is None

    def test_empty_registry_always_misses(self):
        reg = CalibrationRegistry()
        assert measured_time_for(
            reg, _key_from(_base_csv_row()),
            current_peak_tflops_at=lambda _dtype: 989.0,
            current_peak_bw_gbps=3350.0,
        ) is None


class TestStaleIdentityRejection:
    """Different spec_hash → distinct row (stale-join
    rejection works at lookup time, not by accidental collision)."""

    def test_different_spec_hash_is_distinct(self):
        row = _base_csv_row()
        reg = _registry_with_row(row)
        stale_key = _key_from({**row, "spec_hash": "llama32_3b_v2"})
        assert measured_time_for(
            reg, stale_key,
            current_peak_tflops_at=lambda _dtype: 989.0,
            current_peak_bw_gbps=3350.0,
        ) is None

    def test_different_framework_version_is_distinct(self):
        row = _base_csv_row()
        reg = _registry_with_row(row)
        diff_ver_key = _key_from({**row, "framework_version": "0.7.0"})
        assert measured_time_for(
            reg, diff_ver_key,
            current_peak_tflops_at=lambda _dtype: 989.0,
            current_peak_bw_gbps=3350.0,
        ) is None

    def test_different_shape_is_distinct(self):
        """Plan Update 1 for M3: shape_json is part of identity."""
        row = _base_csv_row()
        reg = _registry_with_row(row)
        diff_shape = json.dumps(
            {"B": 1, "s_ctx": 4096, "n_heads": 24, "d_head": 128}, separators=(",", ":")
        )
        diff_shape_key = _key_from({**row, "shape_json": diff_shape})
        assert measured_time_for(
            reg, diff_shape_key,
            current_peak_tflops_at=lambda _dtype: 989.0,
            current_peak_bw_gbps=3350.0,
        ) is None


class TestPeakDriftCompute:
    """Compute-bound row scales by TFLOPs ratio."""

    def test_compute_bound_row_scales_by_tflops_ratio(self):
        row = _base_csv_row()
        # achieved_tflops/peak = 690/989 ≈ 0.697; achieved_bw/peak = 920/3350 ≈ 0.275
        # → compute-bound
        reg = _registry_with_row(row)
        result = measured_time_for(
            reg, _key_from(row),
            current_peak_tflops_at=lambda _dtype: 1500.0,  # 1.5x faster compute
            current_peak_bw_gbps=3350.0,
        )
        # Faster current peak → SHORTER latency. ratio = test/current.
        expected_t_us = 387.5 * (989.0 / 1500.0)
        assert result.t_op_s == pytest.approx(expected_t_us * 1e-6, rel=1e-9)
        assert result.source_badge["bound"] == "compute"


class TestPeakDriftMemory:
    """Memory-bound row scales by bandwidth ratio."""

    def test_memory_bound_row_scales_by_bandwidth_ratio(self):
        row = _base_csv_row()
        # Make memory dominate by zeroing-out compute saturation
        row["achieved_tflops"] = "10.0"  # tiny compute utilization
        row["achieved_bw_gbps"] = "3200.0"  # near-peak bandwidth
        reg = _registry_with_row(row)
        result = measured_time_for(
            reg, _key_from(row),
            current_peak_tflops_at=lambda _dtype: 989.0,
            current_peak_bw_gbps=5000.0,  # 1.49x faster memory
        )
        # Faster current peak → SHORTER latency. ratio = test/current.
        expected_t_us = 387.5 * (3350.0 / 5000.0)
        assert result.t_op_s == pytest.approx(expected_t_us * 1e-6, rel=1e-9)
        assert result.source_badge["bound"] == "memory"


class TestPeakDriftDirection:
    """A faster current peak should produce a SHORTER calibrated
    latency, not a longer one. An inverted formula like
    `latency × current_peak / test_peak` would have a 2× faster GPU
    predict 2× slower."""

    def test_double_compute_peak_halves_latency(self):
        # Compute-bound row at peak_tflops_at_test=989.
        row = _base_csv_row()
        reg = _registry_with_row(row)
        # current peak = 2× test peak → t_op should be 0.5× measured.
        result = measured_time_for(
            reg, _key_from(row),
            current_peak_tflops_at=lambda _dtype: 1978.0,  # 2 × 989
            current_peak_bw_gbps=3350.0,
        )
        assert result.t_op_s == pytest.approx(387.5e-6 * 0.5, rel=1e-9)

    def test_drift_reads_the_rows_own_mma_family(self):
        # An FP8 GEMM records the card's FP8 peak. Applied to the same card it
        # has not drifted: the current peak is read at the row's dtype, not
        # the step's model dtype (a BF16 comparator would halve nothing and
        # double the FP8 row's time).
        row = _base_csv_row()
        row["dtype_env_json"] = json.dumps(
            {**json.loads(row["dtype_env_json"]),
             "activation": "fp8_e4m3", "weight": "fp8_e4m3"})
        row["peak_tflops_at_test"] = "1979.0"
        row["achieved_tflops"] = "1300.0"
        reg = _registry_with_row(row)
        peaks = {"fp8_e4m3": 1979.0, "bf16": 989.0}
        result = measured_time_for(
            reg, _key_from(row),
            current_peak_tflops_at=peaks.get,
            current_peak_bw_gbps=3350.0,
        )
        assert result.source_badge["bound"] == "compute"
        assert result.t_op_s == pytest.approx(387.5e-6, rel=1e-12)

    def test_half_memory_peak_doubles_latency(self):
        # Memory-bound row at peak_bw_at_test=3350.
        row = _base_csv_row()
        row["achieved_tflops"] = "10.0"
        row["achieved_bw_gbps"] = "3200.0"
        reg = _registry_with_row(row)
        # current peak = 0.5× test peak → t_op should be 2× measured.
        result = measured_time_for(
            reg, _key_from(row),
            current_peak_tflops_at=lambda _dtype: 989.0,
            current_peak_bw_gbps=1675.0,  # 0.5 × 3350
        )
        assert result.t_op_s == pytest.approx(387.5e-6 * 2.0, rel=1e-9)


class TestCurrentPeakEquality:
    """When current peaks equal test peaks, t_op == measured
    exactly (no scaling factor distorts the displayed value)."""

    @pytest.mark.parametrize("bound_setup", [
        # (achieved_tflops, achieved_bw_gbps) tuned to land in each bound
        (690.0, 920.0),     # compute-bound
        (10.0, 3200.0),     # memory-bound
    ])
    def test_equal_peaks_pin_latency_exactly(self, bound_setup):
        a_tflops, a_bw = bound_setup
        row = _base_csv_row()
        row["achieved_tflops"] = str(a_tflops)
        row["achieved_bw_gbps"] = str(a_bw)
        reg = _registry_with_row(row)
        result = measured_time_for(
            reg, _key_from(row),
            current_peak_tflops_at=lambda _dtype: 989.0,
            current_peak_bw_gbps=3350.0,
        )
        # t_op_s must equal latency_us * 1e-6 with abs=0.0
        assert result.t_op_s == pytest.approx(387.5e-6, abs=0.0)


class TestAnomalyWarning:
    """When measured/roofline_at_test < 0.95, source flips to
    measured_anomalous + anomaly_reason populated. The returned t_op_s
    is STILL the measured value (no silent clamping)."""

    def test_anomalous_measurement_flags_source(self):
        row = _base_csv_row()
        # Force the measurement to look impossibly fast: pretend the op
        # has huge flops, so roofline_at_test >> measured.
        reg = _registry_with_row(row)
        result = measured_time_for(
            reg, _key_from(row),
            current_peak_tflops_at=lambda _dtype: 989.0,
            current_peak_bw_gbps=3350.0,
            flops_for_anomaly=1e15,        # gigantic
            access_bytes_for_anomaly=1e12, # also large
        )
        assert result.source == "measured_anomalous"
        assert result.anomaly_reason is not None
        assert "ANOMALY_THRESHOLD" in result.anomaly_reason

    def test_anomalous_t_op_s_is_not_clamped_to_roofline(self):
        """Transparency: even when anomalous, t_op_s reflects the
        measured value (with peak-drift compensation), never the roofline."""
        row = _base_csv_row()
        reg = _registry_with_row(row)
        result = measured_time_for(
            reg, _key_from(row),
            current_peak_tflops_at=lambda _dtype: 989.0,
            current_peak_bw_gbps=3350.0,
            flops_for_anomaly=1e15,
            access_bytes_for_anomaly=1e12,
        )
        # t_op_s must equal latency_us * 1e-6, NOT the roofline-at-test.
        assert result.t_op_s == pytest.approx(387.5e-6, abs=0.0)

    def test_non_anomalous_measurement_keeps_source_measured(self):
        row = _base_csv_row()
        reg = _registry_with_row(row)
        result = measured_time_for(
            reg, _key_from(row),
            current_peak_tflops_at=lambda _dtype: 989.0,
            current_peak_bw_gbps=3350.0,
            flops_for_anomaly=1e9,        # small enough that roofline is well below measured
            access_bytes_for_anomaly=1e6,
        )
        assert result.source == "measured"
        assert result.anomaly_reason is None
        assert "anomaly_reason" not in result.source_badge


class TestAnomalyOnlyFiresWithFlopsContext:
    """When the caller doesn't supply flops/access_bytes, anomaly check
    is skipped (the function has nothing to compare against)."""

    def test_no_anomaly_check_without_context(self):
        row = _base_csv_row()
        reg = _registry_with_row(row)
        result = measured_time_for(
            reg, _key_from(row),
            current_peak_tflops_at=lambda _dtype: 989.0,
            current_peak_bw_gbps=3350.0,
        )
        assert result.source == "measured"
        assert result.anomaly_reason is None


class TestMergeValidatedRows:
    """`merge_validated_rows` overlays request-supplied rows (e.g. a
    logged-in user's private corpus) onto a base registry."""

    def test_merge_into_empty_registry(self):
        reg = CalibrationRegistry()
        merge_validated_rows(reg, [_base_csv_row()], source="request")
        assert len(reg) == 1
        assert _key_from(_base_csv_row()) in reg.rows_by_key

    def test_empty_rows_noop(self):
        reg = _registry_with_row(_base_csv_row())
        before = len(reg)
        merge_validated_rows(reg, [], source="request")
        assert len(reg) == before

    def test_request_row_wins_on_identity_collision(self):
        # Base (shared corpus) row and a request row with the SAME identity
        # but a different latency: the request row must override.
        base = _base_csv_row()
        reg = _registry_with_row(base)
        assert reg.rows_by_key[_key_from(base)].latency_us == pytest.approx(387.5)
        override = dict(base)
        override["latency_us"] = "123.0"
        merge_validated_rows(reg, [override], source="request")
        assert len(reg) == 1  # same identity → replaced, not appended
        assert reg.rows_by_key[_key_from(base)].latency_us == pytest.approx(123.0)

    def test_merge_stamps_mine_origin(self):
        # Request rows are the user's own → origin "mine", surfaced on the
        # measured badge for per-row attribution.
        reg = CalibrationRegistry()
        merge_validated_rows(reg, [_base_csv_row()], source="request")
        assert reg.rows_by_key[_key_from(_base_csv_row())].origin == "mine"

    def test_an_indexed_row_defaults_shared(self):
        # No origin named → "shared".
        from backend.inference.op_calibration import _index_row
        reg = CalibrationRegistry()
        _index_row(reg, _base_csv_row(), source="t", on_collision="override")
        assert reg.rows_by_key[_key_from(_base_csv_row())].origin == "shared"

    def test_corpus_rows_carry_utilization(self):
        # Each corpus row states its MFU / MBU / arithmetic intensity: the
        # achieved rates over the peaks recorded at test time, so the
        # explorer shows them without dividing anything itself.
        from backend.inference.op_calibration import _index_row
        row = dict(_base_csv_row(), achieved_tflops="200", peak_tflops_at_test="800",
                   achieved_bw_gbps="1500", peak_bw_at_test="3000")
        reg = CalibrationRegistry()
        _index_row(reg, row, source="t", on_collision="override")
        out = next(iter(reg.rows_by_key.values())).utilization()
        assert out["mfu"] == pytest.approx(0.25)
        assert out["mbu"] == pytest.approx(0.5)
        assert out["arith_intensity"] == pytest.approx(200 * 1000 / 1500)
        assert out["roof_status"] == "ok"

    def test_a_row_faster_than_its_peak_is_flagged(self):
        # A measurement beating the peak it recorded names the ratio(s) over 1.
        from backend.inference.op_calibration import measured_utilization
        row = {"achieved_tflops": 900, "peak_tflops_at_test": 800,
               "achieved_bw_gbps": 1500, "peak_bw_at_test": 3000}
        assert measured_utilization(row)["roof_status"] == "mfu"
        row["achieved_bw_gbps"] = 3100
        assert measured_utilization(row)["roof_status"] == "mfu+mbu"
        assert measured_utilization({})["roof_status"] is None

    def test_invalid_request_row_raises(self):
        reg = CalibrationRegistry()
        bad = dict(_base_csv_row())
        bad["latency_us"] = "not-a-number"
        with pytest.raises(_benchmarks.BenchmarkValidationError):
            merge_validated_rows(reg, [bad], source="request")

    def test_json_columns_accepted_as_objects(self):
        # Rows coming back from Supabase JSONB carry shape_json /
        # dtype_env_json as parsed objects, not strings. merge must
        # normalize them rather than reject.
        row = _base_csv_row()
        row["shape_json"] = {"B": 1, "s_ctx": 2048, "n_heads": 24, "d_head": 128}
        row["dtype_env_json"] = {
            "output": "bf16", "activation": "bf16", "kv": "bf16",
        }
        reg = CalibrationRegistry()
        merge_validated_rows(reg, [row], source="request")
        assert len(reg) == 1
        # Indexed under the CANONICAL string identity — the producers sort, so
        # normalization must too. It did not, and this assertion passed anyway
        # because the fixture happened to spell the dict in the same unsorted
        # order the test did. A JSONB row in any other order was stored under a
        # key nothing looking it up would ever build.
        assert _key_from(_base_csv_row()) in reg.rows_by_key
        key = next(iter(reg.rows_by_key))
        assert key.shape_json == json.dumps(
            json.loads(key.shape_json), separators=(",", ":"), sort_keys=True)
        assert key.dtype_env_json == json.dumps(
            json.loads(key.dtype_env_json), separators=(",", ":"), sort_keys=True)

    def test_enables_measured_lookup_without_a_service(self):
        # Request rows enable measured lookup even with no Kernel Bench
        # service (the scope reads nothing); without them, nothing is read.
        from backend.measure.scope import NONE
        from backend.inference.op_calibration import registry_for
        from backend.inference.capacity import _resolve_overlay_context
        (_fw, _bind, _strat, enabled) = _resolve_overlay_context(
            framework="vllm", selection_mode="latest",
            measurement_scope=NONE,
            calibration_rows=[_base_csv_row()],
        )
        assert enabled is True
        reg = registry_for(NONE, [], [_base_csv_row()])
        assert len(reg) == 1
        assert _resolve_overlay_context(
            framework="vllm", selection_mode="latest",
            measurement_scope=NONE)[-1] is False
        assert _resolve_overlay_context(
            framework="vllm", selection_mode="theoretical_only",
            calibration_rows=[_base_csv_row()])[-1] is False


class TestLatestUsesDeclaredProviderIdentity:
    """Latest separates the serving-framework declaration holder from the
    provider library on a corpus row, and never weakens an implementation id
    to its base name."""

    def _registry(self):
        torch_row = _base_csv_row(
            framework="torch", kernel_impl="torch_cublas_linear",
            latency_us="300.0")
        vllm_row = _base_csv_row(
            framework="vllm", kernel_impl="vllm_flash_attention_v3",
            latency_us="500.0")
        return merge_validated_rows(
            _registry_with_row(torch_row), [vllm_row],
            source="<test>", origin="shared")

    def _select(self, registry, dispatch, **over):
        peaks = dict(current_peak_tflops_at=lambda _dtype: 989.0,
                     current_peak_bw_gbps=3350.0)
        key = _key_from(_base_csv_row()).to_physical()
        return select_time(
            registry, key, strategy="latest", framework="vllm",
            kernel_dispatch=dispatch, **peaks, **over)

    def test_a_vllm_dispatch_selects_a_declared_torch_row(self):
        got, state = self._select(self._registry(), {
            "state": "declared", "library": "torch",
            "impl_id": "torch_cublas_linear"})
        assert state == "measured"
        assert got.t_op_s == pytest.approx(300.0e-6)
        assert got.source_badge["framework"] == "torch"
        assert got.source_badge["serving_framework"] == "vllm"
        assert got.source_badge["declared_kernel_impl"] == "torch_cublas_linear"

    def test_no_declaration_is_ambiguous_even_when_a_vllm_row_exists(self):
        for dispatch in ({"state": "undeclared"},
                         {"state": "no_active"},
                         {"state": "ambiguous"}):
            got, state = self._select(self._registry(), dispatch)
            assert got is None and state == "ambiguous"

    def test_declared_but_absent_exact_identity_is_missing(self):
        dispatch = {"state": "declared", "library": "torch",
                    "impl_id": "torch_cublas_linear@compute=fp32"}
        got, state = self._select(self._registry(), dispatch)
        assert got is None and state == "missing"

    def test_provider_library_is_part_of_exact_identity(self):
        dispatch = {"state": "declared", "library": "vllm",
                    "impl_id": "torch_cublas_linear"}
        got, state = self._select(self._registry(), dispatch)
        assert got is None and state == "missing"

    def test_an_explicit_pin_remains_an_explicit_pin(self):
        got, state = self._select(
            self._registry(), None, kernel_impl="vllm_flash_attention_v3")
        assert state == "measured"
        assert got.source_badge["kernel_pin"] == "vllm_flash_attention_v3"

    def test_a_pin_can_name_library_and_impl_as_one_identity(self):
        rows = [
            _base_csv_row(framework="torch", kernel_impl="shared_impl",
                          latency_us="300.0"),
            _base_csv_row(framework="vllm", kernel_impl="shared_impl",
                          latency_us="500.0"),
        ]
        registry = merge_validated_rows(
            CalibrationRegistry(), rows, source="<test>", origin="shared")
        got, state = self._select(
            registry, {"state": "undeclared"},
            kernel_impl="vllm:shared_impl")
        assert state == "measured"
        assert got.t_op_s == pytest.approx(500.0e-6)
        assert got.source_badge["framework"] == "vllm"
        assert got.source_badge["kernel_pin"] == "vllm:shared_impl"

        with pytest.raises(ValueError, match="more than one provider library"):
            self._select(registry, {"state": "undeclared"},
                         kernel_impl="shared_impl")

        declared = {"state": "declared", "library": "torch",
                    "impl_id": "torch_cublas_linear"}
        _got, wrong_library = self._select(
            self._registry(), declared,
            kernel_impl="vllm:torch_cublas_linear")
        assert wrong_library == "missing"
        got, right_pair = self._select(
            self._registry(), declared,
            kernel_impl="torch:torch_cublas_linear")
        assert right_pair == "measured"
        assert got.source_badge["declared_kernel_impl"] == (
            "torch_cublas_linear")


class TestPhysicalKeyEnvelope:
    """The physical (library-agnostic) key + envelope lookup: match on
    op + op-relevant shape + dtype + gpu/phase/parallel, ignore framework
    / version / kernel_impl / model / node_spec_id, pick the fastest."""

    def _row_for(self, base: dict) -> dict:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(_benchmarks.ALL_COLUMNS)
        w.writerow([base[c] for c in _benchmarks.ALL_COLUMNS])
        _h, rows = _benchmarks.validate_text(buf.getvalue(), source="<t>")
        return rows[0]

    def test_projection_drops_model_only_dims(self):
        # flash_attention shape projects to B,S,N,Nk,D — H/I/L/V dropped.
        base = _base_csv_row()
        pk = BenchmarkKey.from_csv_row(base).to_physical()
        dims = json.loads(pk.shape_json)
        assert set(dims) == {"B", "s_ctx", "n_heads", "d_head"}  # base row has no Nk key
        assert "d_model" not in dims and "n_vocab" not in dims

    def test_same_attn_dims_different_model_share_physical_key(self):
        a = _base_csv_row()
        b = _base_csv_row()
        # Different model / spec_hash / node_spec_id, but same attention
        # shape → SAME physical key (cross-model reuse).
        b["model"] = "other/Model-7B"
        b["spec_hash"] = "totally_different_hash"
        b["node_spec_id"] = "block|other.attn"
        ka = BenchmarkKey.from_csv_row(a).to_physical()
        kb = BenchmarkKey.from_csv_row(b).to_physical()
        assert ka == kb

    def test_envelope_picks_fastest_across_libraries(self):
        # Two libraries measure the same physical op; envelope returns the
        # faster one and ignores framework/kernel_impl in matching.
        slow = _base_csv_row()
        slow["framework"] = "vllm"
        slow["kernel_impl"] = "flash_attention_v2"
        slow["latency_us"] = "500.0"
        fast = _base_csv_row()
        fast["framework"] = "flashinfer"
        fast["kernel_impl"] = "flashinfer_ragged_prefill"
        fast["latency_us"] = "300.0"

        reg = CalibrationRegistry()
        merge_validated_rows(reg, [slow, fast], source="<t>")
        pk = BenchmarkKey.from_csv_row(slow).to_physical()
        assert len(reg.get_physical(pk)) == 2
        res = physical_time_for(
            reg, pk, current_peak_tflops_at=lambda _dtype: 989.0, current_peak_bw_gbps=3350.0,
        )
        assert res is not None
        assert res.t_op_s == pytest.approx(300.0e-6, abs=0.0)
        assert res.source_badge["selection_mode"] == "envelope"
        assert res.source_badge["envelope_candidates"] == "2"
        assert "flashinfer" in res.source_badge["envelope_winner"]

    def test_envelope_miss_returns_none(self):
        reg = CalibrationRegistry()
        pk = PhysicalKey(signature="gqa_attention", shape_json="{}",
                         dtype_env_json="{}", gpu="H100-SXM",
                         phase="prefill")
        assert physical_time_for(
            reg, pk, current_peak_tflops_at=lambda _dtype: 989.0, current_peak_bw_gbps=3350.0,
        ) is None


_PREFILL_SHAPE_IN = [[1, 24, 2048, 128], [1, 8, 2048, 128], [1, 8, 2048, 128]]
_DECODE_SHAPE_IN = [[1, 24, 1, 128], [1, 8, 2176, 128], [1, 8, 2176, 128]]


class TestShapeInIdentity:
    """Identity/physical shape derived from the engine's per-op `shape_in`
    (Sq/Lq-aware). Attention layout: Q=[B,N,Sq,D], K=[B,Nk,Lq,D]."""

    def _row_with(self, *, shape_in, framework="vllm", kernel="fa", latency="300.0"):
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        base = _base_csv_row()
        base["framework"] = framework
        base["kernel_impl"] = kernel
        base["latency_us"] = latency
        base["shape_json"] = identity_shape_from_shape_in(
            "flash_attention", shape_in, spec_dims={"kv_block_size": 16})
        return base

    def test_prefill_vs_decode_distinct_physical_key(self):
        pre = BenchmarkKey.from_csv_row(self._row_with(shape_in=_PREFILL_SHAPE_IN))
        dec = BenchmarkKey.from_csv_row(self._row_with(shape_in=_DECODE_SHAPE_IN))
        # Same phase field would be a crutch — the SHAPE itself (Sq=2048 vs
        # Sq=1) must already make the physical keys differ.
        assert pre.to_physical().shape_json != dec.to_physical().shape_json
        assert pre.to_physical() != dec.to_physical()

    def test_cross_model_same_attn_shape_in_shares_key(self):
        a = self._row_with(shape_in=_DECODE_SHAPE_IN)
        b = self._row_with(shape_in=_DECODE_SHAPE_IN)
        b["model"] = "other/Model-70B"
        b["spec_hash"] = "different"
        b["node_spec_id"] = "other.attn"
        assert (BenchmarkKey.from_csv_row(a).to_physical()
                == BenchmarkKey.from_csv_row(b).to_physical())

    def test_norm_act_projection_phase_aware(self):
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        import json as _json
        # rmsnorm [B,Sq,H]: prefill Sq=2048 vs decode Sq=1 → distinct
        pre = _json.loads(identity_shape_from_shape_in("rmsnorm", [1, 2048, 3072]))
        dec = _json.loads(identity_shape_from_shape_in("rmsnorm", [1, 1, 3072]))
        assert pre == {"B": 1, "s_query": 2048, "d_model": 3072}
        assert dec["s_query"] == 1 and dec["d_model"] == 3072
        # silu_and_mul list-of-tensors → {B,Sq,I}
        silu = _json.loads(identity_shape_from_shape_in(
            "silu_and_mul", [[1, 2048, 8192], [1, 2048, 8192]]))
        assert silu == {"B": 1, "s_query": 2048, "d_ffn": 8192}
        # non-projected op → None (fallback)
        assert identity_shape_from_shape_in("linear", [1, 2048, 3072]) is None

    def test_gemm_identity_from_shape_out_and_shape_outs(self):
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        import json as _json
        # N from shape_out (o_proj-style)
        a = _json.loads(identity_shape_from_shape_in(
            "linear", [1, 2048, 3072], shape_out=[1, 2048, 3072]))
        assert a == {"B": 1, "s_query": 2048, "d_reduce": 3072, "d_out": 3072}
        # N from per-port outputs when the fused node has no single tensor
        # (qkv → q/k/v): shape_out is a list-of-ports; N = 3072+1024+1024.
        b = _json.loads(identity_shape_from_shape_in(
            "qkv_parallel_linear", [1, 2048, 3072],
            shape_out=[[1, 2048, 3072], [1, 2048, 1024], [1, 2048, 1024]]))
        assert b["d_out"] == 5120 and b["d_reduce"] == 3072
        # phase-aware: decode Sq=1
        c = _json.loads(identity_shape_from_shape_in(
            "linear", [1, 1, 3072], shape_out=[1, 1, 3072]))
        assert c["s_query"] == 1
        # no shape_out and no shape_outs → None (fallback); NO flops reverse-calc
        assert identity_shape_from_shape_in("qkv_parallel_linear", [1, 2048, 3072]) is None

    def test_physical_key_uses_signature_not_fused_op(self):
        k = BenchmarkKey.from_csv_row(self._row_with(shape_in=_DECODE_SHAPE_IN))
        assert k.to_physical().signature == "gqa_attention"  # not "flash_attention"

    def test_envelope_across_frameworks_via_shape_in(self):
        slow = self._row_with(shape_in=_DECODE_SHAPE_IN,
                              framework="vllm", kernel="fa2", latency="500.0")
        fast = self._row_with(shape_in=_DECODE_SHAPE_IN,
                              framework="flashinfer", kernel="fi", latency="300.0")
        reg = CalibrationRegistry()
        merge_validated_rows(reg, [slow, fast], source="<t>")
        pk = BenchmarkKey.from_csv_row(slow).to_physical()
        assert len(reg.get_physical(pk)) == 2
        res = physical_time_for(reg, pk, current_peak_tflops_at=lambda _dtype: 989.0,
                                current_peak_bw_gbps=3350.0)
        assert res.t_op_s == pytest.approx(300.0e-6, abs=0.0)
        assert "flashinfer" in res.source_badge["envelope_winner"]


class TestMoeIdentityExpertCount:
    """The MoE rows' `n_experts` is the experts one rank runs: the bank's
    local copies under this deployment, read off its copy endpoint."""

    @staticmethod
    def _experts(parallel):
        from backend.inference.benchmark_identity import RowIdentity
        from backend.inference.framework_bindings import (
            DEFAULT_FRAMEWORK, get_binding)
        from backend.inference.pipeline import analyze
        from tests.snapshots import resolve_snapshot

        resolved = resolve_snapshot("Qwen__Qwen3-30B-A3B")
        result = analyze(resolved, parallel=parallel,
                         framework_binding=get_binding(DEFAULT_FRAMEWORK),
                         batch_size=8, avg_input_tokens=1024, avg_output_tokens=0, breakdown=True)
        identity = RowIdentity(result)
        found = {}
        for row in result.per_op_timing:
            op, shape_json = identity.of(row)
            if op in ("grouped_gemm", "permute", "combine"):
                found.setdefault(op, set()).add(
                    json.loads(shape_json)["n_experts"])
        return found

    def test_expert_parallelism_leaves_each_rank_its_share_of_the_bank(self):
        from backend.inference.parallel_config import ParallelConfig

        whole = self._experts(ParallelConfig())
        shard = self._experts(ParallelConfig(ep_moe=8, dp_attention=8))
        assert set(whole) == set(shard) == {"grouped_gemm", "permute", "combine"}
        assert all(counts == {128} for counts in whole.values())
        assert all(counts == {16} for counts in shard.values())


class TestPhysicalDedup:
    """Storage is physically-indexed: N models measuring the same physical
    op + implementation collapse to one shared envelope candidate; distinct
    implementations are kept; divergent re-measures warn."""

    def _attn_row(self, *, model, framework, kernel, latency, samples="100"):
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        r = _base_csv_row()
        r["model"] = model
        r["spec_hash"] = "h_" + model
        r["node_spec_id"] = "n_" + model
        r["framework"] = framework
        r["kernel_impl"] = kernel
        r["latency_us"] = latency
        r["p50_us"] = latency
        r["p95_us"] = latency
        r["samples"] = samples
        r["shape_json"] = identity_shape_from_shape_in(
            "flash_attention", _DECODE_SHAPE_IN, spec_dims={"kv_block_size": 16})
        return r

    def _pk(self):
        return BenchmarkKey.from_csv_row(
            self._attn_row(model="A", framework="fa3", kernel="fa3", latency="300")
        ).to_physical()

    def test_cross_model_same_impl_dedups(self):
        reg = CalibrationRegistry()
        merge_validated_rows(reg, [
            self._attn_row(model="A", framework="fa3", kernel="fa3", latency="300"),
            self._attn_row(model="B", framework="fa3", kernel="fa3", latency="305"),
        ], source="<t>")
        assert len(reg.get_physical(self._pk())) == 1   # collapsed to one
        assert len(reg.rows_by_key) == 2                 # exact mode keeps both

    def test_distinct_impls_kept_as_envelope_candidates(self):
        reg = CalibrationRegistry()
        merge_validated_rows(reg, [
            self._attn_row(model="A", framework="fa3", kernel="fa3", latency="300"),
            self._attn_row(model="A", framework="flashinfer", kernel="fi", latency="280"),
        ], source="<t>")
        assert len(reg.get_physical(self._pk())) == 2

    def test_more_samples_wins(self):
        reg = CalibrationRegistry()
        merge_validated_rows(reg, [
            self._attn_row(model="A", framework="fa3", kernel="fa3", latency="300", samples="100"),
            self._attn_row(model="B", framework="fa3", kernel="fa3", latency="305", samples="500"),
        ], source="<t>")
        cands = reg.get_physical(self._pk())
        assert len(cands) == 1
        assert cands[0].samples == 500
        assert cands[0].latency_us == pytest.approx(305.0)

    def test_divergent_remeasure_warns(self, caplog):
        import logging
        reg = CalibrationRegistry()
        with caplog.at_level(logging.WARNING):
            merge_validated_rows(reg, [
                self._attn_row(model="A", framework="fa3", kernel="fa3", latency="300"),
                self._attn_row(model="B", framework="fa3", kernel="fa3", latency="450"),
            ], source="<t>")
        assert any("diverges" in r.getMessage() for r in caplog.records)

    def test_covers(self):
        reg = CalibrationRegistry()
        merge_validated_rows(reg, [
            self._attn_row(model="A", framework="fa3", kernel="fa3", latency="300"),
        ], source="<t>")
        pk = self._pk()
        assert reg.covers(pk, framework="fa3", framework_version="0.6.3")
        assert not reg.covers(pk, framework="sglang", framework_version="0.5.6")


class TestBenchmarkKeyFromRow:
    def test_round_trip_field_count(self):
        row = _base_csv_row()
        key = BenchmarkKey.from_csv_row(row)
        # 12 identity fields exactly
        assert len(BenchmarkKey.__dataclass_fields__) == 12
        # Round-trip: building a key from a row whose identity columns
        # match the key fields must reconstruct exactly.
        assert key.model == row["model"]
        assert key.shape_json == row["shape_json"]
        assert key.dtype_env_json == row["dtype_env_json"]
        assert key.kernel_impl == row["kernel_impl"]

    @pytest.mark.parametrize("cell", [
        "", "not json", "[]",
        '{"activation":"bf16","weight":"bf16","kv":"bf16"}',
        '{"activation":"fp8_e4m3","weight":"fp8_e4m3","kv":"bf16",'
        '"output":"bf16","weight_granularity":"per-tile:128x128"}',
    ])
    def test_a_dtype_cell_that_states_no_environment_has_no_slim_key(self, cell):
        """The slim key hashes every dtype slot the schema declares. A cell
        that does not parse or leaves a slot out is refused, naming the row,
        instead of being hashed as if it had said BF16 / FP32."""
        key = _key_from(_base_csv_row(dtype_env_json=cell))
        with pytest.raises(_benchmarks.BenchmarkValidationError) as exc:
            key.to_slim()
        assert key.node_spec_id in str(exc.value)


class TestReductionIdentityIsTheAxisTheGraphSums:
    """`sum_reduce` collapses whichever axis the node names. Projected as a
    last-axis pointwise op, DSv4's hyper-connection reduce `[16,2048,4,4096] →
    [16,2048,4096]` was timed summing the 4096-wide axis of a
    `[131072, 4096]` tensor while billed for the graph's reduce."""

    def _dims(self, shape_in, shape_out):
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        got = identity_shape_from_shape_in("sum_reduce", shape_in, shape_out=shape_out)
        return None if got is None else json.loads(got)

    def test_middle_axis(self):
        assert self._dims([[16, 2048, 4, 4096]], [[16, 2048, 4096]]) == {
            "B": 16, "s_query": 2048, "d_reduce": 4, "d_model": 4096}

    def test_last_axis_has_unit_inner_extent(self):
        assert self._dims([[2, 128, 4]], [[2, 128]]) == {
            "B": 2, "s_query": 128, "d_reduce": 4, "d_model": 1}

    def test_empty_axis_is_no_cell(self):
        assert self._dims([[757, 0, 8, 512]], [[757, 0, 512]]) is None
